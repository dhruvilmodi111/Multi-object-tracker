"""Occlusion-Aware Multi-Object Tracker (OcclusionDeepSORT).
Implements multi-stage matching, lost-track buffering, appearance memory gallery,
and occlusion detection to minimize identity switches.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.tracker.track import Track, TrackState
from src.tracker.kalman_filter import KalmanFilter
from src.tracker.matching import (
    iou_cost_matrix,
    appearance_cost_matrix,
    fused_cost_matrix,
    linear_assignment
)
from src.detectors.yolo_detector import Detection
from src.utils.box_utils import xyxy_to_z, compute_inter_min_matrix, compute_iou_matrix


class OcclusionTracker:
    """Multi-object tracker tuned for robust tracking under occlusions and crossings."""

    def __init__(
        self,
        max_lost_age: int = 30,
        n_init: int = 3,
        appearance_weight: float = 0.70,
        stage1_match_thresh: float = 0.40,  # Fused cost threshold
        stage2_iou_thresh: float = 0.50,     # IoU fallback threshold (cost = 1 - IoU)
        reid_lost_thresh: float = 0.35,      # Cosine distance threshold for reviving lost tracks
        occlusion_iou_thresh: float = 0.25,  # Overlap threshold to flag crossing/occlusion
        gallery_size: int = 10,
        feature_ema_alpha: float = 0.90,
        enable_lost_reid: bool = True,
        enable_occlusion_freeze: bool = True,
    ):
        """Args:
            max_lost_age: Maximum frames to buffer a lost track before deletion
            n_init: Consecutive matches needed to confirm tentative tracks
            appearance_weight: Weight of appearance vs IoU in stage 1 association
            stage1_match_thresh: Maximum cost for stage 1 match
            stage2_iou_thresh: Maximum IoU distance (1 - IoU) for stage 2 match
            reid_lost_thresh: Cosine distance threshold for matching lost track to new detection
            occlusion_iou_thresh: Inter-track overlap threshold that marks occlusion risk
            gallery_size: Number of past feature vectors to keep in appearance gallery
            feature_ema_alpha: Moving average weight for smoothed feature
            enable_lost_reid: Whether to perform Stage 3 Re-ID against buffered lost tracks
            enable_occlusion_freeze: Whether to freeze feature updating during overlap/occlusion
        """
        self.max_lost_age = max_lost_age
        self.n_init = n_init
        self.appearance_weight = appearance_weight
        self.stage1_match_thresh = stage1_match_thresh
        self.stage2_iou_thresh = stage2_iou_thresh
        self.reid_lost_thresh = reid_lost_thresh
        self.occlusion_iou_thresh = occlusion_iou_thresh
        self.gallery_size = gallery_size
        self.feature_ema_alpha = feature_ema_alpha
        self.enable_lost_reid = enable_lost_reid
        self.enable_occlusion_freeze = enable_occlusion_freeze

        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        self._next_id = 1
        self.frame_count = 0

    def reset(self) -> None:
        """Reset tracker state and ID counter."""
        self.tracks = []
        self._next_id = 1
        self.frame_count = 0

    @property
    def active_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.state in (TrackState.ACTIVE, TrackState.TENTATIVE)]

    @property
    def confirmed_active_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.state == TrackState.ACTIVE]

    @property
    def lost_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.state == TrackState.LOST]

    def _check_and_mark_occlusions(self) -> None:
        """Detect when tracks overlap with each other (e.g. crossing paths).
        Sets `is_occluded` flag to prevent feature memory corruption.
        """
        active_list = self.confirmed_active_tracks
        if len(active_list) < 2:
            for t in active_list:
                t.is_occluded = False
                t.occlusion_score = 0.0
            return

        boxes = np.array([t.to_xyxy() for t in active_list], dtype=np.float32)
        inter_min = compute_inter_min_matrix(boxes, boxes)
        iou_mat = compute_iou_matrix(boxes, boxes)

        for i, t in enumerate(active_list):
            # Check overlap with any OTHER track
            max_inter_min = 0.0
            for j in range(len(active_list)):
                if i != j:
                    overlap = max(float(inter_min[i, j]), float(iou_mat[i, j]))
                    if overlap > max_inter_min:
                        max_inter_min = overlap

            t.occlusion_score = max_inter_min
            t.is_occluded = (max_inter_min >= self.occlusion_iou_thresh)

    def step(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with new frame detections.

        Args:
            detections: List of Detection objects for the current frame

        Returns:
            List of confirmed active Track objects
        """
        self.frame_count += 1

        # 1. Kalman Prediction step for all alive tracks
        for t in self.tracks:
            t.predict(self.kf)

        # 2. Identify potential occlusions among active tracks
        if self.enable_occlusion_freeze:
            self._check_and_mark_occlusions()

        active_tracks = self.active_tracks

        # 3. Stage 1: Associate Active Tracks with Detections via Appearance + Motion
        if len(active_tracks) > 0 and len(detections) > 0:
            if self.appearance_weight > 0.0:
                cost_mat = fused_cost_matrix(
                    active_tracks,
                    detections,
                    kf=self.kf,
                    appearance_weight=self.appearance_weight,
                    motion_gate=True
                )
            else:
                cost_mat = iou_cost_matrix(active_tracks, detections)

            matches_1, unmatched_tracks_1, unmatched_dets_1 = linear_assignment(
                cost_mat, cost_threshold=self.stage1_match_thresh
            )
        else:
            matches_1 = []
            unmatched_tracks_1 = list(range(len(active_tracks)))
            unmatched_dets_1 = list(range(len(detections)))

        # 4. Stage 2: Associate remaining active tracks using IoU fallback
        remaining_tracks = [active_tracks[i] for i in unmatched_tracks_1]
        remaining_dets = [detections[j] for j in unmatched_dets_1]

        if len(remaining_tracks) > 0 and len(remaining_dets) > 0:
            iou_cost = iou_cost_matrix(remaining_tracks, remaining_dets)
            matches_2_rel, unmatched_tracks_2_rel, unmatched_dets_2_rel = linear_assignment(
                iou_cost, cost_threshold=self.stage2_iou_thresh
            )
            # Map relative indices back
            matches_2 = [
                (unmatched_tracks_1[r], unmatched_dets_1[c])
                for r, c in matches_2_rel
            ]
            final_unmatched_tracks = [unmatched_tracks_1[i] for i in unmatched_tracks_2_rel]
            final_unmatched_dets_indices = [unmatched_dets_1[j] for j in unmatched_dets_2_rel]
        else:
            matches_2 = []
            final_unmatched_tracks = unmatched_tracks_1
            final_unmatched_dets_indices = unmatched_dets_1

        all_active_matches = matches_1 + matches_2

        # Update matched active tracks
        matched_track_indices = set()
        for track_idx, det_idx in all_active_matches:
            track = active_tracks[track_idx]
            det = detections[det_idx]
            z = xyxy_to_z(det.bbox)
            freeze = (track.is_occluded and self.enable_occlusion_freeze)
            track.update(self.kf, z, feature=det.feature, is_occluded_measurement=freeze)
            matched_track_indices.add(track_idx)

        # Mark unmatched active tracks as missed
        for i, track in enumerate(active_tracks):
            if i not in matched_track_indices:
                track.mark_missed()

        # 5. Stage 3 (Crucial Re-ID recovery):
        # Never immediately spawn a new ID when a temporarily occluded person reappears.
        # Match unmatched detections against LOST tracks using appearance gallery!
        unmatched_dets = [detections[idx] for idx in final_unmatched_dets_indices]
        lost_tracks = self.lost_tracks

        truly_unmatched_det_indices = []

        if self.enable_lost_reid and len(lost_tracks) > 0 and len(unmatched_dets) > 0:
            lost_app_cost = appearance_cost_matrix(lost_tracks, unmatched_dets, use_gallery_min=True)
            lost_matches_rel, _, unmatched_dets_3_rel = linear_assignment(
                lost_app_cost, cost_threshold=self.reid_lost_thresh
            )

            for lost_idx, det_rel_idx in lost_matches_rel:
                recovered_track = lost_tracks[lost_idx]
                det = unmatched_dets[det_rel_idx]
                z = xyxy_to_z(det.bbox)
                # Re-activate lost track with its original track_id!
                recovered_track.update(
                    self.kf, z, feature=det.feature, is_occluded_measurement=False
                )

            for rel_idx in unmatched_dets_3_rel:
                truly_unmatched_det_indices.append(final_unmatched_dets_indices[rel_idx])
        else:
            truly_unmatched_det_indices = final_unmatched_dets_indices

        # 6. Initialize tentative tracks for truly unmatched detections
        for det_idx in truly_unmatched_det_indices:
            det = detections[det_idx]
            self._initiate_track(det)

        # 7. Update lost tracks that were not re-identified
        for t in lost_tracks:
            if t.state == TrackState.LOST:
                t.mark_missed()

        # 8. Clean up deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted]

        # Return confirmed active tracks
        return self.confirmed_active_tracks

    def _initiate_track(self, detection: Detection) -> None:
        """Create a new tentative track from an unmatched detection."""
        z = xyxy_to_z(detection.bbox)
        mean, covariance = self.kf.initiate(z)
        new_track = Track(
            mean=mean,
            covariance=covariance,
            track_id=self._next_id,
            n_init=self.n_init,
            max_lost_age=self.max_lost_age,
            feature=detection.feature,
            gallery_size=self.gallery_size,
            feature_ema_alpha=self.feature_ema_alpha
        )
        self._next_id += 1
        self.tracks.append(new_track)
