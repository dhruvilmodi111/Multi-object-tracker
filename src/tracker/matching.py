"""Association and assignment algorithms (Hungarian, IoU, Appearance distance, Gating).
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils.box_utils import compute_iou_matrix
from src.tracker.track import Track
from src.detectors.yolo_detector import Detection
from src.tracker.kalman_filter import KalmanFilter


def iou_cost_matrix(tracks: List[Track], detections: List[Detection]) -> np.ndarray:
    """Compute IoU distance matrix (1.0 - IoU) between tracks and detections."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)), dtype=np.float32)

    track_boxes = np.array([t.to_xyxy() for t in tracks], dtype=np.float32)
    det_boxes = np.array([d.bbox for d in detections], dtype=np.float32)

    iou_mat = compute_iou_matrix(track_boxes, det_boxes)
    return (1.0 - iou_mat).astype(np.float32)


def appearance_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
    use_gallery_min: bool = True
) -> np.ndarray:
    """Compute cosine distance matrix between tracks' appearance memory and detection features.
    Values are in [0, 2], where 0 indicates identical appearance.
    """
    num_tracks = len(tracks)
    num_dets = len(detections)
    if num_tracks == 0 or num_dets == 0:
        return np.empty((num_tracks, num_dets), dtype=np.float32)

    cost_matrix = np.ones((num_tracks, num_dets), dtype=np.float32)

    det_features = []
    has_det_features = []
    for d in detections:
        if d.feature is not None:
            det_features.append(d.feature)
            has_det_features.append(True)
        else:
            det_features.append(np.zeros(1, dtype=np.float32))
            has_det_features.append(False)

    for i, t in enumerate(tracks):
        track_gallery = t.features if (use_gallery_min and len(t.features) > 0) else ([t.smooth_feature] if t.smooth_feature is not None else [])
        if not track_gallery:
            continue

        gallery_matrix = np.array(track_gallery, dtype=np.float32)  # (G, D)

        for j, d in enumerate(detections):
            if not has_det_features[j]:
                continue
            det_feat = det_features[j]  # (D,)
            # Cosine similarity: (G, D) @ (D,) -> (G,)
            sims = np.dot(gallery_matrix, det_feat)
            best_sim = float(np.max(sims))
            # Cosine distance = 1.0 - similarity (clipped to [0, 1])
            cost_matrix[i, j] = max(0.0, 1.0 - best_sim)

    return cost_matrix


def fused_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
    kf: Optional[KalmanFilter] = None,
    appearance_weight: float = 0.65,
    motion_gate: bool = True,
    gate_threshold: float = 9.4877
) -> np.ndarray:
    """Combine appearance cosine distance and IoU distance with optional Kalman Mahalanobis gating."""
    iou_cost = iou_cost_matrix(tracks, detections)
    app_cost = appearance_cost_matrix(tracks, detections)

    if iou_cost.size == 0:
        return iou_cost

    fused = appearance_weight * app_cost + (1.0 - appearance_weight) * iou_cost

    # Apply Kalman gating if requested
    if motion_gate and kf is not None:
        for i, t in enumerate(tracks):
            measurements = np.array([
                [
                    d.bbox[0] + (d.bbox[2] - d.bbox[0]) / 2.0,
                    d.bbox[1] + (d.bbox[3] - d.bbox[1]) / 2.0,
                    (d.bbox[2] - d.bbox[0]) / max(1.0, (d.bbox[3] - d.bbox[1])),
                    d.bbox[3] - d.bbox[1]
                ]
                for d in detections
            ], dtype=np.float32)

            squared_dist = kf.gating_distance(t.mean, t.covariance, measurements)
            for j, dist in enumerate(squared_dist):
                if dist > gate_threshold:
                    fused[i, j] = 1e5  # Disallow association

    return fused


def linear_assignment(
    cost_matrix: np.ndarray,
    cost_threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Solve the bipartite matching problem via Hungarian Algorithm with cost gating.

    Returns:
        matches: List of (track_idx, det_idx) pairs where cost <= cost_threshold
        unmatched_tracks: List of track indices
        unmatched_detections: List of detection indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(range(cost_matrix.shape[0]))
    unmatched_dets = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= cost_threshold:
            matches.append((r, c))
            unmatched_tracks.remove(r)
            unmatched_dets.remove(c)

    return matches, sorted(list(unmatched_tracks)), sorted(list(unmatched_dets))
