"""Occlusion-Aware Track State Representation with Appearance Memory Gallery.
"""

from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

from src.utils.box_utils import z_to_xyxy, xyxy_to_z
from src.tracker.kalman_filter import KalmanFilter


class TrackState(Enum):
    TENTATIVE = 1
    ACTIVE = 2
    LOST = 3
    DELETED = 4


class Track:
    """Represents a tracked individual over time with Kalman state and appearance memory."""

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        track_id: int,
        n_init: int = 3,
        max_lost_age: int = 30,
        feature: Optional[np.ndarray] = None,
        gallery_size: int = 8,
        feature_ema_alpha: float = 0.9
    ):
        self.track_id = track_id
        self.mean = mean
        self.covariance = covariance
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self._n_init = n_init
        self._max_lost_age = max_lost_age

        # Appearance features and gallery
        self.gallery_size = gallery_size
        self.feature_ema_alpha = feature_ema_alpha
        self.features: List[np.ndarray] = []
        self.smooth_feature: Optional[np.ndarray] = None

        if feature is not None:
            self.add_feature(feature)

        # Occlusion and motion tracking
        self.is_occluded = False
        self.occlusion_score = 0.0
        self.reid_recovered_count = 0

        # Trajectory center history
        box = z_to_xyxy(self.mean[:4])
        cx, cy = float(self.mean[0]), float(self.mean[1])
        self.history_centers: List[Tuple[float, float]] = [(cx, cy)]
        self.last_bbox: np.ndarray = box

    @property
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    @property
    def is_active(self) -> bool:
        return self.state == TrackState.ACTIVE

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED

    def to_xyxy(self) -> np.ndarray:
        """Current bounding box [x1, y1, x2, y2] estimate from Kalman state."""
        return z_to_xyxy(self.mean[:4])

    def to_tlwh(self) -> np.ndarray:
        """Current bounding box [x_top_left, y_top_left, width, height]."""
        xyxy = self.to_xyxy()
        return np.array([
            xyxy[0],
            xyxy[1],
            xyxy[2] - xyxy[0],
            xyxy[3] - xyxy[1]
        ], dtype=np.float32)

    def add_feature(self, feat: np.ndarray) -> None:
        """Add feature to gallery and update exponential moving average feature."""
        feat = feat / np.maximum(1e-12, np.linalg.norm(feat))
        self.features.append(feat)
        if len(self.features) > self.gallery_size:
            self.features.pop(0)

        if self.smooth_feature is None:
            self.smooth_feature = feat.copy()
        else:
            self.smooth_feature = (
                self.feature_ema_alpha * self.smooth_feature +
                (1.0 - self.feature_ema_alpha) * feat
            )
            self.smooth_feature /= np.maximum(1e-12, np.linalg.norm(self.smooth_feature))

    def predict(self, kf: KalmanFilter) -> None:
        """Propagate state distribution using Kalman filter prediction step."""
        self.mean, self.covariance = kf.predict(
            self.mean, self.covariance, lost_age=self.time_since_update
        )
        self.age += 1
        self.time_since_update += 1

        # Keep trajectory history updated with predicted position during coasting
        cx, cy = float(self.mean[0]), float(self.mean[1])
        self.history_centers.append((cx, cy))
        if len(self.history_centers) > 60:
            self.history_centers.pop(0)
        self.last_bbox = self.to_xyxy()

    def update(
        self,
        kf: KalmanFilter,
        measurement: np.ndarray,
        feature: Optional[np.ndarray] = None,
        is_occluded_measurement: bool = False
    ) -> None:
        """Update track with matched detection measurement and optional Re-ID feature."""
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)
        self.hits += 1
        self.time_since_update = 0

        # Update appearance memory ONLY if not in heavy occlusion to avoid feature corruption
        if feature is not None and not is_occluded_measurement:
            self.add_feature(feature)

        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.ACTIVE
        elif self.state == TrackState.LOST:
            # Successfully re-identified from lost buffer!
            self.state = TrackState.ACTIVE
            self.reid_recovered_count += 1

        self.last_bbox = self.to_xyxy()

    def mark_missed(self) -> None:
        """Mark track as missed in current frame. Transitions state appropriately."""
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.state == TrackState.ACTIVE:
            self.state = TrackState.LOST
        elif self.state == TrackState.LOST:
            if self.time_since_update > self._max_lost_age:
                self.state = TrackState.DELETED
