"""Kalman Filter for 2D bounding box tracking.
State: [center_x, center_y, aspect_ratio, height, vx, vy, va, vh]
Measurement: [center_x, center_y, aspect_ratio, height]
"""

from typing import Tuple
import numpy as np
import scipy.linalg


class KalmanFilter:
    """A standard Kalman filter for tracking bounding boxes in image space."""

    # Chi-square distribution table for 4 degrees of freedom (0.95 confidence = 9.4877)
    CHI2_THRESHOLD_95 = 9.4877

    def __init__(self):
        ndim = 4
        dt = 1.0

        # Motion transition matrix F (8x8)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement matrix H (4x8)
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Motion and measurement noise scaling weights
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create track state and covariance from initial measurement [cx, cy, s, h]."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2.0 * self._std_weight_position * measurement[3],
            2.0 * self._std_weight_position * measurement[3],
            1e-2,
            2.0 * self._std_weight_position * measurement[3],
            10.0 * self._std_weight_velocity * measurement[3],
            10.0 * self._std_weight_velocity * measurement[3],
            1e-5,
            10.0 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean.astype(np.float32), covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray, lost_age: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step.

        Args:
            mean: State vector (8,)
            covariance: Covariance matrix (8, 8)
            lost_age: Number of consecutive frames the track was missing.
                      Slightly scales process noise to reflect position drift during occlusion.
        """
        # Noise factor inflates moderately if track is occluded/lost
        drift_factor = 1.0 + 0.15 * min(lost_age, 30)

        std_pos = [
            self._std_weight_position * mean[3] * drift_factor,
            self._std_weight_position * mean[3] * drift_factor,
            1e-2,
            self._std_weight_position * mean[3] * drift_factor,
        ]
        std_vel = [
            self._std_weight_velocity * mean[3] * drift_factor,
            self._std_weight_velocity * mean[3] * drift_factor,
            1e-5,
            self._std_weight_velocity * mean[3] * drift_factor,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std)).astype(np.float32)

        mean_proj = np.dot(self._update_mat, mean)
        cov_proj = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return mean_proj.astype(np.float32), cov_proj.astype(np.float32)

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter update step with measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean.astype(np.float32), new_covariance.astype(np.float32)

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False
    ) -> np.ndarray:
        """Compute squared Mahalanobis distance between projected state and measurements."""
        mean_proj, cov_proj = self.project(mean, covariance)
        if only_position:
            mean_proj = mean_proj[:2]
            cov_proj = cov_proj[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(cov_proj)
        d = measurements - mean_proj
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=False)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha.astype(np.float32)
