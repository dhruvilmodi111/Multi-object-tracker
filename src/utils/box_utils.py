"""Bounding box geometry, format conversions, and spatial distance metrics.
"""

from typing import Union, Tuple, List
import numpy as np


def xyxy_to_tlwh(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to top-left format [top_x, top_y, width, height]."""
    bbox = np.asarray(bbox, dtype=np.float32)
    tlwh = bbox.copy()
    tlwh[..., 2] = bbox[..., 2] - bbox[..., 0]
    tlwh[..., 3] = bbox[..., 3] - bbox[..., 1]
    return tlwh


def tlwh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """Convert [top_x, top_y, width, height] to [x1, y1, x2, y2]."""
    bbox = np.asarray(bbox, dtype=np.float32)
    xyxy = bbox.copy()
    xyxy[..., 2] = bbox[..., 0] + bbox[..., 2]
    xyxy[..., 3] = bbox[..., 1] + bbox[..., 3]
    return xyxy


def xyxy_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to Kalman state measurement [center_x, center_y, aspect_ratio, height].
    Aspect ratio is defined as width / height.
    """
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = float(x1 + w / 2.0)
    cy = float(y1 + h / 2.0)
    s = w / h  # aspect ratio
    return np.array([cx, cy, s, h], dtype=np.float32)


def z_to_xyxy(state: np.ndarray) -> np.ndarray:
    """Convert Kalman state [cx, cy, s, h, ...] to [x1, y1, x2, y2]."""
    cx, cy, s, h = state[0], state[1], state[2], state[3]
    w = max(1.0, float(s * h))
    h = max(1.0, float(h))
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise Intersection-over-Union (IoU) matrix between two sets of [x1, y1, x2, y2] boxes.

    Args:
        boxes_a: Shape (N, 4) in [x1, y1, x2, y2]
        boxes_b: Shape (M, 4) in [x1, y1, x2, y2]

    Returns:
        iou_matrix: Shape (N, M) with values in [0, 1]
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    # (N, 1, 2) vs (1, M, 2)
    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    inter_wh = np.maximum(0.0, br - tl)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    union_area = np.maximum(union_area, 1e-6)

    return (inter_area / union_area).astype(np.float32)


def compute_inter_min_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute Intersection-over-Minimum-Area matrix to detect when one box is largely inside another
    (e.g., partial or total visual occlusion).

    Returns:
        matrix: Shape (N, M) where matrix[i, j] = intersection_area / min(area_i, area_j)
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    inter_wh = np.maximum(0.0, br - tl)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    min_area = np.minimum(area_a[:, None], area_b[None, :])
    min_area = np.maximum(min_area, 1e-6)

    return (inter_area / min_area).astype(np.float32)


def clip_box_to_image(box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Clip [x1, y1, x2, y2] to within image bounds [0, 0, img_w, img_h]."""
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(img_w - 1)))
    y1 = max(0.0, min(float(y1), float(img_h - 1)))
    x2 = max(x1 + 1.0, min(float(x2), float(img_w)))
    y2 = max(y1 + 1.0, min(float(y2), float(img_h)))
    return np.array([x1, y1, x2, y2], dtype=np.float32)
