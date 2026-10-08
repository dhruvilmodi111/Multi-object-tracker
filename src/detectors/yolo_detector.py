"""Pretrained YOLO Person Detector using Ultralytics (YOLOv8 / YOLO11).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
import cv2
from ultralytics import YOLO

def _clip_box(box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(img_w - 1)))
    y1 = max(0.0, min(float(y1), float(img_h - 1)))
    x2 = max(x1 + 1.0, min(float(x2), float(img_w)))
    y2 = max(y1 + 1.0, min(float(y2), float(img_h)))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class Detection:
    """Represents a single detected bounding box and its metadata."""
    def __init__(
        self,
        bbox: np.ndarray,
        conf: float,
        class_id: int = 0,
        feature: Optional[np.ndarray] = None
    ):
        self.bbox = np.asarray(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.conf = float(conf)
        self.class_id = int(class_id)
        self.feature = feature  # L2-normalized Re-ID embedding

    @property
    def tlwh(self) -> np.ndarray:
        return np.array([
            self.bbox[0],
            self.bbox[1],
            self.bbox[2] - self.bbox[0],
            self.bbox[3] - self.bbox[1]
        ], dtype=np.float32)


class YOLOPersonDetector:
    """Wrapper around Ultralytics YOLO to extract person detections."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.5,
        device: str = "auto",
        person_class_id: int = 0
    ):
        """Args:
            model_name: Pretrained weights name (e.g. yolov8n.pt, yolov8s.pt, yolo11n.pt)
            conf_thresh: Minimum detection confidence score
            iou_thresh: NMS IoU threshold
            device: 'cuda', 'cpu', or 'auto'
            person_class_id: COCO class ID for person (0)
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.person_class_id = person_class_id

        # Load pretrained model (Ultralytics handles weights caching automatically)
        self.model = YOLO(model_name)

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Perform object detection on a BGR image frame and return person detections.

        Args:
            image: BGR image frame as numpy array (H, W, 3)

        Returns:
            List of Detection objects
        """
        img_h, img_w = image.shape[:2]
        results = self.model.predict(
            source=image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False
        )

        detections: List[Detection] = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for b, c, cl in zip(xyxy, confs, classes):
            if int(cl) != self.person_class_id:
                continue
            clipped_box = _clip_box(b, img_w, img_h)
            detections.append(Detection(bbox=clipped_box, conf=c, class_id=int(cl)))

        return detections
