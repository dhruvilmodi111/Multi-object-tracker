"""Deep Appearance Feature Extractor (Re-ID) for Person Tracking.
"""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2


class ReIDFeatureExtractor:
    """Extracts L2-normalized deep appearance embeddings from person image crops."""

    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        embedding_dim: int = 512,
        input_size: tuple = (128, 64),  # (height, width) standard for Re-ID
        device: str = "auto"
    ):
        """Args:
            model_name: Backbone architecture ('mobilenet_v3_small', 'resnet18', etc.)
            embedding_dim: Dimensionality of the output embedding vector
            input_size: Tuple (height, width) for image crop resizing
            device: 'cuda', 'cpu', or 'auto'
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.input_size = input_size
        self.embedding_dim = embedding_dim

        # Build feature extraction backbone
        if model_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            backbone = models.mobilenet_v3_small(weights=weights)
            in_features = backbone.classifier[0].in_features
            # Replace classifier with embedding projection
            backbone.classifier = nn.Sequential(
                nn.Linear(in_features, embedding_dim),
                nn.BatchNorm1d(embedding_dim)
            )
            self.model = backbone
        elif model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Linear(in_features, embedding_dim),
                nn.BatchNorm1d(embedding_dim)
            )
            self.model = backbone
        else:
            raise ValueError(f"Unsupported Re-ID backbone: {model_name}")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def extract(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract L2-normalized feature vectors for a list of BGR person crops.

        Args:
            crops: List of BGR numpy arrays (H, W, 3)

        Returns:
            features: np.ndarray of shape (N, embedding_dim)
        """
        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        batch_tensors = []
        for crop in crops:
            if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                # Handle edge cases / blank patches
                blank = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
                crop_rgb = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
            else:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            tensor = self.transform(crop_rgb)
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors, dim=0).to(self.device)
        embeddings = self.model(batch)

        # L2 Normalization
        norm = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        normalized = embeddings / norm

        return normalized.cpu().numpy().astype(np.float32)

    def extract_from_boxes(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Crop boxes from frame and extract embeddings.

        Args:
            frame: Full BGR frame (H, W, 3)
            boxes: Array of [x1, y1, x2, y2]
        """
        if len(boxes) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        h, w = frame.shape[:2]
        crops = []
        for box in boxes:
            x1 = max(0, min(int(round(box[0])), w - 1))
            y1 = max(0, min(int(round(box[1])), h - 1))
            x2 = max(x1 + 1, min(int(round(box[2])), w))
            y2 = max(y1 + 1, min(int(round(box[3])), h))
            crop = frame[y1:y2, x1:x2]
            crops.append(crop)

        return self.extract(crops)
