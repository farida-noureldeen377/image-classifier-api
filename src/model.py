"""Image classification model.

Wraps a pre-trained ResNet50 from torchvision. ResNet50 trained on ImageNet
already recognises 1000 everyday object categories, so no fine-tuning is
needed. The model is loaded once (it is expensive to construct) and reused
for every request.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

logger = logging.getLogger(__name__)


class ImageDecodeError(Exception):
    """Raised when the provided bytes cannot be decoded as an image."""


@dataclass
class Prediction:
    """A single classification result."""

    label: str
    confidence: float

    def as_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 4)}


class Classifier:
    """Loads ResNet50 once and serves predictions.

    Parameters
    ----------
    confidence_threshold:
        Predictions with a softmax probability below this value are dropped
        from the returned list.
    top_k:
        Maximum number of predictions to consider before threshold filtering.
    """

    def __init__(self, confidence_threshold: float, top_k: int = 5) -> None:
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

        # Use the canonical pretrained weights. The weights bundle ships with
        # the matching preprocessing transform and the ImageNet category
        # names, so we don't have to fetch labels from a separate URL.
        self._weights = ResNet50_Weights.DEFAULT
        self._categories = self._weights.meta["categories"]

        logger.info("Loading ResNet50 pretrained weights...")
        self._model = resnet50(weights=self._weights)
        self._model.eval()  # inference mode: disable dropout/batchnorm updates

        # The official transform handles resize, center-crop to 224x224, tensor
        # conversion, and ImageNet mean/std normalization.
        self._preprocess = self._weights.transforms()
        logger.info("ResNet50 ready (%d categories).", len(self._categories))

    @property
    def categories(self) -> list[str]:
        return list(self._categories)

    def predict(self, image_bytes: bytes) -> list[dict]:
        """Classify raw image bytes.

        Returns a list of ``{"label", "confidence"}`` dicts sorted by
        descending confidence, with anything below the configured threshold
        filtered out.
        """
        image = self._decode(image_bytes)

        # Preprocess -> add batch dimension -> run forward pass with grad off.
        tensor = self._preprocess(image).unsqueeze(0)
        with torch.inference_mode():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        k = min(self.top_k, probs.shape[0])
        top_probs, top_idx = torch.topk(probs, k)

        predictions = [
            Prediction(self._categories[idx], float(prob))
            for prob, idx in zip(top_probs.tolist(), top_idx.tolist())
            if prob >= self.confidence_threshold
        ]
        return [p.as_dict() for p in predictions]

    @staticmethod
    def _decode(image_bytes: bytes) -> Image.Image:
        """Decode bytes into an RGB PIL image, or raise ImageDecodeError."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise ImageDecodeError("Could not decode image data.") from exc
        # Convert to RGB so grayscale/RGBA/palette images all work uniformly.
        return image.convert("RGB")
