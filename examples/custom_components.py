"""Minimal examples of replaceable Toontra components.

Real model adapters should keep framework tensors and preprocessing inside the
callable.  Toontra's boundary stays a NumPy RGB ``uint8`` image in and typed,
pixel-coordinate results out.
"""

from __future__ import annotations

import numpy as np

from toontra.models import Box, Detection, GrayMask, ModelMetadata, Recognition, RGBImage
from toontra.modules import CallableBubbleDetector


def center_box_detector(image: RGBImage) -> list[Detection]:
    """Example callable standing in for a framework-specific model."""

    height, width = image.shape[:2]
    if height < 4 or width < 4:
        return []
    return [
        Detection(
            box=Box(
                x1=width // 4,
                y1=height // 4,
                x2=max(width // 4 + 1, width * 3 // 4),
                y2=max(height // 4 + 1, height * 3 // 4),
            ),
            score=0.5,
        )
    ]


class CustomRecognizer:
    """Shape of a custom OCR component; replace the body with model inference."""

    def recognize(self, bubble_crop: RGBImage, *, language: str = "ko") -> Recognition:
        if not isinstance(bubble_crop, np.ndarray) or bubble_crop.dtype != np.uint8:
            raise ValueError("expected an RGB uint8 NumPy image")
        if bubble_crop.ndim != 3 or bubble_crop.shape[2] != 3:
            raise ValueError("expected image shape (height, width, 3)")
        del language
        return Recognition(text="", confidence=0.0)


class CustomMasker:
    """Valid placeholder for a segmentation model adapter."""

    metadata = ModelMetadata(
        name="replace-with-your-model-name",
        version="replace-with-an-immutable-version",
        source_url=None,
        license_spdx=None,
        sha256=None,
    )

    def create_mask(self, bubble_crop: RGBImage) -> GrayMask:
        """Return one coverage mask with the crop's exact height and width."""

        if bubble_crop.dtype != np.uint8 or bubble_crop.ndim != 3:
            raise ValueError("expected an RGB uint8 NumPy image")
        # Replace this with model inference and output resizing. A full mask is
        # a visible, contract-correct placeholder; 0 would preserve the crop.
        return np.full(bubble_crop.shape[:2], 255, dtype=np.uint8)


# The adapter validates every box and score before the pipeline receives it.
custom_detector = CallableBubbleDetector(center_box_detector)
custom_masker = CustomMasker()
