from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .errors import ImageValidationError, ModelContractError
from .models import Detection, GrayMask, Recognition, RGBImage


@runtime_checkable
class BubbleDetector(Protocol):
    def detect(self, image: RGBImage) -> Sequence[Detection]:
        """Return detections in the coordinate system of the supplied image."""


@runtime_checkable
class BubbleMasker(Protocol):
    def create_mask(self, bubble: RGBImage) -> GrayMask:
        """Return a uint8 coverage mask matching the bubble crop."""


@runtime_checkable
class TextRecognizer(Protocol):
    def recognize(self, bubble: RGBImage, *, language: str) -> Recognition:
        """Recognize one bubble crop."""


@runtime_checkable
class Translator(Protocol):
    def translate(
        self,
        texts: Sequence[str],
        *,
        source_language: str,
        target_language: str,
    ) -> Sequence[str]:
        """Translate texts while preserving their order and count."""


def validate_rgb_image(image: RGBImage, *, name: str = "image") -> RGBImage:
    if not isinstance(image, np.ndarray):
        raise ImageValidationError(f"{name} must be a NumPy array")
    if image.dtype != np.uint8:
        raise ImageValidationError(f"{name} must use uint8 pixels")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageValidationError(f"{name} must have shape [height, width, 3]")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ImageValidationError(f"{name} cannot be empty")
    return image


def validate_gray_mask(
    mask: GrayMask,
    expected_shape: tuple[int, int],
    *,
    name: str = "mask",
) -> GrayMask:
    if not isinstance(mask, np.ndarray):
        raise ModelContractError(f"{name} must be a NumPy array")
    if mask.dtype != np.uint8:
        raise ModelContractError(f"{name} must use uint8 coverage values")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(expected_shape):
        raise ModelContractError(
            f"{name} must have shape {expected_shape}, got {tuple(mask.shape)}"
        )
    return mask


def normalize_detections(
    detections: Sequence[Detection],
    *,
    width: int,
    height: int,
) -> list[Detection]:
    normalized: list[Detection] = []
    try:
        candidates = list(detections)
    except TypeError as error:
        message = "Detector output must be a sequence of Detection objects"
        raise ModelContractError(message) from error

    for index, detection in enumerate(candidates):
        if not isinstance(detection, Detection):
            raise ModelContractError(f"Detector item {index} is not a Detection")
        clipped = detection.box.clip(width, height)
        if clipped is None:
            continue
        normalized.append(
            Detection(clipped, detection.score, detection.label, detection.tile_id)
        )
    return normalized


def validate_recognition(value: Recognition) -> Recognition:
    if not isinstance(value, Recognition):
        raise ModelContractError("Recognizer must return a Recognition object")
    return value


def validate_translations(translations: Sequence[str], expected_count: int) -> list[str]:
    if isinstance(translations, str):
        raise ModelContractError("Translator output must be a sequence of strings")

    try:
        values = list(translations)
    except TypeError as error:
        raise ModelContractError("Translator output must be a sequence of strings") from error
    if len(values) != expected_count:
        raise ModelContractError(
            f"Translator returned {len(values)} items for {expected_count} input texts"
        )
    if any(not isinstance(value, str) for value in values):
        raise ModelContractError("Every translated value must be a string")
    return values
