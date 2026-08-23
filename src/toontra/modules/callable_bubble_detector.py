"""Adapt a plain callable to Toontra's detector contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any

import numpy as np

from toontra.contracts import validate_rgb_image
from toontra.errors import ModelContractError
from toontra.models import Detection, RGBImage


class CallableBubbleDetector:
    """Adapt a framework-specific callable to Toontra's detector contract.

    The wrapped callable receives one RGB ``uint8`` array and must return
    ``Detection`` objects in coordinates relative to that array.  Framework
    tensors and raw dictionaries should be converted inside the callable.
    """

    def __init__(self, detector: Callable[[RGBImage], Sequence[Detection]]) -> None:
        if not callable(detector):
            raise TypeError("detector must be callable")
        self._detector = detector

    def detect(self, image: RGBImage) -> list[Detection]:
        rgb = validate_rgb_image(image)
        raw_output: Any = self._detector(rgb)
        if isinstance(raw_output, (str, bytes, np.ndarray)) or not isinstance(
            raw_output, Sequence
        ):
            raise ModelContractError("custom detector must return a sequence of Detection objects")

        height, width = rgb.shape[:2]
        validated: list[Detection] = []
        for index, detection in enumerate(raw_output):
            if not isinstance(detection, Detection):
                raise ModelContractError(f"custom detector item {index} is not a Detection")

            box = detection.box
            coordinates = (box.x1, box.y1, box.x2, box.y2)
            integer_coordinates = all(
                isinstance(value, int) and not isinstance(value, bool) for value in coordinates
            )
            if not integer_coordinates:
                raise ModelContractError(
                    f"custom detector item {index} has non-integer coordinates"
                )
            if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
                raise ModelContractError(
                    f"custom detector item {index} lies outside the input image"
                )
            if box.x1 >= box.x2 or box.y1 >= box.y2:
                raise ModelContractError(f"custom detector item {index} has an empty box")

            score = detection.score
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ModelContractError(f"custom detector item {index} has a non-numeric score")
            if not isfinite(float(score)) or not 0 <= float(score) <= 1:
                raise ModelContractError(
                    f"custom detector item {index} score must be between 0 and 1"
                )
            validated.append(detection)

        return validated
