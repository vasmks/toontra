from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

RGBImage: TypeAlias = np.ndarray
GrayMask: TypeAlias = np.ndarray


@dataclass(frozen=True, slots=True)
class Box:
    """Half-open xyxy pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        for field_name in ("x1", "y1", "x2", "y2"):
            value = getattr(self, field_name)
            if not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}")
            object.__setattr__(self, field_name, int(value))
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Box must have positive width and height")

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def as_slices(self) -> tuple[slice, slice]:
        return slice(self.y1, self.y2), slice(self.x1, self.x2)

    def translate(self, x_offset: int = 0, y_offset: int = 0) -> Box:
        return Box(
            self.x1 + x_offset,
            self.y1 + y_offset,
            self.x2 + x_offset,
            self.y2 + y_offset,
        )

    def expand(self, pad_x: int, pad_y: int, width: int, height: int) -> Box:
        """Pad left/right by ``pad_x`` and top/bottom by ``pad_y``, then clip."""
        if pad_x < 0 or pad_y < 0:
            raise ValueError("padding must be non-negative")
        clipped = Box(
            self.x1 - pad_x,
            self.y1 - pad_y,
            self.x2 + pad_x,
            self.y2 + pad_y,
        ).clip(width, height)
        if clipped is None:
            raise ValueError("Expanded box does not intersect the image")
        return clipped

    def clip(self, width: int, height: int) -> Box | None:
        if width <= 0 or height <= 0:
            raise ValueError("Image width and height must be positive")
        x1 = max(0, min(self.x1, width))
        y1 = max(0, min(self.y1, height))
        x2 = max(0, min(self.x2, width))
        y2 = max(0, min(self.y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return Box(x1, y1, x2, y2)


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    score: float = 1.0
    label: str = "speech_bubble"
    tile_id: int | None = None

    def __post_init__(self) -> None:
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Detection score must be a finite value between 0 and 1")
        if not self.label:
            raise ValueError("Detection label cannot be empty")
        object.__setattr__(self, "score", score)


@dataclass(frozen=True, slots=True)
class Recognition:
    text: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Recognition text must be a string")
        if self.confidence is not None:
            confidence = float(self.confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("Recognition confidence must be between 0 and 1")
            object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    name: str
    version: str
    source_url: str | None = None
    license_spdx: str | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, str]:
        values = {
            "name": self.name,
            "version": self.version,
            "source_url": self.source_url,
            "license_spdx": self.license_spdx,
            "sha256": self.sha256,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass(slots=True)
class BubbleResult:
    index: int
    detection: Detection
    recognition: Recognition | None = None
    translation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "label": self.detection.label,
            "score": round(self.detection.score, 6),
            "box": list(self.detection.box.as_tuple()),
        }
        if self.recognition is not None:
            payload["text"] = self.recognition.text
            payload["ocr_confidence"] = self.recognition.confidence
        if self.translation is not None:
            payload["translation"] = self.translation
        return payload


@dataclass(slots=True)
class PageResult:
    original: RGBImage
    cleaned: RGBImage
    mask: GrayMask
    overlay: np.ndarray
    bubbles: list[BubbleResult]
    source: Path | None = None
    pipeline: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        height, width = self.original.shape[:2]
        payload = {
            "schema_version": "1.0",
            "source": str(self.source) if self.source is not None else None,
            "image": {"width": width, "height": height, "color_space": "RGB"},
            "mask": {"dtype": "uint8", "meaning": "0 keeps, 255 erases"},
            "bubble_count": len(self.bubbles),
            "bubbles": [bubble.to_dict() for bubble in self.bubbles],
        }
        if self.pipeline:
            payload["pipeline"] = self.pipeline
        return payload

    def save(
        self,
        output_dir: str | Path,
        *,
        save_crops: bool = False,
        force: bool = False,
    ) -> Path:
        from .output import save_page_result

        return save_page_result(self, output_dir, save_crops=save_crops, force=force)


@dataclass(frozen=True, slots=True)
class Tile:
    index: int
    image: RGBImage
    x0: int
    y0: int

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def x1(self) -> int:
        return self.x0 + self.width

    @property
    def y1(self) -> int:
        return self.y0 + self.height
