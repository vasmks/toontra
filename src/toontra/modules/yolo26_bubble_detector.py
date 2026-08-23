"""Bundled YOLO26s speech-bubble detector.

The checkpoint (`weights/speech_bubble_yolo26s.pt`, next to this package) was
trained for this public reconstruction with Ultralytics on a public Roboflow
speech-bubble dataset. It is the only bundled detector implementation. See
docs/model_choices.md for training and license details.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from toontra.contracts import validate_rgb_image
from toontra.models import Box, Detection, ModelMetadata, RGBImage

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "speech_bubble_yolo26s.pt"


class Yolo26BubbleDetector:
    """Wrap the bundled YOLO26s checkpoint behind Toontra's detector contract.

    Requires the ``ultralytics`` dependency, which Toontra installs by
    default since this is the pipeline's default detector.
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        imgsz: int = 800,
        device: str | None = None,
    ) -> None:
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(f"YOLO26 weights not found at {WEIGHTS_PATH}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")

        from ultralytics import YOLO

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self._model = YOLO(str(WEIGHTS_PATH))
        self.metadata = ModelMetadata(
            name="YOLO26s speech-bubble detector",
            version="1.0.0",
            source_url="https://github.com/ultralytics/ultralytics",
            license_spdx="AGPL-3.0-only",
            sha256=_weights_sha256(),
        )

    def detect(self, image: RGBImage) -> list[Detection]:
        rgb = validate_rgb_image(image)
        height, width = rgb.shape[:2]
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        predict_kwargs = {
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "imgsz": self.imgsz,
            "end2end": False,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self._model.predict(
            bgr,
            **predict_kwargs,
        )[0]

        detections: list[Detection] = []
        for box_tensor, conf_tensor in zip(results.boxes.xyxy, results.boxes.conf, strict=True):
            x1, y1, x2, y2 = (int(round(value)) for value in box_tensor.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(box=Box(x1, y1, x2, y2), score=float(conf_tensor.item()))
            )

        detections.sort(key=lambda item: (item.box.y1, item.box.x1, item.box.y2, item.box.x2))
        return detections


_CACHED_SHA256: str | None = None


def _weights_sha256() -> str:
    global _CACHED_SHA256
    if _CACHED_SHA256 is None:
        digest = hashlib.sha256()
        with WEIGHTS_PATH.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
        _CACHED_SHA256 = digest.hexdigest()
    return _CACHED_SHA256
