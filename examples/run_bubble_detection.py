"""Run the bundled YOLO26 detector on the AI-generated sample page.

This detection-only demonstration uses an AI-generated sample without
ground-truth annotations, so it does not compute evaluation metrics. It runs
the detector with tiled inference, ownership and NMS deduplication, and ROI
expansion. Other pipeline stages are not run.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from toontra import Toontra
from toontra.imaging import load_rgb, save_rgb
from toontra.modules import Yolo26BubbleDetector

HERE = Path(__file__).resolve().parent
INPUT_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "toontra"
    / "assets"
    / "sample_webtoon.png"
)
OUTPUT_DIR = HERE / "output"


def draw_boxes(image, detections):
    annotated = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for detection in detections:
        x1, y1, x2, y2 = detection.box.as_tuple()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{detection.score:.2f}"
        cv2.putText(
            annotated, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1
        )
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def main() -> None:
    if not INPUT_IMAGE.is_file():
        raise FileNotFoundError(f"Missing sample image: {INPUT_IMAGE}")

    image = load_rgb(INPUT_IMAGE)
    toontra = Toontra(detector=Yolo26BubbleDetector(), tile_height=1600, tile_overlap=256)
    detections = toontra.detect_bubbles(image)

    OUTPUT_DIR.mkdir(exist_ok=True)
    visualization = draw_boxes(image, detections)
    save_rgb(OUTPUT_DIR / "detections.png", visualization)

    import json

    payload = {
        "source": str(INPUT_IMAGE.name),
        "detector": "Yolo26BubbleDetector",
        "detection_expansion_ratio": toontra.detection_expansion_ratio,
        "bubble_count": len(detections),
        "bubbles": [
            {"box": list(item.box.as_tuple()), "score": round(item.score, 6)}
            for item in detections
        ],
    }
    (OUTPUT_DIR / "detections.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Detected {len(detections)} bubble(s)")
    print(f"Saved visualization to {(OUTPUT_DIR / 'detections.png').resolve()}")
    print(f"Saved coordinates to {(OUTPUT_DIR / 'detections.json').resolve()}")


if __name__ == "__main__":
    main()
