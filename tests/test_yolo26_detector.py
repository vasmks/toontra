from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from toontra.imaging import load_rgb
from toontra.modules import Yolo26BubbleDetector

SAMPLE_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "toontra"
    / "assets"
    / "sample_webtoon.png"
)


class Yolo26DetectorContractTests(unittest.TestCase):
    def test_rejects_invalid_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            Yolo26BubbleDetector(confidence_threshold=1.5)
        with self.assertRaises(ValueError):
            Yolo26BubbleDetector(iou_threshold=-0.1)
        with self.assertRaises(ValueError):
            Yolo26BubbleDetector(imgsz=0)

    def test_detect_returns_contract_compliant_detections(self) -> None:
        # A blank page checks the BubbleDetector contract on a real inference
        # pass; detector accuracy is evaluated separately.
        image = np.full((256, 256, 3), 255, dtype=np.uint8)
        detector = Yolo26BubbleDetector()
        detections = detector.detect(image)
        self.assertIsInstance(detections, list)
        for detection in detections:
            self.assertGreaterEqual(detection.box.x1, 0)
            self.assertGreaterEqual(detection.box.y1, 0)
            self.assertLessEqual(detection.box.x2, 256)
            self.assertLessEqual(detection.box.y2, 256)
            self.assertTrue(0.0 <= detection.score <= 1.0)
        self.assertIsNotNone(detector.metadata.sha256)

    def test_detect_converts_rgb_to_contiguous_bgr_for_ultralytics(self) -> None:
        rgb = np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        predict = Mock(
            return_value=[SimpleNamespace(boxes=SimpleNamespace(xyxy=[], conf=[]))]
        )
        detector = object.__new__(Yolo26BubbleDetector)
        detector.confidence_threshold = 0.25
        detector.iou_threshold = 0.7
        detector.imgsz = 800
        detector.device = None
        detector._model = SimpleNamespace(predict=predict)

        self.assertEqual(detector.detect(rgb), [])

        ultralytics_input = predict.call_args.args[0]
        np.testing.assert_array_equal(ultralytics_input, rgb[..., ::-1])
        self.assertEqual(ultralytics_input.dtype, np.uint8)
        self.assertEqual(ultralytics_input.shape, rgb.shape)
        self.assertTrue(ultralytics_input.flags.c_contiguous)

    def test_real_sample_detection_smoke(self) -> None:
        image = load_rgb(SAMPLE_IMAGE)
        height, width = image.shape[:2]

        detections = Yolo26BubbleDetector().detect(image)

        self.assertGreater(len(detections), 0)
        for detection in detections:
            self.assertGreaterEqual(detection.box.x1, 0)
            self.assertGreaterEqual(detection.box.y1, 0)
            self.assertLessEqual(detection.box.x2, width)
            self.assertLessEqual(detection.box.y2, height)
            self.assertTrue(0.0 <= detection.score <= 1.0)

    def test_detect_is_sorted_top_to_bottom(self) -> None:
        image = np.full((512, 384, 3), 255, dtype=np.uint8)
        detections = Yolo26BubbleDetector().detect(image)
        boxes = [item.box for item in detections]
        self.assertEqual(
            boxes,
            sorted(boxes, key=lambda box: (box.y1, box.x1, box.y2, box.x2)),
        )

    def test_detect_forwards_explicit_device_to_ultralytics(self) -> None:
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        predict = Mock(
            return_value=[SimpleNamespace(boxes=SimpleNamespace(xyxy=[], conf=[]))]
        )

        detector = object.__new__(Yolo26BubbleDetector)
        detector.confidence_threshold = 0.25
        detector.iou_threshold = 0.7
        detector.imgsz = 800
        detector.device = "cpu"
        detector._model = SimpleNamespace(predict=predict)

        self.assertEqual(detector.detect(image), [])
        self.assertEqual(predict.call_args.kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
