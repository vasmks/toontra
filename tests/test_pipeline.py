from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from stub_detector import SequencedStubBubbleDetector, StubBubbleDetector

from toontra import Recognition, Toontra
from toontra.errors import ModelContractError, OutputExistsError
from toontra.models import Box
from toontra.modules import DictionaryTranslator
from toontra.synthetic import create_synthetic_long_webtoon, create_synthetic_webtoon

# Bounding boxes of the two ellipse bubbles create_synthetic_webtoon() (at
# its default 720x1080 size) actually draws, so WhiteBubbleMasker -- a real
# algorithm, not a stub -- has real bubble pixels to find at these locations.
TWO_BUBBLE_BOXES = [Box(336, 121, 624, 265), Box(96, 644, 384, 788)]


class FixedRecognizer:
    def recognize(self, bubble: np.ndarray, *, language: str) -> Recognition:
        return Recognition(f"{language}:{bubble.shape[1]}", confidence=0.9)

class FailIfCalledTranslator:
    def translate(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
    ) -> list[str]:
        raise AssertionError("translator should not be called without recognized text")

class BadMasker:
    def create_mask(self, bubble: np.ndarray) -> np.ndarray:
        return np.zeros((1, 1), dtype=np.uint8)


def two_bubble_detector() -> StubBubbleDetector:
    return StubBubbleDetector(TWO_BUBBLE_BOXES)


class PipelineTests(unittest.TestCase):
    def test_default_pipeline_processes_original_demo(self) -> None:
        image = create_synthetic_webtoon()
        before = image.copy()
        result = Toontra(detector=two_bubble_detector()).process(image)
        np.testing.assert_array_equal(image, before)
        self.assertEqual(len(result.bubbles), 2)
        self.assertEqual(result.mask.shape, image.shape[:2])
        self.assertEqual(result.overlay.shape, (*image.shape[:2], 4))
        self.assertGreater(int(result.mask.max()), 0)

    def test_ocr_and_translation_keep_bubble_order(self) -> None:
        image = create_synthetic_webtoon()
        translator = DictionaryTranslator({}, keep_unknown=True)
        result = Toontra(
            detector=two_bubble_detector(),
            recognizer=FixedRecognizer(),
            translator=translator,
        ).process(
            image,
            source_language="ko",
            target_language="en",
        )
        for bubble in result.bubbles:
            self.assertIsNotNone(bubble.recognition)
            self.assertEqual(bubble.translation, bubble.recognition.text)

    def test_invalid_masker_output_fails_at_component_boundary(self) -> None:
        with self.assertRaisesRegex(ModelContractError, "shape"):
            Toontra(detector=two_bubble_detector(), masker=BadMasker()).process(
                create_synthetic_webtoon()
            )

    def test_result_save_writes_schema_and_refuses_implicit_overwrite(self) -> None:
        result = Toontra(detector=two_bubble_detector()).process(create_synthetic_webtoon())
        with tempfile.TemporaryDirectory(prefix="toontra-output-") as directory:
            output = Path(directory) / "page"
            result.save(output, save_crops=True)
            metadata = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], "1.0")
            self.assertEqual(metadata["bubble_count"], 2)
            self.assertIn("StubBubbleDetector", metadata["pipeline"]["detector"]["adapter"])
            self.assertIn("detection_expansion_ratio", metadata["pipeline"])
            self.assertEqual(len(list((output / "crops").glob("*.png"))), 2)
            with self.assertRaises(OutputExistsError):
                result.save(output)

    def test_force_removes_stale_crops_but_preserves_unrelated_files(self) -> None:
        result = Toontra(detector=two_bubble_detector()).process(create_synthetic_webtoon())
        with tempfile.TemporaryDirectory(prefix="toontra-output-") as directory:
            output = Path(directory) / "page"
            result.save(output, save_crops=True)
            stale_crop = output / "crops" / "bubble_999.png"
            unrelated_crop = output / "crops" / "bubble_reference.png"
            unrelated_crop.write_bytes(b"keep me")
            stale_crop.write_bytes(b"stale")
            unrelated = output / "notes.txt"
            unrelated.write_text("keep me", encoding="utf-8")

            result.save(output, save_crops=False, force=True)

            self.assertFalse(stale_crop.exists())
            self.assertEqual(unrelated_crop.read_bytes(), b"keep me")
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep me")

    def test_batch_rejects_single_source_path(self) -> None:
        image = create_synthetic_webtoon()
        toontra = Toontra(detector=two_bubble_detector())

        with self.assertRaisesRegex(ValueError, "source cannot be used with batch input"):
            toontra.process([image, image], source="page.png")

    def test_output_never_overwrites_the_source_image(self) -> None:
        image = create_synthetic_webtoon()
        with tempfile.TemporaryDirectory(prefix="toontra-source-") as directory:
            source = Path(directory) / "cleaned.png"
            source.write_bytes(b"original source bytes")
            result = Toontra(detector=two_bubble_detector()).process(image, source=source)

            with self.assertRaisesRegex(OutputExistsError, "input image"):
                result.save(source.parent, force=True)

            self.assertEqual(source.read_bytes(), b"original source bytes")

    def test_output_stores_absolute_source_path(self) -> None:
        image = create_synthetic_webtoon()
        with tempfile.TemporaryDirectory(prefix="toontra-source-") as directory:
            source = Path(directory) / "input.png"
            source.write_bytes(b"original source bytes")

            result = Toontra(detector=two_bubble_detector()).process(
                image,
                source=source,
            )

            self.assertEqual(
                result.source,
                source.resolve(),
            )

    def test_output_rejects_linked_crops_directory(self) -> None:
        result = Toontra(detector=two_bubble_detector()).process(
            create_synthetic_webtoon()
        )

        with tempfile.TemporaryDirectory(prefix="toontra-output-") as directory:
            root = Path(directory)
            output = root / "page"
            external_crops = root / "external_crops"

            output.mkdir()
            external_crops.mkdir()

            crops_link = output / "crops"

            if os.name == "nt":
                result_link = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(crops_link), str(external_crops)],
                    capture_output=True,
                    text=True,
                )
                if result_link.returncode != 0:
                    self.skipTest(f"directory junctions are unavailable: {result_link.stderr}")
            else:
                try:
                    crops_link.symlink_to(external_crops, target_is_directory=True)
                except OSError as error:
                    self.skipTest(f"directory symlinks are unavailable: {error}")

            external_file = external_crops / "bubble_999.png"
            external_file.write_bytes(b"external")

            with self.assertRaises(OutputExistsError):
                result.save(output, save_crops=True, force=True)

            self.assertEqual(external_file.read_bytes(), b"external")

    def test_long_page_tiling_finds_every_bubble_once(self) -> None:
        # Six predefined boxes, two per 1080px "page" of a 3-page stack, each
        # placed inside the tile that will own it at tile_height=1600 /
        # overlap=256 (tile0 owns center_y<1472, tile1 owns [1472,2292),
        # tile2 owns >=2292 -- see test_tiling.py for the ownership math).
        page_count = 3
        image = create_synthetic_long_webtoon(page_count=page_count)
        tile0_boxes = [Box(150, 260, 350, 340), Box(400, 760, 600, 840)]
        tile1_boxes = [Box(150, 316, 350, 396), Box(400, 616, 600, 696)]
        tile2_boxes = [Box(150, 820, 350, 900), Box(400, 1220, 600, 1300)]
        detector = SequencedStubBubbleDetector([tile0_boxes, tile1_boxes, tile2_boxes])
        result = Toontra(detector=detector, tile_height=1600, tile_overlap=256).process(image)
        self.assertEqual(len(result.bubbles), 2 * page_count)

    def test_erase_bubbles_crops_exactly_the_given_box(self) -> None:
        # detect_bubbles() is the only place the pipeline expands a box (see
        # DETECTION_EXPANSION_RATIO); erase_bubbles masks whatever boxes it is
        # given, with no further expansion of its own.
        image = create_synthetic_webtoon()
        toontra = Toontra(detector=two_bubble_detector())
        detections = toontra.detect_bubbles(image)
        seen_boxes: list[tuple[int, int, int, int]] = []

        class RecordingMasker:
            def create_mask(self, bubble: np.ndarray) -> np.ndarray:
                seen_boxes.append(bubble.shape[:2])
                return np.zeros(bubble.shape[:2], dtype=np.uint8)

        Toontra(detector=two_bubble_detector(), masker=RecordingMasker()).erase_bubbles(
            image, detections
        )
        for detection, seen_shape in zip(detections, seen_boxes, strict=True):
            self.assertEqual(seen_shape[0], detection.box.height)
            self.assertEqual(seen_shape[1], detection.box.width)

    def test_detection_expansion_ratio_grows_reported_boxes(self) -> None:
        image = create_synthetic_webtoon()
        tight = Toontra(detector=two_bubble_detector(), detection_expansion_ratio=0.0)
        expanded = Toontra(detector=two_bubble_detector(), detection_expansion_ratio=0.10)
        tight_boxes = tight.detect_bubbles(image)
        expanded_boxes = expanded.detect_bubbles(image)
        for original, grown in zip(tight_boxes, expanded_boxes, strict=True):
            self.assertGreaterEqual(grown.box.width, original.box.width)
            self.assertGreaterEqual(grown.box.height, original.box.height)
            self.assertGreater(grown.box.area, original.box.area)

    def test_detection_expansion_ratio_pads_each_axis_independently(self) -> None:
        # Locks in the historical ROI expansion this ratio must reproduce:
        # pad_x = width * ratio, pad_y = height * ratio (not a shared padding
        # derived from the box's longer side).
        box = TWO_BUBBLE_BOXES[0]
        expanded = Toontra(
            detector=StubBubbleDetector([box]), detection_expansion_ratio=0.05
        ).detect_bubbles(create_synthetic_webtoon())[0]
        pad_x = round(0.05 * box.width)
        pad_y = round(0.05 * box.height)
        expected = Box(box.x1 - pad_x, box.y1 - pad_y, box.x2 + pad_x, box.y2 + pad_y)
        self.assertEqual(expanded.box, expected)

    def test_translator_is_not_called_without_recognizer(self) -> None:
        result = Toontra(
            detector=two_bubble_detector(),
            translator=FailIfCalledTranslator(),
        ).process(
            create_synthetic_webtoon(),
            source_language="ko",
            target_language="en",
        )

        self.assertTrue(all(bubble.recognition is None for bubble in result.bubbles))
        self.assertTrue(all(bubble.translation is None for bubble in result.bubbles))


if __name__ == "__main__":
    unittest.main()
