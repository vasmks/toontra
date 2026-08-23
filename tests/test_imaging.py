from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from toontra.errors import ImageValidationError
from toontra.imaging import (
    alpha_composite,
    apply_erase_mask,
    load_rgb,
    mask_to_rgba,
    save_rgb,
)


class ImagingTests(unittest.TestCase):
    def test_erase_mask_supports_partial_coverage_without_mutation(self) -> None:
        image = np.zeros((1, 3, 3), dtype=np.uint8)
        mask = np.array([[0, 128, 255]], dtype=np.uint8)
        image_before = image.copy()
        output = apply_erase_mask(image, mask)
        np.testing.assert_array_equal(image, image_before)
        self.assertEqual(output[0].tolist(), [[0, 0, 0], [128, 128, 128], [255, 255, 255]])

    def test_rgba_overlay_uses_alpha_not_rgb_as_coverage(self) -> None:
        mask = np.array([[0, 128, 255]], dtype=np.uint8)
        overlay = mask_to_rgba(mask, fill_rgb=(255, 255, 255))
        self.assertEqual(overlay[0, 0].tolist(), [255, 255, 255, 0])
        self.assertEqual(overlay[0, 1].tolist(), [255, 255, 255, 128])

    def test_alpha_composite_handles_zero_half_and_full_opacity(self) -> None:
        base = np.zeros((1, 3, 3), dtype=np.uint8)
        overlay = np.full((1, 3, 4), 255, dtype=np.uint8)
        base_before = base.copy()
        overlay[0, :, 3] = (0, 128, 255)
        overlay_before = overlay.copy()
        output = alpha_composite(base, overlay)
        self.assertEqual(output[0].tolist(), [[0, 0, 0], [128, 128, 128], [255, 255, 255]])
        np.testing.assert_array_equal(base, base_before)
        np.testing.assert_array_equal(overlay, overlay_before)

    def test_zero_alpha_ignores_overlay_color(self) -> None:
        base = np.array([[[20, 40, 60]]], dtype=np.uint8)
        overlay = np.array([[[200, 10, 90, 0]]], dtype=np.uint8)
        np.testing.assert_array_equal(alpha_composite(base, overlay), base)

    def test_unicode_path_round_trip(self) -> None:
        image = np.zeros((6, 7, 3), dtype=np.uint8)
        image[:, :, 0] = 200
        with tempfile.TemporaryDirectory(prefix="toontra-image-") as directory:
            path = Path(directory) / "sample_한글.png"
            save_rgb(path, image)
            np.testing.assert_array_equal(load_rgb(path), image)

    def test_transparent_input_is_flattened_on_white(self) -> None:
        bgra = np.array([[[0, 0, 0, 0], [0, 0, 255, 128]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory(prefix="toontra-alpha-") as directory:
            path = Path(directory) / "transparent.png"
            success, encoded = cv2.imencode(".png", bgra)
            self.assertTrue(success)
            encoded.tofile(path)
            loaded = load_rgb(path)
        self.assertEqual(loaded[0].tolist(), [[255, 255, 255], [255, 127, 127]])

    def test_16_bit_input_is_rejected_at_load_boundary(self) -> None:
        image = np.full((2, 2), 1024, dtype=np.uint16)
        with tempfile.TemporaryDirectory(prefix="toontra-16bit-") as directory:
            path = Path(directory) / "page.png"
            success, encoded = cv2.imencode(".png", image)
            self.assertTrue(success)
            encoded.tofile(path)
            with self.assertRaisesRegex(ImageValidationError, "8-bit"):
                load_rgb(path)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ImageValidationError):
            apply_erase_mask(
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.zeros((3, 4), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
