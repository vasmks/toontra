from __future__ import annotations

import unittest

import cv2
import numpy as np

from toontra.errors import ImageValidationError
from toontra.modules import WhiteBubbleMasker


def bubble_page() -> np.ndarray:
    page = np.full((180, 260, 3), 35, dtype=np.uint8)
    cv2.ellipse(page, (130, 90), (62, 38), 0, 0, 360, (250, 250, 250), -1)
    cv2.ellipse(page, (130, 90), (62, 38), 0, 0, 360, (5, 5, 5), 3)
    cv2.line(page, (108, 90), (152, 90), (10, 10, 10), 5)
    return page


class WhiteBubbleMaskerTests(unittest.TestCase):
    def test_masker_fills_text_holes(self) -> None:
        crop = bubble_page()[45:136, 65:196].copy()
        original = crop.copy()
        mask = WhiteBubbleMasker().create_mask(crop)
        np.testing.assert_array_equal(crop, original)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, crop.shape[:2])
        self.assertEqual(mask[45, 65], 255)
        self.assertEqual(mask[0, 0], 0)

    def test_masker_selects_bubble_in_white_crop(self) -> None:
        crop = np.full((100, 160, 3), 255, dtype=np.uint8)
        cv2.ellipse(crop, (80, 50), (50, 30), 0, 0, 360, (0, 0, 0), 3)
        cv2.line(crop, (70, 50), (90, 50), (0, 0, 0), 3)
        mask = WhiteBubbleMasker().create_mask(crop)
        self.assertEqual(mask[50, 80], 255)
        self.assertEqual(mask[0, 0], 0)

    def test_masker_rejects_invalid_images(self) -> None:
        bad_images = (
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.float32),
            np.zeros((0, 10, 3), dtype=np.uint8),
        )
        for image in bad_images:
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaises(ImageValidationError):
                    WhiteBubbleMasker().create_mask(image)


if __name__ == "__main__":
    unittest.main()
