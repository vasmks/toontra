from __future__ import annotations

import unittest

from toontra.geometry import (
    cross_tile_non_max_suppression,
    intersection_over_smaller,
    iou,
    non_max_suppression,
)
from toontra.models import Box, Detection


class GeometryTests(unittest.TestCase):
    def test_box_uses_half_open_coordinates(self) -> None:
        box = Box(10, 20, 30, 55)
        self.assertEqual(box.width, 20)
        self.assertEqual(box.height, 35)
        self.assertEqual(box.area, 700)
        self.assertEqual(box.as_slices(), (slice(20, 55), slice(10, 30)))

    def test_box_clip_and_translate(self) -> None:
        self.assertEqual(Box(-5, -2, 20, 30).clip(15, 12), Box(0, 0, 15, 12))
        self.assertEqual(Box(1, 2, 4, 8).translate(10, 100), Box(11, 102, 14, 108))
        self.assertIsNone(Box(20, 20, 30, 30).clip(10, 10))

    def test_box_expand_pads_and_clips(self) -> None:
        self.assertEqual(Box(10, 10, 20, 20).expand(2, 2, 100, 100), Box(8, 8, 22, 22))
        self.assertEqual(Box(0, 0, 5, 5).expand(3, 3, 100, 100), Box(0, 0, 8, 8))
        with self.assertRaises(ValueError):
            Box(0, 0, 5, 5).expand(-1, 0, 100, 100)

    def test_box_expand_pads_each_axis_independently(self) -> None:
        # pad_x only widens left/right; pad_y only widens top/bottom.
        self.assertEqual(Box(10, 10, 20, 30).expand(1, 4, 100, 100), Box(9, 6, 21, 34))

    def test_touching_boxes_have_zero_iou(self) -> None:
        self.assertEqual(iou(Box(0, 0, 10, 10), Box(10, 0, 20, 10)), 0.0)

    def test_overlap_values_are_exact(self) -> None:
        first = Box(0, 0, 10, 10)
        second = Box(5, 0, 15, 10)
        self.assertAlmostEqual(iou(first, second), 1 / 3)
        self.assertAlmostEqual(intersection_over_smaller(first, second), 0.5)

    def test_nms_keeps_highest_score_and_reading_order(self) -> None:
        detections = [
            Detection(Box(100, 100, 160, 160), 0.7, tile_id=0),
            Detection(Box(101, 100, 161, 160), 0.9, tile_id=1),
            Detection(Box(10, 10, 40, 40), 0.8, tile_id=0),
        ]
        kept = non_max_suppression(detections, iou_threshold=0.5)
        self.assertEqual([item.score for item in kept], [0.8, 0.9])

    def test_nms_keeps_different_labels(self) -> None:
        box = Box(0, 0, 20, 20)
        kept = non_max_suppression(
            [Detection(box, 0.9, "bubble"), Detection(box, 0.8, "text")]
        )
        self.assertEqual(len(kept), 2)

    def test_zero_threshold_does_not_merge_disjoint_boxes(self) -> None:
        detections = [
            Detection(Box(0, 0, 10, 10), 0.9, tile_id=0),
            Detection(Box(20, 20, 30, 30), 0.8, tile_id=1),
        ]
        kept = non_max_suppression(
            detections,
            iou_threshold=0.0,
            containment_threshold=0.0,
        )
        self.assertEqual(kept, detections)

    def test_cross_tile_containment_removes_a_seam_fragment(self) -> None:
        full = Detection(Box(10, 10, 70, 70), 0.8, tile_id=0)
        fragment = Detection(Box(10, 25, 70, 50), 0.95, tile_id=1)
        self.assertLess(iou(full.box, fragment.box), 0.5)
        self.assertEqual(intersection_over_smaller(full.box, fragment.box), 1.0)
        kept = non_max_suppression([fragment, full], iou_threshold=0.5)
        self.assertEqual(kept, [full])

    def test_a_larger_replacement_absorbs_every_fragment_it_contains(self) -> None:
        # Three tile fragments of one physical object. The full reconstruction
        # has the lowest score, so it is processed last and must replace both
        # earlier fragments, not just the first one it is compared against.
        top = Detection(Box(10, 10, 70, 40), 0.90, tile_id=0)
        bottom = Detection(Box(10, 35, 70, 70), 0.85, tile_id=1)
        full = Detection(Box(10, 10, 70, 70), 0.60, tile_id=2)
        kept = non_max_suppression([top, bottom, full], iou_threshold=0.5)
        self.assertEqual(kept, [full])

    def test_cross_tile_nms_does_not_compare_same_tile_detections(self) -> None:
        first = Detection(Box(10, 10, 70, 70), 0.9, tile_id=0)
        second = Detection(Box(12, 10, 72, 70), 0.8, tile_id=0)

        kept = cross_tile_non_max_suppression(
            [first, second],
            owned=[True, True],
            iou_threshold=0.5,
        )

        self.assertEqual(kept, [first, second])

    def test_cross_tile_nms_uses_ownership_as_a_preference(self) -> None:
        unowned = Detection(Box(10, 10, 70, 70), 0.95, tile_id=0)
        owned = Detection(Box(12, 10, 72, 70), 0.70, tile_id=1)

        kept = cross_tile_non_max_suppression(
            [unowned, owned],
            owned=[False, True],
            iou_threshold=0.5,
        )

        self.assertEqual(kept, [owned])

    def test_cross_tile_nms_keeps_containment_fragment_handling(self) -> None:
        full = Detection(Box(10, 10, 70, 70), 0.70, tile_id=0)
        fragment = Detection(Box(10, 25, 70, 50), 0.95, tile_id=1)

        kept = cross_tile_non_max_suppression(
            [fragment, full],
            owned=[False, False],
            iou_threshold=0.5,
        )

        self.assertEqual(kept, [full])


if __name__ == "__main__":
    unittest.main()
