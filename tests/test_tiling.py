from __future__ import annotations

import unittest

import numpy as np
from stub_detector import SequencedStubBubbleDetector

from toontra.models import Box, Detection, Tile
from toontra.tiling import (
    add_mask_patch,
    detect_on_tiles,
    globalize_detections,
    make_vertical_tiles,
    ownership_boundaries,
    select_owned_detections,
)


def make_tiles(spans: list[tuple[int, int]], *, width: int = 20) -> list[Tile]:
    return [
        Tile(index, np.zeros((y1 - y0, width, 3), dtype=np.uint8), 0, y0)
        for index, (y0, y1) in enumerate(spans)
    ]


class TilingTests(unittest.TestCase):
    def test_short_image_creates_one_tile(self) -> None:
        image = np.zeros((700, 100, 3), dtype=np.uint8)
        tiles = make_vertical_tiles(image, tile_height=1024, overlap=256)
        self.assertEqual([(tile.y0, tile.y1) for tile in tiles], [(0, 700)])

    def test_long_image_has_complete_nonredundant_coverage(self) -> None:
        image = np.zeros((2500, 100, 3), dtype=np.uint8)
        tiles = make_vertical_tiles(image, tile_height=1024, overlap=256)
        self.assertEqual(
            [(tile.y0, tile.y1) for tile in tiles],
            [(0, 1024), (768, 1792), (1476, 2500)],
        )
        self.assertTrue(all(tile.height <= 1024 for tile in tiles))
        coverage = np.zeros(image.shape[0], dtype=np.uint8)
        for tile in tiles:
            coverage[tile.y0 : tile.y1] += 1
        self.assertTrue(np.all(coverage > 0))

    def test_final_tile_is_shifted_instead_of_stretched(self) -> None:
        image = np.zeros((1800, 100, 3), dtype=np.uint8)
        tiles = make_vertical_tiles(image, tile_height=1024, overlap=256)
        self.assertEqual(
            [(tile.y0, tile.y1) for tile in tiles],
            [(0, 1024), (768, 1792), (776, 1800)],
        )

    def test_page_just_over_one_tile_keeps_the_height_limit(self) -> None:
        image = np.zeros((1025, 100, 3), dtype=np.uint8)
        tiles = make_vertical_tiles(image, tile_height=1024, overlap=256)
        self.assertEqual([(tile.y0, tile.y1) for tile in tiles], [(0, 1024), (1, 1025)])

    def test_regular_tiles_are_not_replaced_by_the_shifted_final_tile(self) -> None:
        cases = [
            (1601, [(0, 1600), (1, 1601)]),
            (3199, [(0, 1600), (1344, 2944), (1599, 3199)]),
            (3200, [(0, 1600), (1344, 2944), (1600, 3200)]),
        ]
        for page_height, expected in cases:
            with self.subTest(page_height=page_height):
                image = np.zeros((page_height, 100, 3), dtype=np.uint8)
                tiles = make_vertical_tiles(image, tile_height=1600, overlap=256)
                self.assertEqual([(tile.y0, tile.y1) for tile in tiles], expected)

    def test_invalid_tiling_options_raise(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        for tile_height, overlap in ((0, 0), (10, -1), (10, 10)):
            with self.subTest(tile_height=tile_height, overlap=overlap):
                with self.assertRaises(ValueError):
                    make_vertical_tiles(image, tile_height=tile_height, overlap=overlap)

    def test_local_box_is_globalized_with_real_tile_origin(self) -> None:
        tile = Tile(2, np.zeros((400, 200, 3), dtype=np.uint8), 0, 768)
        local = Detection(Box(10, 132, 110, 332), 0.8)
        result = globalize_detections(
            tile, [local], page_width=200, page_height=1600
        )
        self.assertEqual(result[0].box, Box(10, 900, 110, 1100))
        self.assertEqual(result[0].tile_id, 2)

    def test_duplicate_bubble_in_overlap_is_returned_once(self) -> None:
        # tile0=[0,100), tile1=[60,160), ownership boundary at actual y=80.
        # A bubble spanning y=63-87 (center 75, tile0's side of the
        # boundary) falls inside both tile crops, so a real detector would
        # report it from each; only tile0's copy should survive.
        image = np.zeros((160, 120, 3), dtype=np.uint8)
        detector = SequencedStubBubbleDetector(
            [
                [Box(32, 63, 88, 87)],  # tile0, local == global (y0=0)
                [Box(32, 3, 88, 27)],  # tile1, local = global - 60
            ]
        )
        detections = detect_on_tiles(
            detector,
            image,
            tile_height=100,
            overlap=40,
            iou_threshold=0.5,
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].tile_id, 0)

    def test_jittered_seam_copies_cannot_both_fail_ownership(self) -> None:
        # tile0=[0,100), tile1=[60,160), ownership boundary at y=80.
        # Detector jitter puts each copy's center in the other tile's zone.
        image = np.zeros((160, 120, 3), dtype=np.uint8)
        detector = SequencedStubBubbleDetector(
            [
                [Box(32, 72, 88, 92)],  # global center=82, outside tile0 ownership
                [Box(32, 8, 88, 28)],  # global y=68..88, center=78, outside tile1
            ]
        )

        detections = detect_on_tiles(
            detector,
            image,
            tile_height=100,
            overlap=40,
            iou_threshold=0.5,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].tile_id, 0)

    def test_unique_unowned_seam_detection_survives(self) -> None:
        image = np.zeros((160, 120, 3), dtype=np.uint8)
        detector = SequencedStubBubbleDetector(
            [
                [Box(32, 72, 88, 92)],  # center=82, outside tile0 ownership
                [],
            ]
        )

        detections = detect_on_tiles(
            detector,
            image,
            tile_height=100,
            overlap=40,
            iou_threshold=0.5,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].tile_id, 0)

    def test_same_tile_overlaps_survive_seam_deduplication(self) -> None:
        image = np.zeros((160, 120, 3), dtype=np.uint8)
        detector = SequencedStubBubbleDetector(
            [
                [Box(20, 10, 80, 60), Box(22, 10, 82, 60)],
                [],
            ]
        )

        detections = detect_on_tiles(
            detector,
            image,
            tile_height=100,
            overlap=40,
            iou_threshold=0.5,
        )

        self.assertEqual(len(detections), 2)
        self.assertEqual({item.tile_id for item in detections}, {0})

    def test_ownership_uses_the_actual_overlap_midpoint(self) -> None:
        tiles = make_tiles([(0, 100), (50, 150)])
        self.assertEqual(ownership_boundaries(tiles), (150,))
        box = Box(2, 65, 12, 85)
        candidates = [
            Detection(box, 0.8, tile_id=0),
            Detection(box, 0.9, tile_id=1),
        ]

        selected = select_owned_detections(
            tiles,
            candidates,
            page_width=20,
            page_height=150,
        )

        self.assertEqual(selected, [candidates[1]])

    def test_ownership_handles_a_shifted_final_tile_and_half_pixel_boundary(self) -> None:
        shifted = make_tiles([(0, 1024), (776, 1800)])
        self.assertEqual(ownership_boundaries(shifted), (1800,))

        almost_one_tile = make_tiles([(0, 1024), (1, 1025)])
        self.assertEqual(ownership_boundaries(almost_one_tile), (1025,))
        candidates = [
            Detection(Box(1, 500, 9, 524), 0.7, tile_id=0),
            Detection(Box(10, 500, 18, 525), 0.8, tile_id=1),
        ]
        selected = select_owned_detections(
            almost_one_tile,
            [candidates[1], candidates[0]],
            page_width=20,
            page_height=1025,
        )
        self.assertEqual(selected, candidates)

    def test_ownership_supports_touching_and_triple_overlap_tiles(self) -> None:
        touching = make_tiles([(0, 100), (100, 200)])
        self.assertEqual(ownership_boundaries(touching), (200,))

        triple = make_tiles([(0, 100), (20, 120), (40, 140)])
        self.assertEqual(ownership_boundaries(triple), (120, 160))
        box = Box(2, 60, 12, 80)
        candidates = [Detection(box, 0.8, tile_id=index) for index in range(3)]
        selected = select_owned_detections(
            triple,
            candidates,
            page_width=20,
            page_height=140,
        )
        self.assertEqual(selected, [candidates[1]])

    def test_ownership_rejects_invalid_tile_plans_and_unknown_sources(self) -> None:
        candidates = [Detection(Box(1, 10, 9, 20), 0.8, tile_id=4)]
        with self.assertRaisesRegex(ValueError, "known tile_id"):
            select_owned_detections(
                make_tiles([(0, 100)]),
                candidates,
                page_width=20,
                page_height=100,
            )
        with self.assertRaisesRegex(ValueError, "gap"):
            ownership_boundaries(make_tiles([(0, 50), (60, 100)]))
        with self.assertRaisesRegex(ValueError, "top to bottom"):
            select_owned_detections(
                make_tiles([(0, 90)]),
                [],
                page_width=20,
                page_height=100,
            )

    def test_detect_on_tiles_deduplicates_across_multiple_seams(self) -> None:
        # A 3240px page tiles into tile0=[0,1600), tile1=[1344,2944),
        # tile2=[1640,3240) at tile_height=1600/overlap=256, with ownership
        # boundaries at actual y=1472 and y=2292 (see test_tiling.py's
        # boundary tests for the doubled-coordinate math). One bubble sits
        # on the tile0/tile1 seam and is reported by both tiles; a second,
        # unambiguous bubble is reported only by tile2.
        image = np.zeros((3240, 720, 3), dtype=np.uint8)
        detector = SequencedStubBubbleDetector(
            [
                [Box(100, 1430, 300, 1510)],  # tile0 (y0=0): seam bubble, center 1470
                [Box(100, 86, 300, 166)],  # tile1 (y0=1344): same seam bubble
                [Box(100, 1160, 300, 1240)],  # tile2 (y0=1640): unambiguous bubble
            ]
        )
        tiled = detect_on_tiles(
            detector,
            image,
            tile_height=1600,
            overlap=256,
            iou_threshold=0.5,
        )

        self.assertEqual(len(tiled), 2)
        self.assertEqual({item.tile_id for item in tiled}, {0, 2})
        self.assertTrue(all(item.box.y2 <= image.shape[0] for item in tiled))

    def test_overlapping_mask_patches_are_combined_as_union(self) -> None:
        page_mask = np.zeros((8, 8), dtype=np.uint8)
        add_mask_patch(page_mask, Box(1, 1, 5, 5), np.full((4, 4), 255, np.uint8))
        add_mask_patch(page_mask, Box(3, 3, 7, 7), np.zeros((4, 4), np.uint8))
        self.assertEqual(page_mask[3, 3], 255)
        self.assertEqual(page_mask[6, 6], 0)


if __name__ == "__main__":
    unittest.main()
