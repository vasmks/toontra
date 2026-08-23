"""Deterministic detector stand-ins, for tests only.

Neither class is exported from `toontra.modules` or presented as a usable
detector implementation -- they exist so pipeline/CLI/tiling tests can assert
on known, predefined boxes instead of a real model's output.
"""

from __future__ import annotations

from collections.abc import Sequence

from toontra.models import Box, Detection, RGBImage


class StubBubbleDetector:
    """Returns the same predefined detections for every call."""

    def __init__(self, boxes: Sequence[Box]) -> None:
        self._detections = [Detection(box) for box in boxes]

    def detect(self, image: RGBImage) -> list[Detection]:
        del image
        return list(self._detections)


class SequencedStubBubbleDetector:
    """Returns one predefined box list per call, in a fixed order.

    Useful for tiling tests: each tile is detected with a separate `detect()`
    call, in tile order, so a per-call sequence can simulate a bubble that a
    real detector would report from two overlapping tile crops at once.
    """

    def __init__(self, box_lists: Sequence[Sequence[Box]]) -> None:
        self._box_lists = [list(boxes) for boxes in box_lists]
        self._call_count = 0

    def detect(self, image: RGBImage) -> list[Detection]:
        del image
        if self._call_count >= len(self._box_lists):
            raise AssertionError("SequencedStubBubbleDetector called more times than expected")
        boxes = self._box_lists[self._call_count]
        self._call_count += 1
        return [Detection(box) for box in boxes]
