from __future__ import annotations

from collections.abc import Sequence

from .models import Box, Detection


def intersection_area(first: Box, second: Box) -> int:
    width = max(0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0, min(first.y2, second.y2) - max(first.y1, second.y1))
    return width * height


def iou(first: Box, second: Box) -> float:
    intersection = intersection_area(first, second)
    if intersection == 0:
        return 0.0
    union = first.area + second.area - intersection
    return intersection / union


def intersection_over_smaller(first: Box, second: Box) -> float:
    intersection = intersection_area(first, second)
    if intersection == 0:
        return 0.0
    return intersection / min(first.area, second.area)


def non_max_suppression(
    detections: Sequence[Detection],
    *,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.8,
    class_aware: bool = True,
) -> list[Detection]:
    """Remove duplicate global detections and keep the strongest candidate."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    if not 0.0 <= containment_threshold <= 1.0:
        raise ValueError("containment_threshold must be between 0 and 1")

    indexed = list(enumerate(detections))
    indexed.sort(key=lambda item: (-item[1].score, -item[1].box.area, item[0]))
    kept: list[Detection] = []

    for _, candidate in indexed:
        # A candidate can be a cross-tile "containment" duplicate of more than
        # one already-kept box (e.g. two smaller seam fragments of the same
        # object). Check it against every kept box before mutating `kept`,
        # instead of stopping at the first match, so a bigger replacement
        # candidate can't leave a second, smaller duplicate behind.
        absorbed_by_larger_box = False
        superseded_indices: list[int] = []
        for selected_index, selected in enumerate(kept):
            if class_aware and candidate.label != selected.label:
                continue
            overlap_area = intersection_area(candidate.box, selected.box)
            if overlap_area == 0:
                continue
            from_different_tiles = (
                candidate.tile_id is not None
                and selected.tile_id is not None
                and candidate.tile_id != selected.tile_id
            )
            containment_duplicate = from_different_tiles and (
                intersection_over_smaller(candidate.box, selected.box)
                >= containment_threshold
            )
            if containment_duplicate:
                # A crop boundary can make a partial component look more
                # confident than the complete component in its neighbor. For
                # cross-tile containment, area is the more reliable signal.
                if candidate.box.area > selected.box.area:
                    superseded_indices.append(selected_index)
                else:
                    absorbed_by_larger_box = True
                continue
            if iou(candidate.box, selected.box) >= iou_threshold:
                absorbed_by_larger_box = True
        if absorbed_by_larger_box:
            continue
        if superseded_indices:
            kept = [item for index, item in enumerate(kept) if index not in superseded_indices]
        kept.append(candidate)

    kept.sort(key=lambda item: (item.box.y1, item.box.x1, -item.score))
    return kept


def cross_tile_non_max_suppression(
    detections: Sequence[Detection],
    *,
    owned: Sequence[bool],
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.8,
    class_aware: bool = True,
) -> list[Detection]:
    """Deduplicate only detections from different tiles.

    Ownership ranks candidates before score. It is deliberately a preference,
    not a filter: unowned candidates remain eligible when no better cross-tile
    copy exists. Cross-tile containment favors the larger box when ownership is
    equal, matching the crop-fragment handling in :func:`non_max_suppression`.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    if not 0.0 <= containment_threshold <= 1.0:
        raise ValueError("containment_threshold must be between 0 and 1")
    if len(owned) != len(detections):
        raise ValueError("owned must have one value per detection")

    indexed = [
        (index, detection, bool(is_owned))
        for index, (detection, is_owned) in enumerate(
            zip(detections, owned, strict=True)
        )
    ]
    indexed.sort(
        key=lambda item: (
            -int(item[2]),
            -item[1].score,
            -item[1].box.area,
            item[0],
        )
    )
    kept: list[tuple[Detection, bool]] = []

    for _, candidate, candidate_owned in indexed:
        absorbed = False
        superseded_indices: list[int] = []
        for selected_index, (selected, selected_owned) in enumerate(kept):
            if candidate.tile_id is None or selected.tile_id is None:
                continue
            if candidate.tile_id == selected.tile_id:
                continue
            if class_aware and candidate.label != selected.label:
                continue
            overlap_area = intersection_area(candidate.box, selected.box)
            if overlap_area == 0:
                continue
            containment_duplicate = (
                intersection_over_smaller(candidate.box, selected.box)
                >= containment_threshold
            )
            if containment_duplicate:
                if candidate_owned != selected_owned:
                    # Owned candidates are sorted first, so the already-kept
                    # candidate is the preferred one in this case.
                    absorbed = True
                elif candidate.box.area > selected.box.area:
                    superseded_indices.append(selected_index)
                else:
                    absorbed = True
                continue
            if iou(candidate.box, selected.box) >= iou_threshold:
                absorbed = True
        if absorbed:
            continue
        if superseded_indices:
            kept = [item for index, item in enumerate(kept) if index not in superseded_indices]
        kept.append((candidate, candidate_owned))

    result = [detection for detection, _ in kept]
    result.sort(key=lambda item: (item.box.y1, item.box.x1, -item.score))
    return result
