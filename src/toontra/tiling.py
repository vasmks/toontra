from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .contracts import BubbleDetector, normalize_detections, validate_gray_mask, validate_rgb_image
from .geometry import cross_tile_non_max_suppression
from .models import Box, Detection, GrayMask, RGBImage, Tile


def make_vertical_tiles(
    image: RGBImage,
    *,
    tile_height: int,
    overlap: int,
) -> list[Tile]:
    """Split a page into vertical tiles with explicit page offsets."""
    rgb = validate_rgb_image(image)
    if tile_height <= 0:
        raise ValueError("tile_height must be positive")
    if overlap < 0 or overlap >= tile_height:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_height")

    page_height = int(rgb.shape[0])
    if page_height <= tile_height:
        return [Tile(index=0, image=rgb.copy(), x0=0, y0=0)]

    stride = tile_height - overlap
    final_start = page_height - tile_height
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)

    tiles: list[Tile] = []
    for index, y0 in enumerate(starts):
        y1 = min(page_height, y0 + tile_height)
        tiles.append(Tile(index=index, image=rgb[y0:y1].copy(), x0=0, y0=y0))
    return tiles


def globalize_detections(
    tile: Tile,
    detections: Sequence[Detection],
    *,
    page_width: int,
    page_height: int,
) -> list[Detection]:
    """Convert a tile's local detections into full-page coordinates."""
    local = normalize_detections(detections, width=tile.width, height=tile.height)
    global_detections: list[Detection] = []
    for detection in local:
        translated = detection.box.translate(tile.x0, tile.y0).clip(page_width, page_height)
        if translated is None:
            continue
        global_detections.append(
            Detection(
                box=translated,
                score=detection.score,
                label=detection.label,
                tile_id=tile.index,
            )
        )
    return global_detections


def ownership_boundaries(tiles: Sequence[Tile]) -> tuple[int, ...]:
    """Return doubled global y boundaries between adjacent tile ownership zones.

    A doubled coordinate keeps half-pixel overlap midpoints exact without
    floating-point arithmetic. The upper tile owns centers below a boundary;
    the lower tile owns centers on or below it.
    """
    ordered = list(tiles)
    if len({tile.index for tile in ordered}) != len(ordered):
        raise ValueError("tile indices must be unique")

    boundaries: list[int] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.y0 <= previous.y0:
            raise ValueError("tiles must be ordered by increasing y origin")
        if current.y0 > previous.y1:
            raise ValueError("tiles must overlap or touch without a vertical gap")
        if current.y1 <= previous.y1:
            raise ValueError("each tile must extend below the previous tile")
        if (current.x0, current.x1) != (previous.x0, previous.x1):
            raise ValueError("ownership supports aligned vertical tiles only")
        # The overlap is [current.y0, previous.y1). Its midpoint, multiplied
        # by two, is the sum of those endpoints.
        boundaries.append(current.y0 + previous.y1)
    return tuple(boundaries)


def detection_ownership_flags(
    tiles: Sequence[Tile],
    detections: Sequence[Detection],
    *,
    page_width: int,
    page_height: int,
) -> list[bool]:
    """Return whether each detection belongs to its source tile's ownership zone."""
    if page_width <= 0 or page_height <= 0:
        raise ValueError("page width and height must be positive")

    ordered = list(tiles)
    if not ordered:
        raise ValueError("tiles cannot be empty")
    if ordered[0].y0 != 0 or ordered[-1].y1 != page_height:
        raise ValueError("tiles must cover the page from top to bottom")
    if any(
        tile.width <= 0
        or tile.height <= 0
        or tile.x0 != 0
        or tile.x1 != page_width
        or tile.y0 < 0
        or tile.y1 > page_height
        for tile in ordered
    ):
        raise ValueError("every tile must be a valid full-width slice of the page")

    boundaries = ownership_boundaries(ordered)
    positions = {tile.index: position for position, tile in enumerate(ordered)}

    owned: list[bool] = []
    for candidate_index, detection in enumerate(detections):
        if not isinstance(detection, Detection):
            raise ValueError(f"candidate {candidate_index} is not a Detection")
        if detection.tile_id is None or detection.tile_id not in positions:
            raise ValueError("every ownership candidate must reference a known tile_id")

        position = positions[detection.tile_id]
        lower = boundaries[position - 1] if position > 0 else None
        upper = boundaries[position] if position < len(boundaries) else None
        center_y_twice = detection.box.y1 + detection.box.y2

        box = detection.box
        if box.x1 < 0 or box.y1 < 0 or box.x2 > page_width or box.y2 > page_height:
            raise ValueError("ownership candidates must already be clipped to the page")

        owned.append(
            (lower is None or center_y_twice >= lower)
            and (upper is None or center_y_twice < upper)
        )

    return owned

def select_owned_detections(
    tiles: Sequence[Tile],
    detections: Sequence[Detection],
    *,
    page_width: int,
    page_height: int,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.8,
) -> list[Detection]:
    """Resolve cross-tile duplicates with source-tile ownership as preference."""
    owned = detection_ownership_flags(
        tiles,
        detections,
        page_width=page_width,
        page_height=page_height,
    )

    return cross_tile_non_max_suppression(
        detections,
        owned=owned,
        iou_threshold=iou_threshold,
        containment_threshold=containment_threshold,
    )


def detect_on_tiles(
    detector: BubbleDetector,
    image: RGBImage,
    *,
    tile_height: int,
    overlap: int,
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Detect on overlapping vertical tiles and resolve seam duplicates.

    Ownership ranks cross-tile duplicate candidates, while IoU and containment
    provide the fallback when detector jitter leaves neither copy owned.
    """
    rgb = validate_rgb_image(image)
    page_height, page_width = rgb.shape[:2]
    tiles = make_vertical_tiles(rgb, tile_height=tile_height, overlap=overlap)

    candidates: list[Detection] = []
    for tile in tiles:
        candidates.extend(
            globalize_detections(
                tile,
                detector.detect(tile.image.copy()),
                page_width=page_width,
                page_height=page_height,
            )
        )

    return select_owned_detections(
        tiles,
        candidates,
        page_width=page_width,
        page_height=page_height,
        iou_threshold=iou_threshold,
    )


def add_mask_patch(page_mask: GrayMask, box: Box, local_mask: GrayMask) -> None:
    """Union one local coverage mask into a page mask in place."""
    if not isinstance(page_mask, np.ndarray) or page_mask.dtype != np.uint8:
        raise ValueError("page_mask must be a uint8 NumPy array")
    if page_mask.ndim != 2:
        raise ValueError("page_mask must be two-dimensional")
    clipped = box.clip(page_mask.shape[1], page_mask.shape[0])
    if clipped != box:
        raise ValueError("box must lie inside page_mask")
    validate_gray_mask(local_mask, (box.height, box.width), name="local_mask")
    rows, columns = box.as_slices()
    page_mask[rows, columns] = np.maximum(page_mask[rows, columns], local_mask)
