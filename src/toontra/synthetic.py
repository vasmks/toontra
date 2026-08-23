from __future__ import annotations

import cv2
import numpy as np

from .models import RGBImage


def create_synthetic_webtoon(width: int = 720, height: int = 1080) -> RGBImage:
    """Create a small original page used by the offline demo and tests."""
    if width < 480 or height < 720:
        raise ValueError("Synthetic page must be at least 480 x 720 pixels")

    page = np.full((height, width, 3), (244, 239, 225), dtype=np.uint8)
    margin = max(18, width // 30)
    gutter = max(14, height // 70)
    panel_height = (height - margin * 2 - gutter) // 2

    top = (margin, margin, width - margin, margin + panel_height)
    bottom = (margin, top[3] + gutter, width - margin, height - margin)
    cv2.rectangle(page, top[:2], top[2:], (94, 142, 154), thickness=-1)
    cv2.rectangle(page, bottom[:2], bottom[2:], (157, 119, 103), thickness=-1)
    cv2.rectangle(page, top[:2], top[2:], (25, 31, 39), thickness=3)
    cv2.rectangle(page, bottom[:2], bottom[2:], (25, 31, 39), thickness=3)

    _draw_character(page, (width // 4, top[3] - 115), 1.0)
    _draw_character(page, (width * 3 // 4, bottom[3] - 125), 1.1)

    _draw_bubble(
        page,
        center=(width * 2 // 3, margin + panel_height // 3),
        axes=(width // 5, panel_height // 7),
        tail=(width // 2, margin + panel_height * 2 // 3),
        lines=("HELLO", "TOONTRA"),
    )
    _draw_bubble(
        page,
        center=(width // 3, bottom[1] + panel_height // 3),
        axes=(width // 5, panel_height // 7),
        tail=(width // 2, bottom[1] + panel_height * 2 // 3),
        lines=("MODULAR", "PIPELINE"),
    )
    return page


def create_synthetic_long_webtoon(
    page_count: int = 8,
    *,
    width: int = 720,
    page_height: int = 1080,
) -> RGBImage:
    """Stack original demo pages into a deterministic long-scroll image."""

    if page_count < 1:
        raise ValueError("page_count must be at least 1")
    section = create_synthetic_webtoon(width=width, height=page_height)
    return np.concatenate([section] * page_count, axis=0)


def _draw_character(page: RGBImage, base: tuple[int, int], scale: float) -> None:
    x, y = base
    skin = (225, 180, 150)
    ink = (35, 31, 36)
    shirt = (63, 76, 113)
    cv2.circle(page, (x, y - int(120 * scale)), int(38 * scale), skin, thickness=-1)
    cv2.circle(page, (x, y - int(120 * scale)), int(38 * scale), ink, thickness=3)
    cv2.rectangle(
        page,
        (x - int(45 * scale), y - int(78 * scale)),
        (x + int(45 * scale), y),
        shirt,
        thickness=-1,
    )
    cv2.line(page, (x - 45, y), (x - 62, y + 55), ink, thickness=8)
    cv2.line(page, (x + 45, y), (x + 62, y + 55), ink, thickness=8)


def _draw_bubble(
    page: RGBImage,
    *,
    center: tuple[int, int],
    axes: tuple[int, int],
    tail: tuple[int, int],
    lines: tuple[str, ...],
) -> None:
    white = (255, 255, 255)
    ink = (24, 29, 36)
    cx, cy = center
    ax, ay = axes

    base_left = (cx - ax // 3, cy + ay - 4)
    base_right = (cx - ax // 10, cy + ay - 1)
    points = np.array([base_left, base_right, tail], dtype=np.int32)
    cv2.fillConvexPoly(page, points, white)
    cv2.ellipse(page, center, axes, 0, 0, 360, white, thickness=-1)
    cv2.ellipse(page, center, axes, 0, 0, 360, ink, thickness=3)
    cv2.line(page, base_left, tail, ink, thickness=3)
    cv2.line(page, tail, base_right, ink, thickness=3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(page.shape[:2]) / 900)
    line_height = int(32 * font_scale)
    first_y = cy - (len(lines) - 1) * line_height // 2
    for index, text in enumerate(lines):
        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, 2)
        text_x = cx - text_width // 2
        text_y = first_y + index * line_height
        cv2.putText(page, text, (text_x, text_y), font, font_scale, ink, 2, cv2.LINE_AA)
