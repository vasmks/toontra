"""Weight-free masking for light speech bubbles."""

from __future__ import annotations

import cv2
import numpy as np

from toontra.contracts import validate_rgb_image
from toontra.models import GrayMask, ModelMetadata, RGBImage


class WhiteBubbleMasker:
    """Create a filled mask for a white bubble interior.

    Dark letters inside the selected white component are treated as holes and
    filled.  Applying the resulting mask therefore erases both the white
    interior and its text in one pass, while retaining the dark outline.
    """

    metadata = ModelMetadata(
        name="Toontra white-component masker",
        version="0.1.0",
        source_url="https://github.com/opencv/opencv",
        license_spdx="Apache-2.0",
    )

    def __init__(
        self,
        *,
        white_threshold: int = 205,
        close_kernel_size: int = 3,
        min_area_ratio: float = 0.03,
    ) -> None:
        if not 0 <= white_threshold <= 255:
            raise ValueError("white_threshold must be between 0 and 255")
        if close_kernel_size < 1 or close_kernel_size % 2 == 0:
            raise ValueError("close_kernel_size must be a positive odd number")
        if not 0 <= min_area_ratio <= 1:
            raise ValueError("min_area_ratio must be between 0 and 1")
        self.white_threshold = white_threshold
        self.close_kernel_size = close_kernel_size
        self.min_area_ratio = min_area_ratio

    def create_mask(self, bubble_crop: RGBImage) -> GrayMask:
        crop = validate_rgb_image(bubble_crop, name="bubble crop")
        height, width = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        binary = np.where(gray >= self.white_threshold, 255, 0).astype(np.uint8)

        if self.close_kernel_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.close_kernel_size, self.close_kernel_size),
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        minimum_area = height * width * self.min_area_ratio
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        candidates: list[tuple[float, int, int]] = []

        for component_id in range(1, count):
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            component_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            component_labels = labels[
                y : y + component_height,
                x : x + component_width,
            ]
            local_ys, local_xs = np.where(component_labels == component_id)
            xs = local_xs + x
            ys = local_ys + y
            distance_squared = np.min((xs - center_x) ** 2 + (ys - center_y) ** 2)
            candidates.append((float(distance_squared), -area, component_id))

        if not candidates:
            return np.zeros((height, width), dtype=np.uint8)

        _, _, selected_id = min(candidates)
        component = np.where(labels == selected_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((height, width), dtype=np.uint8)

        selected = max(contours, key=cv2.contourArea)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [selected], contourIdx=-1, color=255, thickness=cv2.FILLED)
        return mask
