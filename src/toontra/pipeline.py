from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .contracts import (
    BubbleDetector,
    BubbleMasker,
    TextRecognizer,
    Translator,
    normalize_detections,
    validate_gray_mask,
    validate_recognition,
    validate_rgb_image,
    validate_translations,
)
from .imaging import apply_erase_mask, load_rgb, mask_to_rgba
from .models import BubbleResult, Detection, ModelMetadata, PageResult, RGBImage
from .tiling import add_mask_patch, detect_on_tiles

# Applied once, to final, already-deduplicated detection boxes (after
# tiling, ownership, and cross-tile NMS), so it never distorts seam dedup.
# A tight detection box can clip an irregular bubble outline or its tail;
# padding each side independently by this fraction of the box's own width
# (left/right) and height (top/bottom) -- roughly 10% width/height growth
# at the default -- gives both OCR and masking room to see the full shape.
# This is the only box expansion in the pipeline: no stage after this one
# (including any masker) expands the box again.
DETECTION_EXPANSION_RATIO = 0.05


class Toontra:
    """Modular speech-bubble extraction and masking pipeline."""

    def __init__(
        self,
        detector: BubbleDetector | None = None,
        masker: BubbleMasker | None = None,
        recognizer: TextRecognizer | None = None,
        translator: Translator | None = None,
        *,
        tile_height: int | None = 1600,
        tile_overlap: int = 256,
        duplicate_iou: float = 0.5,
        detection_expansion_ratio: float = DETECTION_EXPANSION_RATIO,
        fill_rgb: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        if detector is None or masker is None:
            from .modules import WhiteBubbleMasker, Yolo26BubbleDetector

            detector = detector or Yolo26BubbleDetector()
            masker = masker or WhiteBubbleMasker()
        if tile_height is not None and tile_height <= 0:
            raise ValueError("tile_height must be positive or None")
        if tile_overlap < 0:
            raise ValueError("tile_overlap cannot be negative")
        if tile_height is not None and tile_overlap >= tile_height:
            raise ValueError("tile_overlap must be smaller than tile_height")
        if not 0.0 <= duplicate_iou <= 1.0:
            raise ValueError("duplicate_iou must be between 0 and 1")
        if not 0.0 <= detection_expansion_ratio <= 1.0:
            raise ValueError("detection_expansion_ratio must be between 0 and 1")

        self.detector = detector
        self.masker = masker
        self.recognizer = recognizer
        self.translator = translator
        self.tile_height = tile_height
        self.tile_overlap = tile_overlap
        self.duplicate_iou = duplicate_iou
        self.detection_expansion_ratio = detection_expansion_ratio
        self.fill_rgb = fill_rgb

    def detect_bubbles(self, image: RGBImage) -> list[Detection]:
        rgb = validate_rgb_image(image)
        height, width = rgb.shape[:2]
        if self.tile_height is not None and height > self.tile_height:
            detections = detect_on_tiles(
                self.detector,
                rgb,
                tile_height=self.tile_height,
                overlap=self.tile_overlap,
                iou_threshold=self.duplicate_iou,
            )
        else:
            detections = normalize_detections(
                self.detector.detect(rgb.copy()), width=width, height=height
            )
        if self.detection_expansion_ratio == 0.0:
            return detections
        return [
            _expand_detection(detection, self.detection_expansion_ratio, width, height)
            for detection in detections
        ]

    def erase_bubbles(
        self,
        image: RGBImage,
        detections: list[Detection],
    ) -> tuple[RGBImage, np.ndarray]:
        rgb = validate_rgb_image(image)
        height, width = rgb.shape[:2]
        normalized = normalize_detections(detections, width=width, height=height)
        page_mask = np.zeros((height, width), dtype=np.uint8)
        mask_boxes = [item.box for item in normalized]

        for box in mask_boxes:
            rows, columns = box.as_slices()
            crop = rgb[rows, columns].copy()
            local_mask = self.masker.create_mask(crop)
            validate_gray_mask(local_mask, crop.shape[:2], name="masker output")
            add_mask_patch(page_mask, box, local_mask)

        # Apply the completed page mask only inside detected regions to avoid
        # blending the full webtoon page.
        cleaned = rgb.copy()
        for box in mask_boxes:
            rows, columns = box.as_slices()
            cleaned[rows, columns] = apply_erase_mask(
                rgb[rows, columns], page_mask[rows, columns], fill_rgb=self.fill_rgb
            )
        return cleaned, page_mask

    def merge_mask_with_image(self, image: RGBImage, mask: np.ndarray) -> RGBImage:
        """Apply an editable grayscale mask to an RGB page."""
        return apply_erase_mask(image, mask, fill_rgb=self.fill_rgb)

    def process(
        self,
        images: RGBImage | str | Path | Sequence[RGBImage | str | Path],
        *,
        source: str | Path | None = None,
        source_language: str = "ko",
        target_language: str | None = None,
    ) -> PageResult | list[PageResult]:
        """Run the full pipeline on one image or a list of images/paths.

        A single array or path returns one ``PageResult``. A list returns one
        ``PageResult`` per item, in the same order.
        """
        if _is_batch(images):
            if source is not None:
                raise ValueError("source cannot be used with batch input")
            return [
                self._process_one(
                    item,
                    source_language=source_language,
                    target_language=target_language,
                )
                for item in images
            ]
        return self._process_one(
            images, source=source, source_language=source_language, target_language=target_language
        )

    def _process_one(
        self,
        image: RGBImage | str | Path,
        *,
        source: str | Path | None = None,
        source_language: str = "ko",
        target_language: str | None = None,
    ) -> PageResult:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image = load_rgb(image_path)
            source = source if source is not None else image_path

        original = validate_rgb_image(image).copy()
        detections = self.detect_bubbles(original)
        cleaned, mask = self.erase_bubbles(original, detections)

        recognitions = []
        for detection in detections:
            if self.recognizer is None:
                recognitions.append(None)
                continue
            rows, columns = detection.box.as_slices()
            crop = original[rows, columns].copy()
            recognition = self.recognizer.recognize(crop, language=source_language)
            recognitions.append(validate_recognition(recognition))

        translations: list[str | None] = [None] * len(detections)
        if (
            self.translator is not None
            and self.recognizer is not None
            and target_language is not None
        ):
            texts = [item.text if item is not None else "" for item in recognitions]
            translated = self.translator.translate(
                texts,
                source_language=source_language,
                target_language=target_language,
            )
            translations = validate_translations(translated, len(texts))

        bubbles = [
            BubbleResult(
                index=index,
                detection=detection,
                recognition=recognitions[index - 1],
                translation=translations[index - 1],
            )
            for index, detection in enumerate(detections, start=1)
        ]
        source_path = Path(source).expanduser().resolve() if source is not None else None
        return PageResult(
            original=original,
            cleaned=cleaned,
            mask=mask,
            overlay=mask_to_rgba(mask, fill_rgb=self.fill_rgb),
            bubbles=bubbles,
            source=source_path,
            pipeline={
                "detector": _component_details(self.detector),
                "masker": _component_details(self.masker),
                "recognizer": _component_details(self.recognizer),
                "translator": _component_details(self.translator),
                "tile_height": self.tile_height,
                "tile_overlap": self.tile_overlap,
                "duplicate_iou": self.duplicate_iou,
                "detection_expansion_ratio": self.detection_expansion_ratio,
                "fill_rgb": list(self.fill_rgb),
            },
        )

    def process_file(
        self,
        path: str | Path,
        *,
        source_language: str = "ko",
        target_language: str | None = None,
    ) -> PageResult:
        return self._process_one(
            path, source_language=source_language, target_language=target_language
        )


def _expand_detection(detection: Detection, ratio: float, width: int, height: int) -> Detection:
    box = detection.box
    pad_x = round(ratio * box.width)
    pad_y = round(ratio * box.height)
    if not pad_x and not pad_y:
        return detection
    expanded = box.expand(pad_x, pad_y, width, height)
    return Detection(expanded, detection.score, detection.label, detection.tile_id)


def _is_batch(images: object) -> bool:
    if isinstance(images, (np.ndarray, str, bytes, Path)):
        return False
    return isinstance(images, Sequence)


def _component_details(component: object | None) -> dict[str, object] | None:
    if component is None:
        return None
    details: dict[str, object] = {
        "adapter": f"{type(component).__module__}.{type(component).__qualname__}"
    }
    metadata = getattr(component, "metadata", None)
    if isinstance(metadata, ModelMetadata):
        details["model"] = metadata.to_dict()
    return details
