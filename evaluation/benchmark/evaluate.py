from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toontra.geometry import (
    cross_tile_non_max_suppression,
    iou,
    non_max_suppression,
)
from toontra.models import Box, Detection, Tile
from toontra.tiling import detection_ownership_flags

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class GroundTruth:
    annotation_id: int
    box: tuple[float, float, float, float]


def float_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(
        0.0, min(first[2], second[2]) - max(first[0], second[0])
    )
    intersection_height = max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    intersection = intersection_width * intersection_height
    if intersection == 0.0:
        return 0.0

    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def load_ground_truth() -> tuple[
    dict[str, tuple[int, int]],
    dict[str, list[GroundTruth]],
]:
    coco = json.loads((ROOT / "annotations.json").read_text(encoding="utf-8"))

    numeric_to_stable: dict[int, str] = {}
    dimensions: dict[str, tuple[int, int]] = {}

    for image in coco["images"]:
        stable_id = Path(image["file_name"]).stem
        numeric_to_stable[int(image["id"])] = stable_id
        dimensions[stable_id] = (int(image["width"]), int(image["height"]))

    ground_truth: dict[str, list[GroundTruth]] = defaultdict(list)

    for annotation in coco["annotations"]:
        stable_id = numeric_to_stable[int(annotation["image_id"])]
        x, y, width, height = (float(value) for value in annotation["bbox"])

        ground_truth[stable_id].append(
            GroundTruth(
                annotation_id=int(annotation["id"]),
                box=(x, y, x + width, y + height),
            )
        )

    return dimensions, ground_truth


def load_raw_candidates(
    detector_name: str,
    stable_id: str,
    *,
    expected_width: int,
    expected_height: int,
    tile_height: int,
    overlap: int,
) -> tuple[list[Detection], list[list[int]]]:
    path = ROOT / "predictions" / detector_name / f"{stable_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    expected_shape = [expected_height, expected_width, 3]
    if payload["image_shape"] != expected_shape:
        raise RuntimeError(
            f"{detector_name}/{stable_id}: image shape mismatch"
        )

    if int(payload["tile_height"]) != tile_height:
        raise RuntimeError(
            f"{detector_name}/{stable_id}: tile height mismatch"
        )

    if int(payload["overlap"]) != overlap:
        raise RuntimeError(
            f"{detector_name}/{stable_id}: overlap mismatch"
        )

    detections = [
        Detection(
            box=Box(*(int(value) for value in item["box"])),
            score=float(item["score"]),
            label=str(item["label"]),
            tile_id=int(item["tile_id"]),
        )
        for item in payload["raw_candidates"]
    ]

    spans = [
        [int(start), int(end)]
        for start, end in payload["tile_spans"]
    ]

    return detections, spans


def ownership_flags(
    detections: list[Detection],
    tile_spans: list[list[int]],
    *,
    page_width: int,
    page_height: int,
) -> list[bool]:
    if not tile_spans:
        raise RuntimeError("tile_spans cannot be empty")

    backing = np.empty((1, page_width, 3), dtype=np.uint8)
    tiles = [
        Tile(
            index=index,
            image=np.broadcast_to(
                backing,
                (end - start, page_width, 3),
            ),
            x0=0,
            y0=start,
        )
        for index, (start, end) in enumerate(tile_spans)
    ]

    try:
        return detection_ownership_flags(
            tiles,
            detections,
            page_width=page_width,
            page_height=page_height,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def same_tile_non_max_suppression(
    detections: list[Detection],
    *,
    iou_threshold: float,
) -> list[Detection]:
    indexed = list(enumerate(detections))
    indexed.sort(
        key=lambda item: (
            -item[1].score,
            -item[1].box.area,
            item[0],
        )
    )

    kept: list[Detection] = []

    for _, candidate in indexed:
        suppress = False

        for selected in kept:
            if candidate.tile_id is None or selected.tile_id is None:
                continue
            if candidate.tile_id != selected.tile_id:
                continue
            if candidate.label != selected.label:
                continue

            if iou(candidate.box, selected.box) >= iou_threshold:
                suppress = True
                break

        if not suppress:
            kept.append(candidate)

    kept.sort(key=lambda item: (item.box.y1, item.box.x1, -item.score))
    return kept


def expand_predictions(
    detections: list[Detection],
    *,
    ratio: float,
    width: int,
    height: int,
) -> list[Detection]:
    expanded: list[Detection] = []

    for detection in detections:
        box = detection.box
        pad_x = round(ratio * box.width)
        pad_y = round(ratio * box.height)

        if pad_x == 0 and pad_y == 0:
            expanded.append(detection)
            continue

        expanded_box = box.expand(pad_x, pad_y, width, height)

        expanded.append(
            Detection(
                box=expanded_box,
                score=detection.score,
                label=detection.label,
                tile_id=detection.tile_id,
            )
        )

    return expanded


def match_predictions(
    predictions: list[Detection],
    ground_truth: list[GroundTruth],
    *,
    match_iou: float,
) -> dict[str, Any]:
    candidate_pairs: list[tuple[float, int, int, int]] = []

    for prediction_index, prediction in enumerate(predictions):
        predicted_box = tuple(float(value) for value in prediction.box.as_tuple())

        for gt_index, gt in enumerate(ground_truth):
            overlap = float_iou(predicted_box, gt.box)

            if overlap >= match_iou:
                candidate_pairs.append(
                    (
                        -overlap,
                        prediction_index,
                        gt.annotation_id,
                        gt_index,
                    )
                )

    candidate_pairs.sort()

    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()

    for _, prediction_index, _, gt_index in candidate_pairs:
        if prediction_index in used_predictions:
            continue
        if gt_index in used_ground_truth:
            continue

        used_predictions.add(prediction_index)
        used_ground_truth.add(gt_index)

    tp = len(used_ground_truth)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate(
    per_image: dict[str, dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    raw = sum(item["raw"] for item in per_image.values())
    final = sum(item[mode]["final"] for item in per_image.values())
    gt = sum(item["gt"] for item in per_image.values())
    tp = sum(item[mode]["tp"] for item in per_image.values())
    fp = sum(item[mode]["fp"] for item in per_image.values())
    fn = sum(item[mode]["fn"] for item in per_image.values())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "gt": gt,
        "raw": raw,
        "final": final,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_detector(
    detector_name: str,
    metadata: dict[str, Any],
    dimensions: dict[str, tuple[int, int]],
    ground_truth: dict[str, list[GroundTruth]],
) -> dict[str, Any]:
    config = metadata["benchmark"]

    tile_height = int(config["tile_height"])
    overlap = int(config["overlap"])
    cross_tile_iou = float(config["cross_tile_iou"])
    containment = float(config["containment_threshold"])
    same_tile_iou = float(config["same_tile_nms_iou"])
    expansion_ratio = float(config["prediction_expansion_ratio"])
    match_iou = float(config["match_iou"])

    results: dict[str, dict[str, Any]] = {}

    for stable_id in sorted(dimensions):
        width, height = dimensions[stable_id]

        raw, tile_spans = load_raw_candidates(
            detector_name,
            stable_id,
            expected_width=width,
            expected_height=height,
            tile_height=tile_height,
            overlap=overlap,
        )

        global_nms = non_max_suppression(
            raw,
            iou_threshold=cross_tile_iou,
            containment_threshold=containment,
            class_aware=True,
        )

        owned = ownership_flags(
            raw,
            tile_spans,
            page_width=width,
            page_height=height,
        )

        ownership = cross_tile_non_max_suppression(
            raw,
            owned=owned,
            iou_threshold=cross_tile_iou,
            containment_threshold=containment,
            class_aware=True,
        )

        ownership_same_tile = same_tile_non_max_suppression(
            ownership,
            iou_threshold=same_tile_iou,
        )

        modes = {
            "nms_only": global_nms,
            "runtime": ownership,
            "ownership_plus_same_tile_nms": ownership_same_tile,
        }

        image_result: dict[str, Any] = {
            "gt": len(ground_truth[stable_id]),
            "raw": len(raw),
        }

        for mode_name, detections in modes.items():
            expanded = expand_predictions(
                detections,
                ratio=expansion_ratio,
                width=width,
                height=height,
            )

            metrics = match_predictions(
                expanded,
                ground_truth[stable_id],
                match_iou=match_iou,
            )
            metrics["final"] = len(expanded)
            image_result[mode_name] = metrics

        results[stable_id] = image_result

    expected_raw = int(metadata[detector_name]["raw_candidate_count"])
    actual_raw = sum(item["raw"] for item in results.values())

    if actual_raw != expected_raw:
        raise RuntimeError(
            f"{detector_name}: expected {expected_raw} raw candidates, "
            f"found {actual_raw}"
        )

    return {
        "images": results,
        "totals": {
            "nms_only": aggregate(results, "nms_only"),
            "runtime": aggregate(results, "runtime"),
            "ownership_plus_same_tile_nms": aggregate(
                results,
                "ownership_plus_same_tile_nms",
            ),
        },
    }


def main() -> None:
    metadata = json.loads(
        (ROOT / "predictions" / "metadata.json").read_text(encoding="utf-8")
    )

    dimensions, ground_truth = load_ground_truth()

    total_ground_truth = sum(len(items) for items in ground_truth.values())
    expected_ground_truth = int(
        metadata["benchmark"]["ground_truth_annotations"]
    )

    if total_ground_truth != expected_ground_truth:
        raise RuntimeError(
            f"Expected {expected_ground_truth} ground-truth annotations, "
            f"found {total_ground_truth}"
        )

    results = {
        "configuration": metadata["benchmark"],
        "yolo26": evaluate_detector(
            "yolo26",
            metadata,
            dimensions,
            ground_truth,
        ),
        "yolov7": evaluate_detector(
            "yolov7",
            metadata,
            dimensions,
            ground_truth,
        ),
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()