# Model choices

This document records the models used by the public reconstruction, where
they came from, and how they are loaded.

## Bundled: YOLO26 bubble detector

`Yolo26BubbleDetector` (`src/toontra/modules/yolo26_bubble_detector.py`) is
the default and only bundled `BubbleDetector` implementation. Its checkpoint,
`src/toontra/weights/speech_bubble_yolo26s.pt`, was trained for this public
reconstruction with Ultralytics, starting from the official pretrained
`yolo26s.pt` checkpoint.

Training used version 1 of the Roboflow 100-VL
[`speech-bubbles-detection-r22zt-ou0u6-jols`](https://universe.roboflow.com/rf100-vl/speech-bubbles-detection-r22zt-ou0u6-jols)
dataset from workspace `rf100-vl`. The dataset was downloaded through the
Roboflow Python SDK in YOLOv8 export format. This refers to the annotation
format only; the detector trained for this reconstruction is YOLO26s.

The source dataset contains six bubble-shape classes: `Elipse`, `cloude`,
`other`, `rectangle`, `sea_uchirin`, and `thorn`. These were merged into a
single `speech_bubble` class for training. The dataset is published under the
MIT license. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for
dataset, Ultralytics, and checkpoint licensing information.

The bundled checkpoint is:

- File: `src/toontra/weights/speech_bubble_yolo26s.pt`
- Base checkpoint: `yolo26s.pt`
- Training dataset version: 1
- Training image size: 800
- SHA-256: `6E2EBF2820CC0D0A55D1737C126AAC0B73B427E079566B844A4CD156F1804BAC`

`Toontra` processes tall pages with tiled inference, followed by tile
ownership and cross-tile NMS. Each surviving detection is then expanded once
by 5% of its width on the left and right and 5% of its height on the top and
bottom, giving roughly 10% growth in each dimension. Expansion happens after
deduplication, so duplicate comparisons use the original detection boxes.
Downstream components, including `ManetBubbleMasker`, do not expand the box
again. See [custom_models.md](custom_models.md#optional-ma-net-masker).

Training-run results from the retained Kaggle artifacts and results from the
independent manually annotated webtoon benchmark are stored under
[evaluation/](../evaluation/). The README reports them separately because
they measure different datasets and evaluation settings.

The repository retains the training configuration, results, and the best
checkpoint from that run. It does not include the Kaggle notebook or a copy of
the training dataset, so these artifacts document the run rather than provide
an exact retraining environment.

## Bubble masking

`WhiteBubbleMasker` is a deterministic, weight-free OpenCV baseline. It looks
for a light bubble interior with a closed outline and fills it, including text
holes inside the region. It needs no model download, GPU, or account.

## Optional: MA-Net bubble masking

`ManetBubbleMasker` (`src/toontra/modules/manet_bubble_masker.py`) provides
MA-Net-based speech-bubble segmentation as an alternative to `WhiteBubbleMasker`.
Its separately distributed checkpoint was trained for this public
reconstruction using Segmentation Models PyTorch's `MAnet` with a `resnet34`
encoder, fine-tuned on the public Roboflow
[`manga-segment_v2`](https://universe.roboflow.com/ashu-biqfs/manga-segment_v2),
version 5 speech-bubble segmentation dataset. It is a new checkpoint produced
by this training run, not the original production checkpoint.

The public checkpoint is hosted as `manet_bubble_segmentation.pth` at
[toontra-research/toontra-manet-bubble-segmentation](https://huggingface.co/toontra-research/toontra-manet-bubble-segmentation/tree/f2f1c4eb1b2f7c492146da82d51fa474f54560cc).
It is licensed under Apache-2.0 and has SHA256
`6350601676c9e2b5448cf4cf109fb459a1f805d4101cfe34d4db47661f28df21`.

Evaluation numbers from
[docs/training/manet_results.json](training/manet_results.json) at the best
checkpoint epoch:

| Metric | Value |
| --- | --- |
| Validation Dice | 0.9828 |
| Test Dice | 0.9805 |
| Test IoU | 0.9622 |
| Decision threshold | 0.45 (stored in the checkpoint) |

Full per-epoch training history is in
[docs/training/manet_training_history.csv](training/manet_training_history.csv).
These results apply to the recorded training run and dataset split.

MA-Net support is installed with `pip install -e ".[manet]"`, which adds
`segmentation-models-pytorch` and explicitly includes PyTorch for this adapter.
PyTorch may already be present through other dependencies. The checkpoint is
distributed separately and is not stored in Git or included in package data.

`ManetBubbleMasker` requires an explicit checkpoint path and does not download
weights automatically. See
[custom_models.md](custom_models.md#optional-ma-net-masker) for preprocessing
and construction details.

## Other optional adapters

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) is available as an optional
  OCR adapter. Its dependency and model cache stay outside the core
  installation, and downloads require explicit consent.
- `CallableBubbleDetector` and the other component protocols accept a
  private, public, Torch, Paddle, or ONNX implementation without leaking
  framework types into the pipeline.

## Stages with no public model

The original pipeline also included text detection, inpainting, font matching,
text reinsertion, and image enhancement. These stages are not implemented in
this public reconstruction.
