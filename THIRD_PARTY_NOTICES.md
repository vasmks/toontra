# Third-party notices

The TOONTRA public reconstruction is licensed under
AGPL-3.0-only. The licenses listed below apply to the corresponding
third-party dependencies, datasets, and model artifacts.

Toontra does not redistribute training datasets. It bundles one trained
checkpoint: `src/toontra/weights/speech_bubble_yolo26s.pt`.

## Required dependencies

### NumPy

- Project: <https://numpy.org/>
- Source: <https://github.com/numpy/numpy>
- License: BSD-3-Clause

### OpenCV

- Project: <https://opencv.org/>
- Source: <https://github.com/opencv/opencv>
- License: Apache-2.0 for OpenCV 4.5 and later

### Ultralytics and the bundled YOLO26 checkpoint

- Source: <https://github.com/ultralytics/ultralytics>
- License: Ultralytics YOLO code and models are available under AGPL-3.0,
  with commercial licensing available separately.
- Base model: Ultralytics YOLO26s, initialized from the official pretrained
  `yolo26s.pt` checkpoint.
- Training dataset:
  [Roboflow 100-VL `speech-bubbles-detection-r22zt-ou0u6-jols`](https://universe.roboflow.com/rf100-vl/speech-bubbles-detection-r22zt-ou0u6-jols)
- Dataset workspace: `rf100-vl`
- Dataset project: `speech-bubbles-detection-r22zt-ou0u6-jols`
- Dataset version: 1
- Dataset license: MIT
- Export format: YOLOv8
- Source classes: `Elipse`, `cloude`, `other`, `rectangle`, `sea_uchirin`,
  and `thorn`. These were merged into Toontra's single `speech_bubble`
  class for training.
- Bundled checkpoint:
  `src/toontra/weights/speech_bubble_yolo26s.pt`
- SHA-256: `6E2EBF2820CC0D0A55D1737C126AAC0B73B427E079566B844A4CD156F1804BAC`
- The checkpoint was trained specifically for this public reconstruction
  and is not an original TOONTRA production checkpoint.

See [docs/model_choices.md](docs/model_choices.md) for training and evaluation
details.

## Optional adapters

### EasyOCR

- Source: <https://github.com/JaidedAI/EasyOCR>
- License: Apache-2.0
- Behavior: EasyOCR may download selected language weights, but Toontra
  permits that only after the user explicitly opts in. No download occurs
  during import or automated tests.

### PyTorch, Segmentation Models PyTorch, and the optional MA-Net checkpoint

- PyTorch source: <https://github.com/pytorch/pytorch>
- PyTorch license: BSD-3-Clause; binary distributions may include components
  covered by additional notices.
- Segmentation Models PyTorch source:
  <https://github.com/qubvel-org/segmentation_models.pytorch>
- Segmentation Models PyTorch license: MIT
- Behavior: `segmentation-models-pytorch` is installed through the optional
  `manet` extra (`pip install -e ".[manet]"`). PyTorch is also listed in that
  extra for MA-Net support, but may already be present through other
  dependencies. `ManetBubbleMasker` performs no model download; the checkpoint
  path is always supplied explicitly.
- Optional checkpoint: the public MA-Net checkpoint is hosted as
  `manet_bubble_segmentation.pth` at
  https://huggingface.co/toontra-research/toontra-manet-bubble-segmentation/tree/f2f1c4eb1b2f7c492146da82d51fa474f54560cc
  and has SHA256
  `6350601676c9e2b5448cf4cf109fb459a1f805d4101cfe34d4db47661f28df21`.
  The checkpoint is licensed under Apache-2.0 and was trained for this public
  reconstruction using MA-Net with a ResNet34 encoder on Roboflow
  `manga-segment_v2`, version 5. It is distributed separately and is excluded
  from Git and package data. See
  [docs/model_choices.md](docs/model_choices.md) for evaluation numbers and
  [docs/training/](docs/training/) for the training record.

Applicable upstream dependency, model, and dataset terms should be reviewed
separately.

## AI-generated example image

`src/toontra/assets/sample_webtoon.png` was generated with ChatGPT as synthetic
demo artwork. It was not used for training or evaluation.

## Independent benchmark annotations

The speech-bubble annotations under `evaluation/benchmark/` were created for
this project and are provided without the underlying source artwork.

The original webtoon images are not redistributed by this repository and are
not covered by the TOONTRA repository license. Rights in the source artwork
remain with their respective owners. Official source pages and additional
benchmark documentation are listed in
[evaluation/benchmark/README.md](evaluation/benchmark/README.md).

## User-provided components

Custom models are supplied by the user through Toontra's component
interfaces. Toontra does not verify their source, license, training-data
provenance, or file integrity. Pickled or otherwise executable checkpoints
can run code when loaded and should come from a trusted source.
