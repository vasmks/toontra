# Custom model contract

Toontra keeps model-specific tensor work inside adapters. The public pipeline
uses RGB NumPy images and a small set of data classes.

## Image input

- Type: `numpy.ndarray`
- Shape: `[height, width, 3]`
- Dtype: `uint8`
- Channel order: RGB

## Bubble detector output

`BubbleDetector.detect(image)` returns a sequence of `Detection` objects.

- Coordinates use `Box(x1, y1, x2, y2)`.
- Coordinates are pixel values relative to the supplied image.
- `x2` and `y2` are exclusive, matching NumPy slicing.
- `score` must be between 0 and 1.
- Boxes must have positive width and height.

The pipeline clips detections at page boundaries and rejects invalid output with
`ModelContractError`. A custom adapter should still perform its own model resize,
letterboxing, and inverse coordinate transform.

## Bubble masker output

`BubbleMasker.create_mask(bubble_crop)` returns a two-dimensional `uint8` mask
with exactly the same height and width as the crop it receives. The crop is
the reported `Detection.box` region, unmodified: after ownership and NMS,
Toontra expands a box exactly once in `detect_bubbles()` (see
`DETECTION_EXPANSION_RATIO` in `pipeline.py`). That
same expanded box is what gets reported, cropped for OCR, and passed to the
masker -- a masker never receives a tighter or looser crop than what
`Detection.box` records, and must not expand it again.

- `0` preserves the source pixel.
- `255` fully replaces it with the configured fill color.
- Values between 0 and 255 are supported as partial coverage.

Local masks are combined into the page mask using a pixel-wise maximum. The
completed mask is then applied inside detected regions, so overlaps cannot
restore pixels covered by another bubble.

## OCR and translation

`TextRecognizer.recognize(crop, language=...)` returns one `Recognition`.
`Translator.translate(texts, source_language=..., target_language=...)` must
return the same number of strings in the same order.

See `examples/custom_components.py` for small working adapters.

## Replacing the bundled YOLO26 detector

`Yolo26BubbleDetector` is used by default. A replacement implements the same
`BubbleDetector` protocol
(`detect(image) -> Sequence[Detection]`) and is passed to the constructor:

```python
from toontra import Toontra

toontra = Toontra(detector=MyBubbleDetector())
```

Downstream pipeline stages use the `Detection` objects returned by `detect()`,
rather than YOLO26- or Ultralytics-specific types. See `docs/model_choices.md`
for the bundled checkpoint provenance.

## Stages with no bundled model

The original pipeline also included text detection, inpainting, font matching,
text reinsertion, and image enhancement. These stages are not implemented in
this public version.

## Optional MA-Net masker

`ManetBubbleMasker` is a ready-made adapter for the optional MA-Net/ResNet34
speech-bubble checkpoint (Segmentation Models PyTorch `MAnet`, `resnet34`
encoder, one output class). The checkpoint is distributed separately and is
not stored in Git or included in package data. Install the optional runtime:

```console
pip install -e ".[manet]"
```

`ManetBubbleMasker` requires a checkpoint path and does not download the model automatically.

```python
from toontra import Toontra
from toontra.modules import ManetBubbleMasker

masker = ManetBubbleMasker(
    "weights/toontra_manet_resnet34_bubble_segmentation.pth",
    device="auto",
)
toontra = Toontra(masker=masker)
```

The adapter accepts one checkpoint format. It loads the checkpoint with
`torch.load(..., weights_only=True)`, verifies that the embedded metadata
(`architecture="MAnet"`, `encoder="resnet34"`, `in_channels=3`, `classes=1`,
`image_size=512`) matches the model configuration, and then loads the
`state_dict` strictly into
`smp.MAnet(encoder_name="resnet34", encoder_weights=None, in_channels=3,
classes=1, activation=None)`. A mismatched or malformed checkpoint raises
`ModelContractError` instead of being silently coerced.

`ManetBubbleMasker` expects an RGB crop and uses the same preprocessing as the
training pipeline. The crop is resized to fit within 512 × 512 while preserving
its aspect ratio, placed on a white canvas, and normalized with ImageNet mean
and standard deviation values.

After inference, the predicted mask is thresholded using the value stored in
the checkpoint. A different threshold can be supplied with `threshold=`. The
padding is then removed and the mask is resized back to the original crop size.

The checkpoint metadata records `crop_padding=0.15`, which describes the
padding used when preparing training crops. It is not applied by the masker at
runtime.

The crop passed to `ManetBubbleMasker` has already been expanded by the main
pipeline, so the masker does not apply any additional expansion.

For local testing with the real checkpoint, set `TOONTRA_MANET_CHECKPOINT`.
`TOONTRA_MANET_DEVICE` can also be set to choose the inference device.

Images loaded from files are converted to 8-bit RGB. Transparent PNGs are
flattened onto a white background before they enter the pipeline.

See [model_choices.md](model_choices.md) for checkpoint provenance and
evaluation results.
