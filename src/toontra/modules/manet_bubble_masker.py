"""Optional MA-Net bubble masking for a separately distributed checkpoint.

MA-Net weights are distributed separately; the checkpoint path is always
supplied by the caller and no automatic download occurs. See
docs/model_choices.md for how the public checkpoint was trained and
docs/custom_models.md for the exact preprocessing this adapter reproduces.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from toontra.contracts import validate_rgb_image
from toontra.errors import ModelContractError, OptionalDependencyError
from toontra.models import GrayMask, ModelMetadata, RGBImage

_ENCODER_NAME = "resnet34"
_INPUT_SIZE = 512
_LETTERBOX_RGB = (255, 255, 255)
_IMAGE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
_CUDA_PATTERN = re.compile(r"cuda(?::(?P<index>\d+))?\Z")

_EXPECTED_METADATA: dict[str, object] = {
    "architecture": "MAnet",
    "encoder": _ENCODER_NAME,
    "in_channels": 3,
    "classes": 1,
    "image_size": _INPUT_SIZE,
}

_PUBLIC_CHECKPOINT_SHA256 = (
    "6350601676c9e2b5448cf4cf109fb459a1f805d4101cfe34d4db47661f28df21"
)
_PUBLIC_CHECKPOINT_SOURCE_URL = (
    "https://huggingface.co/toontra-research/"
    "toontra-manet-bubble-segmentation/tree/"
    "f2f1c4eb1b2f7c492146da82d51fa474f54560cc"
)
_PUBLIC_CHECKPOINT_LICENSE = "Apache-2.0"


def _checkpoint_provenance(sha256: str) -> tuple[str | None, str | None]:
    if sha256.lower() == _PUBLIC_CHECKPOINT_SHA256:
        return _PUBLIC_CHECKPOINT_SOURCE_URL, _PUBLIC_CHECKPOINT_LICENSE
    return None, None

class ManetBubbleMasker:
    """Segment a bubble crop with the MA-Net/ResNet34 checkpoint.

    The crop is expected to already include whatever expansion the caller's
    pipeline applies to a detection box (Toontra applies its own before
    calling any masker -- see ``DETECTION_EXPANSION_RATIO`` in
    ``pipeline.py``). This adapter does not expand the crop again.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        threshold: float | None = None,
    ) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MA-Net checkpoint does not exist: {path}")
        if threshold is not None:
            threshold = float(threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("threshold must be between 0 and 1")

        torch, smp = _import_optional_dependencies()
        resolved_device = _resolve_device(torch, device)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = _validate_checkpoint(payload)

        model = smp.MAnet(
            encoder_name=_ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ModelContractError(
                "MA-Net checkpoint state_dict does not match the ResNet34 MA-Net architecture"
            ) from error
        model.eval()

        self.checkpoint_path = path
        self.device = resolved_device
        self.threshold = threshold if threshold is not None else float(payload["threshold"])
        checkpoint_sha256 = _checkpoint_sha256(path)
        source_url, license_spdx = _checkpoint_provenance(checkpoint_sha256)

        self.metadata = ModelMetadata(
            name="Toontra MA-Net ResNet34 speech-bubble masker",
            version=f"{payload['training_dataset']} v{payload['training_dataset_version']}",
            source_url=source_url,
            license_spdx=license_spdx,
            sha256=checkpoint_sha256,
        )

        self._model = model.to(resolved_device)
        self._torch = torch

    def create_mask(self, bubble_crop: RGBImage) -> GrayMask:
        """Return a binary mask matching the supplied RGB crop."""

        crop = validate_rgb_image(bubble_crop, name="bubble crop")
        height, width = crop.shape[:2]
        batch, letterbox = _preprocess(crop)

        tensor = self._torch.from_numpy(batch).to(self.device)
        with self._torch.inference_mode():
            logits = self._model(tensor)
        probabilities = self._torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        binary = np.where(probabilities >= self.threshold, 255, 0).astype(np.uint8)
        unpadded = _remove_letterbox(binary, letterbox)
        restored = cv2.resize(unpadded, (width, height), interpolation=cv2.INTER_NEAREST)
        return np.ascontiguousarray(restored, dtype=np.uint8)


def _preprocess(crop: RGBImage) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Letterbox ``crop`` into a normalized ``[1, 3, 512, 512]`` batch.

    Aspect ratio is preserved by scaling the crop to fit within
    ``_INPUT_SIZE`` and centering it on a white canvas, matching the training
    preprocessing recorded in docs/custom_models.md.
    """

    height, width = crop.shape[:2]
    scale = min(_INPUT_SIZE / width, _INPUT_SIZE / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_width, new_height), interpolation=interpolation)

    canvas = np.full((_INPUT_SIZE, _INPUT_SIZE, 3), _LETTERBOX_RGB, dtype=np.uint8)
    x0 = (_INPUT_SIZE - new_width) // 2
    y0 = (_INPUT_SIZE - new_height) // 2
    canvas[y0 : y0 + new_height, x0 : x0 + new_width] = resized

    normalized = canvas.astype(np.float32) / 255.0
    normalized = (normalized - _IMAGE_MEAN) / _IMAGE_STD
    chw = normalized.transpose(2, 0, 1)
    batch = np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)
    return batch, (x0, y0, new_width, new_height)


def _remove_letterbox(mask: np.ndarray, letterbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, content_width, content_height = letterbox
    return mask[y0 : y0 + content_height, x0 : x0 + content_width]


def _validate_checkpoint(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ModelContractError("MA-Net checkpoint must contain a dict payload")
    for key, expected in _EXPECTED_METADATA.items():
        if payload.get(key) != expected:
            raise ModelContractError(
                f"MA-Net checkpoint metadata {key!r} must be {expected!r}, "
                f"got {payload.get(key)!r}"
            )

    normalization = payload.get("normalization")
    if not isinstance(normalization, dict):
        raise ModelContractError(
            "MA-Net checkpoint normalization must contain mean and std values"
        )

    for key, expected in (("mean", _IMAGE_MEAN), ("std", _IMAGE_STD)):
        try:
            actual = np.asarray(normalization.get(key), dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ModelContractError(
                f"MA-Net checkpoint normalization {key!r} is invalid"
            ) from error

        if actual.shape != expected.shape or not np.allclose(
            actual,
            expected,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ModelContractError(
                f"MA-Net checkpoint normalization {key!r} does not match "
                "the adapter preprocessing"
            )

    threshold = payload.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ModelContractError("MA-Net checkpoint threshold must be a number")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ModelContractError("MA-Net checkpoint threshold must be between 0 and 1")

    if not isinstance(payload.get("training_dataset"), str):
        raise ModelContractError("MA-Net checkpoint training_dataset must be a string")
    if not isinstance(payload.get("training_dataset_version"), int):
        raise ModelContractError("MA-Net checkpoint training_dataset_version must be an integer")

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ModelContractError("MA-Net checkpoint state_dict must be a non-empty mapping")
    if not all(isinstance(key, str) for key in state_dict):
        raise ModelContractError("MA-Net state_dict keys must be strings")
    return state_dict


def _import_optional_dependencies() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise OptionalDependencyError(
            "ManetBubbleMasker requires the 'manet' optional dependencies; "
            'install Toontra with `pip install -e ".[manet]"`'
        ) from error
    try:
        smp = importlib.import_module("segmentation_models_pytorch")
    except ImportError as error:
        raise OptionalDependencyError(
            "ManetBubbleMasker requires segmentation-models-pytorch; "
            'install Toontra with `pip install -e ".[manet]"`'
        ) from error
    return torch, smp


def _resolve_device(torch: Any, requested: str) -> Any:
    if not isinstance(requested, str):
        raise TypeError("device must be a string")
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return torch.device("cpu")

    match = _CUDA_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError("device must be 'cpu', 'auto', 'cuda', or 'cuda:N'")
    if not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {requested}")
    index = int(match.group("index") or 0)
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {index} is unavailable; found {torch.cuda.device_count()} device(s)"
        )
    return torch.device(f"cuda:{index}")


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
