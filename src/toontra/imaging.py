from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

from .contracts import validate_gray_mask, validate_rgb_image
from .errors import ImageValidationError, ModelContractError
from .models import GrayMask, RGBImage

PathLike: TypeAlias = str | Path


def load_rgb(path: PathLike) -> RGBImage:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    encoded = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ImageValidationError(f"OpenCV could not decode: {image_path}")
    if image.dtype != np.uint8:
        raise ImageValidationError(
            f"Only 8-bit images are supported, got {image.dtype} in {image_path}"
        )
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3:
        raise ImageValidationError(f"Unsupported image shape: {image.shape}")
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[2] == 4:
        # The public pipeline is RGB, so flatten transparent input against a
        # documented white page instead of silently turning it black.
        alpha = image[..., 3:4].astype(np.float32) / 255.0
        bgr = image[..., :3].astype(np.float32)
        flattened = bgr * alpha + 255.0 * (1.0 - alpha)
        return cv2.cvtColor(np.rint(flattened).astype(np.uint8), cv2.COLOR_BGR2RGB)
    raise ImageValidationError(f"Unsupported channel count: {image.shape[2]}")


def save_rgb(path: PathLike, image: RGBImage) -> Path:
    rgb = validate_rgb_image(image)
    return _encode_and_save(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_gray(path: PathLike, mask: GrayMask) -> Path:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.ndim != 2:
        raise ImageValidationError("Grayscale image must be a two-dimensional uint8 array")
    return _encode_and_save(path, mask)


def save_rgba(path: PathLike, image: np.ndarray) -> Path:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ImageValidationError("RGBA image must be a uint8 NumPy array")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ImageValidationError("RGBA image must have shape [height, width, 4]")
    return _encode_and_save(path, cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))


def _encode_and_save(path: PathLike, image: np.ndarray) -> Path:
    output_path = Path(path)
    suffix = output_path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Output extension must be .png, .jpg, .jpeg, or .webp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise OSError(f"OpenCV could not encode: {output_path}")
    encoded.tofile(output_path)
    return output_path


def apply_erase_mask(
    image: RGBImage,
    mask: GrayMask,
    *,
    fill_rgb: tuple[int, int, int] = (255, 255, 255),
) -> RGBImage:
    rgb = validate_rgb_image(image)
    try:
        coverage = validate_gray_mask(mask, rgb.shape[:2])
    except ModelContractError as error:
        raise ImageValidationError(str(error)) from error
    color = _validate_color(fill_rgb)
    alpha = coverage.astype(np.float32)[..., None] / 255.0
    fill = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    output = rgb.astype(np.float32) * (1.0 - alpha) + fill * alpha
    return np.rint(output).clip(0, 255).astype(np.uint8)


def mask_to_rgba(
    mask: GrayMask,
    *,
    fill_rgb: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.ndim != 2:
        raise ImageValidationError("mask must be a two-dimensional uint8 array")
    color = np.asarray(_validate_color(fill_rgb), dtype=np.uint8)
    overlay = np.empty((*mask.shape, 4), dtype=np.uint8)
    overlay[..., :3] = color
    overlay[..., 3] = mask
    return overlay


def alpha_composite(base: RGBImage, overlay: np.ndarray) -> RGBImage:
    rgb = validate_rgb_image(base, name="base")
    if not isinstance(overlay, np.ndarray) or overlay.dtype != np.uint8:
        raise ImageValidationError("overlay must be a uint8 NumPy array")
    if overlay.shape != (*rgb.shape[:2], 4):
        raise ImageValidationError(
            f"overlay must have shape {(*rgb.shape[:2], 4)}, got {overlay.shape}"
        )
    alpha = overlay[..., 3:4].astype(np.float32) / 255.0
    output = overlay[..., :3].astype(np.float32) * alpha + rgb.astype(np.float32) * (
        1.0 - alpha
    )
    return np.rint(output).clip(0, 255).astype(np.uint8)


def _validate_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(color) != 3:
        raise ValueError("fill_rgb must contain three channels")
    values = tuple(int(value) for value in color)
    if any(value < 0 or value > 255 for value in values):
        raise ValueError("fill_rgb values must be between 0 and 255")
    return values
