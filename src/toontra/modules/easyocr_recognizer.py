"""Optional EasyOCR adapter with lazy, download-free initialization."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from math import isfinite
from numbers import Real
from typing import Any

import cv2
import numpy as np

from toontra.contracts import validate_rgb_image
from toontra.errors import ModelContractError, OptionalDependencyError
from toontra.models import ModelMetadata, Recognition, RGBImage

ReaderFactory = Callable[[Sequence[str]], Any]


def _language_code(language: str) -> str:
    if not isinstance(language, str) or not language.strip():
        raise ModelContractError("language must be a non-empty string")
    normalized = language.strip().lower().replace("_", "-")
    aliases = {
        "kor": "ko",
        "korean": "ko",
        "eng": "en",
        "english": "en",
    }
    return aliases.get(normalized, normalized)


class EasyOcrRecognizer:
    """Recognize bubble text with an optional EasyOCR installation.

    Supplying ``reader`` or ``reader_factory`` is useful for tests and custom
    model hosting. Without either, EasyOCR is imported only on first use.
    Downloads stay disabled unless ``download_enabled=True`` is explicit.
    """

    metadata = ModelMetadata(
        name="EasyOCR",
        version="1.7.2",
        source_url="https://github.com/JaidedAI/EasyOCR",
        license_spdx="Apache-2.0",
    )

    def __init__(
        self,
        *,
        reader: Any | None = None,
        reader_factory: ReaderFactory | None = None,
        gpu: bool = False,
        model_storage_directory: str | None = None,
        download_enabled: bool = False,
    ) -> None:
        if reader is not None and reader_factory is not None:
            raise ValueError("provide reader or reader_factory, not both")
        if reader is not None and not callable(getattr(reader, "readtext", None)):
            raise TypeError("reader must provide a callable readtext method")
        if reader_factory is not None and not callable(reader_factory):
            raise TypeError("reader_factory must be callable")
        self._injected_reader = reader
        self._reader_factory = reader_factory
        self._readers: dict[tuple[str, ...], Any] = {}
        self.gpu = gpu
        self.model_storage_directory = model_storage_directory
        self.download_enabled = download_enabled

    def recognize(self, bubble_crop: RGBImage, *, language: str = "ko") -> Recognition:
        crop = validate_rgb_image(bubble_crop, name="bubble crop")
        code = _language_code(language)
        languages = (code,) if code == "en" else (code, "en")
        reader = self._reader_for(languages)

        # EasyOCR documents NumPy inputs as OpenCV images, which use BGR.  RGB
        # remains the public Toontra contract and the conversion stays here.
        bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        raw_output = reader.readtext(bgr_crop, detail=1, paragraph=False)
        return self._to_recognition(raw_output)

    def _reader_for(self, languages: tuple[str, ...]) -> Any:
        if self._injected_reader is not None:
            return self._injected_reader
        if languages in self._readers:
            return self._readers[languages]

        if self._reader_factory is not None:
            reader = self._reader_factory(list(languages))
        else:
            try:
                easyocr = importlib.import_module("easyocr")
            except ImportError as exc:
                raise OptionalDependencyError(
                    "EasyOCR is optional; install Toontra with `pip install toontra[ocr]`"
                ) from exc

            options: dict[str, Any] = {
                "gpu": self.gpu,
                "download_enabled": self.download_enabled,
            }
            if self.model_storage_directory is not None:
                options["model_storage_directory"] = self.model_storage_directory
            try:
                reader = easyocr.Reader(list(languages), **options)
            except Exception as exc:  # EasyOCR uses several exception types for absent weights.
                download_hint = (
                    "Check the EasyOCR cache and download permissions."
                    if self.download_enabled
                    else "Install the weights manually or explicitly allow the download."
                )
                raise OptionalDependencyError(
                    "EasyOCR could not load its model files. " + download_hint
                ) from exc

        if not callable(getattr(reader, "readtext", None)):
            raise ModelContractError("reader_factory must return an object with readtext()")
        self._readers[languages] = reader
        return reader

    @staticmethod
    def _to_recognition(raw_output: Any) -> Recognition:
        if isinstance(raw_output, (str, bytes, np.ndarray)) or not isinstance(
            raw_output, Sequence
        ):
            raise ModelContractError("EasyOCR output must be a sequence")

        texts: list[str] = []
        confidences: list[float] = []
        for index, item in enumerate(raw_output):
            malformed = (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) < 3
            )
            if malformed:
                raise ModelContractError(
                    f"EasyOCR result {index} must contain box, text, confidence"
                )
            text = item[1]
            confidence = item[2]
            if not isinstance(text, str):
                raise ModelContractError(f"EasyOCR result {index} text must be a string")
            if isinstance(confidence, bool) or not isinstance(confidence, Real):
                raise ModelContractError(f"EasyOCR result {index} confidence must be numeric")
            numeric_confidence = float(confidence)
            if not isfinite(numeric_confidence) or not 0 <= numeric_confidence <= 1:
                raise ModelContractError(
                    f"EasyOCR result {index} confidence must be between 0 and 1"
                )
            cleaned = text.strip()
            if cleaned:
                texts.append(cleaned)
                confidences.append(numeric_confidence)

        if not texts:
            return Recognition(text="", confidence=0.0)
        return Recognition(
            text="\n".join(texts),
            confidence=float(sum(confidences) / len(confidences)),
        )


class NullTextRecognizer:
    """A predictable recognizer for extraction-only pipelines."""

    def recognize(self, bubble_crop: RGBImage, *, language: str = "ko") -> Recognition:
        validate_rgb_image(bubble_crop, name="bubble crop")
        _language_code(language)
        return Recognition(text="", confidence=0.0)
