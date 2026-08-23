from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from toontra.contracts import validate_translations
from toontra.errors import ModelContractError, OptionalDependencyError
from toontra.models import Box, Detection
from toontra.modules import (
    CallableBubbleDetector,
    DictionaryTranslator,
    EasyOcrRecognizer,
    IdentityTranslator,
    NullTextRecognizer,
)


def make_image() -> np.ndarray:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[0, 0] = (255, 0, 0)
    return image


class FakeReader:
    def __init__(self) -> None:
        self.received: np.ndarray | None = None

    def readtext(self, image: np.ndarray, **options: object) -> list[tuple[object, str, float]]:
        if options != {"detail": 1, "paragraph": False}:
            raise AssertionError(options)
        self.received = image.copy()
        return [([], " first ", 0.8), ([], "second", 0.6)]


class ComponentContractTests(unittest.TestCase):
    def test_callable_detector_accepts_typed_output(self) -> None:
        expected = Detection(Box(2, 3, 10, 12), score=0.8)
        detector = CallableBubbleDetector(lambda image: [expected])
        self.assertEqual(detector.detect(make_image()), [expected])

    def test_callable_detector_rejects_untyped_output(self) -> None:
        for output in (None, {"box": [1, 2, 3, 4]}, [object()]):
            with self.subTest(output=output):
                detector = CallableBubbleDetector(lambda image, value=output: value)
                with self.assertRaises(ModelContractError):
                    detector.detect(make_image())

    def test_callable_detector_rejects_out_of_bounds_box(self) -> None:
        detection = Detection(Box(0, 0, 31, 10), score=0.7)
        detector = CallableBubbleDetector(lambda image: [detection])
        with self.assertRaisesRegex(ModelContractError, "outside"):
            detector.detect(make_image())

    def test_easyocr_adapter_uses_injected_reader_and_converts_to_bgr(self) -> None:
        reader = FakeReader()
        recognition = EasyOcrRecognizer(reader=reader).recognize(
            make_image(), language="korean"
        )
        self.assertEqual(recognition.text, "first\nsecond")
        self.assertAlmostEqual(recognition.confidence or 0.0, 0.7)
        self.assertIsNotNone(reader.received)
        self.assertEqual(reader.received[0, 0].tolist(), [0, 0, 255])

    def test_easyocr_reader_factory_is_lazy_and_cached(self) -> None:
        calls: list[list[str]] = []

        def factory(languages: list[str]) -> FakeReader:
            calls.append(languages)
            return FakeReader()

        recognizer = EasyOcrRecognizer(reader_factory=factory)
        self.assertEqual(calls, [])
        recognizer.recognize(make_image(), language="ko")
        recognizer.recognize(make_image(), language="ko")
        self.assertEqual(calls, [["ko", "en"]])

    def test_easyocr_rejects_malformed_output(self) -> None:
        class BadReader:
            def readtext(self, image: np.ndarray, **options: object) -> list[tuple]:
                return [([], "bad confidence", 1.5)]

        with self.assertRaisesRegex(ModelContractError, "confidence"):
            EasyOcrRecognizer(reader=BadReader()).recognize(make_image())

    def test_easyocr_wraps_model_loading_failure(self) -> None:
        class FakeEasyOcr:
            @staticmethod
            def Reader(*args: object, **kwargs: object) -> object:
                raise RuntimeError("weights unavailable")

        recognizer = EasyOcrRecognizer()

        with patch(
            "toontra.modules.easyocr_recognizer.importlib.import_module",
            return_value=FakeEasyOcr,
        ):
            with self.assertRaisesRegex(OptionalDependencyError, "could not load its model files"):
                recognizer.recognize(make_image())

    def test_null_recognizer_is_predictable(self) -> None:
        recognition = NullTextRecognizer().recognize(make_image(), language="ko")
        self.assertEqual(recognition.text, "")
        self.assertEqual(recognition.confidence, 0.0)

    def test_offline_translators_preserve_order(self) -> None:
        texts = ["hello", "unknown"]
        identity = IdentityTranslator()
        dictionary = DictionaryTranslator({"hello": "annyeong"})
        self.assertEqual(
            identity.translate(texts, source_language="en", target_language="ko"),
            texts,
        )
        self.assertEqual(
            dictionary.translate(texts, source_language="en", target_language="ko"),
            ["annyeong", "unknown"],
        )

    def test_translator_rejects_bare_string(self) -> None:
        with self.assertRaises(ModelContractError):
            IdentityTranslator().translate("hello", source_language="en", target_language="ko")

    def test_translation_validation_rejects_bare_string(self) -> None:
        with self.assertRaisesRegex(ModelContractError, "sequence of strings"):
            validate_translations("abc", 3)

if __name__ == "__main__":
    unittest.main()
