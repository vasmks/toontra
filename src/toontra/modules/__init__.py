"""Built-in components for the Toontra pipeline."""

from toontra.modules.callable_bubble_detector import CallableBubbleDetector
from toontra.modules.easyocr_recognizer import EasyOcrRecognizer, NullTextRecognizer
from toontra.modules.manet_bubble_masker import ManetBubbleMasker
from toontra.modules.translation import DictionaryTranslator, IdentityTranslator
from toontra.modules.white_bubble_masker import WhiteBubbleMasker
from toontra.modules.yolo26_bubble_detector import Yolo26BubbleDetector

__all__ = [
    "CallableBubbleDetector",
    "DictionaryTranslator",
    "EasyOcrRecognizer",
    "IdentityTranslator",
    "NullTextRecognizer",
    "ManetBubbleMasker",
    "WhiteBubbleMasker",
    "Yolo26BubbleDetector",
]
