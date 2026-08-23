"""Public Toontra reconstruction API."""

from importlib.metadata import version

from .errors import (
    ImageValidationError,
    ModelContractError,
    OptionalDependencyError,
    OutputExistsError,
    ToontraError,
)
from .models import Box, BubbleResult, Detection, ModelMetadata, PageResult, Recognition
from .pipeline import Toontra

__version__ = version("toontra")

__all__ = [
    "Box",
    "BubbleResult",
    "Detection",
    "ImageValidationError",
    "ModelContractError",
    "ModelMetadata",
    "OptionalDependencyError",
    "OutputExistsError",
    "PageResult",
    "Recognition",
    "Toontra",
    "ToontraError",
    "__version__",
]
