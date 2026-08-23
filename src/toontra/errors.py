class ToontraError(Exception):
    """Base exception for expected Toontra failures."""


class ImageValidationError(ToontraError, ValueError):
    """Raised when an image or mask does not match the public contract."""


class ModelContractError(ToontraError, ValueError):
    """Raised when a replaceable component returns invalid data."""


class OptionalDependencyError(ToontraError, ImportError):
    """Raised when an explicitly requested optional adapter is unavailable."""


class OutputExistsError(ToontraError, FileExistsError):
    """Raised before output files would be overwritten."""
