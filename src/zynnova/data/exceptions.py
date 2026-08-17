"""Exceptions raised by :mod:`zynnova.data`."""


class DataError(RuntimeError):
    """Base class for data-layer failures."""


class DatasetNotFoundError(DataError):
    """Raised when a dataset plugin is not registered."""


class DownloadError(DataError):
    """Raised when downloading or validating a remote artifact fails."""


class PreparationError(DataError):
    """Raised when raw data cannot be converted into samples."""


class SchemaError(DataError):
    """Raised when a sample does not satisfy a task or field schema."""


class StorageError(DataError):
    """Raised when a prepared dataset cannot be read or written."""
