"""Shared, dependency-light infrastructure for ZynNova."""

from .artifacts import ArtifactRecord, RunManifest, sha256_file
from .backend import Availability, BackendDescriptor, BackendRegistry
from .exceptions import (
    BackendExecutionError,
    BackendUnavailableError,
    ConfigurationError,
    ConsentRequiredError,
    GeometryError,
    LicenseNotAcceptedError,
    ZynNovaError,
)
from .licenses import KNOWN_LICENSES, LicenseGate, require_known_license
from .process import ProcessResult, run_process
from .serialization import dump_json, load_json, to_jsonable

__all__ = [
    "ArtifactRecord",
    "Availability",
    "BackendDescriptor",
    "BackendExecutionError",
    "BackendRegistry",
    "BackendUnavailableError",
    "ConfigurationError",
    "ConsentRequiredError",
    "GeometryError",
    "KNOWN_LICENSES",
    "LicenseGate",
    "LicenseNotAcceptedError",
    "ProcessResult",
    "RunManifest",
    "ZynNovaError",
    "dump_json",
    "load_json",
    "require_known_license",
    "run_process",
    "sha256_file",
    "to_jsonable",
]
