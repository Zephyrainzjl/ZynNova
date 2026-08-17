from __future__ import annotations

from enum import Enum


class UnitRole(str, Enum):
    REPEAT = "repeat"
    HEAD = "head"
    TAIL = "tail"
    BRANCH = "branch"
    CROSSLINKER = "crosslinker"
    ADDITIVE = "additive"
    SOLVENT = "solvent"
    FILLER = "filler"
    OTHER = "other"


class ArchitectureType(str, Enum):
    UNKNOWN = "unknown"
    LINEAR = "linear"
    RING = "ring"
    BLOCK = "block"
    RANDOM = "random"
    ALTERNATING = "alternating"
    GRADIENT = "gradient"
    GRAFT = "graft"
    COMB = "comb"
    BRUSH = "brush"
    STAR = "star"
    BRANCHED = "branched"
    DENDRITIC = "dendritic"
    NETWORK = "network"
    BLEND = "blend"
    COMPOSITE = "composite"


class Resolution(str, Enum):
    ATOMISTIC = "atomistic"
    UNITED_ATOM = "united_atom"
    REPEAT_UNIT = "repeat_unit"
    COARSE_GRAINED = "coarse_grained"
    MESOSCALE = "mesoscale"
    FIELD = "field"


class EdgeKind(str, Enum):
    COVALENT = "covalent"
    POLYMER_CONNECTION = "polymer_connection"
    BACKBONE = "backbone"
    SIDECHAIN = "sidechain"
    BRANCH = "branch"
    CROSSLINK = "crosslink"
    SPATIAL_NEIGHBOR = "spatial_neighbor"
    MEMBERSHIP = "membership"
    PROCESS = "process"


class DistributionKind(str, Enum):
    DELTA = "delta"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    SCHULZ_ZIMM = "schulz_zimm"
    HISTOGRAM = "histogram"
    EMPIRICAL_SAMPLES = "empirical_samples"
    CATEGORICAL = "categorical"
    CUSTOM = "custom"
