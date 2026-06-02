from __future__ import annotations

from ranksmith.confidence._errors import ConfidenceDependencyError


def load_lightgbm_scorer(path: str) -> object:
    raise ConfidenceDependencyError("lightgbm scorer loading is not implemented yet")
