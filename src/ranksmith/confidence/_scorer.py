from __future__ import annotations

from pathlib import Path

from ranksmith.confidence._errors import ConfidenceDependencyError


def load_lightgbm_scorer(
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> object:
    del path, metadata_path
    raise ConfidenceDependencyError("lightgbm scorer loading is not implemented yet")
