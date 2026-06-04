from __future__ import annotations

import importlib
from types import ModuleType

from ranksmith.confidence._errors import ConfidenceDependencyError


def import_optional_dependency(name: str, *, extra: str = "confidence") -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ConfidenceDependencyError(
            f"Optional dependency {name!r} is required. "
            f"Install it with `pip install ranksmith[{extra}]`."
        ) from exc
