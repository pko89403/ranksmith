from __future__ import annotations

import importlib

import pytest

from ranksmith.confidence import ConfidenceDependencyError
from ranksmith.confidence._dependencies import import_optional_dependency


def test_import_ranksmith_confidence_without_optional_dependencies() -> None:
    confidence = importlib.import_module("ranksmith.confidence")

    assert confidence.AnswerConfidenceInput is not None


def test_missing_optional_dependency_raises_confidence_dependency_error() -> None:
    with pytest.raises(ConfidenceDependencyError) as error:
        import_optional_dependency(
            "definitely_missing_package",
            extra="confidence",
        )

    assert "pip install ranksmith[confidence]" in str(error.value)
