from __future__ import annotations

import importlib


def test_confidence_public_submodule_exports_are_available() -> None:
    confidence = importlib.import_module("ranksmith.confidence")

    assert hasattr(confidence, "AnswerConfidenceInput")
    assert hasattr(confidence, "JudgmentConfidenceInput")
    assert hasattr(confidence, "StructuralConfidenceEstimator")
    assert hasattr(confidence, "load_lightgbm_scorer")
    assert hasattr(confidence, "ConfidenceError")


def test_confidence_names_are_not_root_exports() -> None:
    ranksmith = importlib.import_module("ranksmith")

    assert not hasattr(ranksmith, "AnswerConfidenceInput")
    assert not hasattr(ranksmith, "JudgmentConfidenceInput")
    assert not hasattr(ranksmith, "StructuralConfidenceEstimator")
    assert not hasattr(ranksmith, "ConfidenceError")


def test_no_batch_or_async_api_in_phase_one() -> None:
    confidence = importlib.import_module("ranksmith.confidence")

    assert not hasattr(confidence.StructuralConfidenceEstimator, "score_batch")
    assert not hasattr(confidence.StructuralConfidenceEstimator, "ascore")
    assert not hasattr(confidence, "AsyncStructuralConfidenceEstimator")
