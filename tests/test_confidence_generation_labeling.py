from __future__ import annotations

import math
from typing import Any, cast

import pytest

from ranksmith.confidence_generation.errors import ConfidenceGenerationInputError
from ranksmith.confidence_generation.labeling import (
    normalized_exact_match,
    relevance_truth,
)


def test_normalized_exact_match_uses_simple_normalization() -> None:
    assert normalized_exact_match(" Nancy   Travis ", "nancy travis")
    assert normalized_exact_match("Nancy Travis", ["Other", "nancy travis"])
    assert not normalized_exact_match("Nancy T.", "Nancy Travis")


def test_no_answer_value_is_always_mismatch() -> None:
    assert not normalized_exact_match("__NO_ANSWER__", "__NO_ANSWER__")
    assert not normalized_exact_match(" __no_answer__ ", "__NO_ANSWER__")


def test_relevance_truth_defaults_to_gt_zero() -> None:
    assert relevance_truth(1, threshold=0.0, operator="gt") == "relevant"
    assert relevance_truth(0, threshold=0.0, operator="gt") == "not_relevant"
    assert relevance_truth(True, threshold=100.0, operator="gt") == "relevant"
    assert relevance_truth(False, threshold=-1.0, operator="gte") == "not_relevant"


def test_relevance_truth_supports_gte_threshold() -> None:
    assert relevance_truth(2, threshold=2.0, operator="gte") == "relevant"
    assert relevance_truth(1, threshold=2.0, operator="gte") == "not_relevant"


def test_relevance_truth_rejects_invalid_values() -> None:
    with pytest.raises(ConfidenceGenerationInputError):
        relevance_truth(cast(Any, "1"), threshold=0.0, operator="gt")

    with pytest.raises(ConfidenceGenerationInputError):
        relevance_truth(1, threshold=0.0, operator="eq")


def test_relevance_truth_rejects_invalid_thresholds() -> None:
    with pytest.raises(ConfidenceGenerationInputError):
        relevance_truth(1, threshold=cast(Any, True), operator="gt")

    with pytest.raises(ConfidenceGenerationInputError):
        relevance_truth(1, threshold=cast(Any, "0"), operator="gt")

    with pytest.raises(ConfidenceGenerationInputError):
        relevance_truth(1, threshold=math.nan, operator="gt")
