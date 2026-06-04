from __future__ import annotations

import numpy as np
import pytest

from ranksmith.confidence import ConfidenceArtifactError, ConfidenceInputError
from ranksmith.confidence._features import extract_structural_features


def test_rejects_nan_hidden_states() -> None:
    hidden_states = np.array([[1.0, 2.0], [np.nan, 3.0]], dtype=np.float64)
    attention_mask = np.array([1, 1], dtype=np.int64)

    with pytest.raises(ConfidenceArtifactError):
        extract_structural_features(hidden_states, attention_mask, max_length=64)


def test_rejects_zero_non_padding_tokens() -> None:
    hidden_states = np.zeros((2, 4), dtype=np.float64)
    attention_mask = np.array([0, 0], dtype=np.int64)

    with pytest.raises(ConfidenceInputError):
        extract_structural_features(hidden_states, attention_mask, max_length=64)


def test_single_token_uses_zero_fallbacks() -> None:
    hidden_states = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    attention_mask = np.array([1], dtype=np.int64)

    features = extract_structural_features(
        hidden_states,
        attention_mask,
        max_length=64,
    )

    assert len(features) == 70
    assert all(np.isfinite(features))


def test_degree_zero_graph_stays_finite() -> None:
    hidden_states = np.zeros((4, 3), dtype=np.float64)
    attention_mask = np.ones(4, dtype=np.int64)

    features = extract_structural_features(
        hidden_states,
        attention_mask,
        max_length=64,
    )

    assert len(features) == 70
    assert all(np.isfinite(features))


def test_rejects_too_small_max_length() -> None:
    hidden_states = np.zeros((4, 3), dtype=np.float64)
    attention_mask = np.ones(4, dtype=np.int64)

    with pytest.raises(ConfidenceInputError):
        extract_structural_features(hidden_states, attention_mask, max_length=33)
