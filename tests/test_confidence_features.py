from __future__ import annotations

import numpy as np

from ranksmith.confidence._features import (
    FEATURE_DIM,
    FEATURE_DTYPE,
    FEATURE_SCHEMA_VERSION,
    extract_structural_features,
)


def test_extract_structural_features_returns_70_finite_values() -> None:
    hidden_states = np.arange(6 * 4, dtype=np.float64).reshape(6, 4)
    attention_mask = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)

    features = extract_structural_features(
        hidden_states,
        attention_mask,
        max_length=64,
    )

    assert FEATURE_SCHEMA_VERSION == "structural-v1"
    assert FEATURE_DIM == 70
    assert FEATURE_DTYPE == "float64"
    assert len(features) == 70
    assert all(np.isfinite(features))


def test_feature_order_has_expected_family_lengths() -> None:
    hidden_states = np.eye(8, 4, dtype=np.float64)
    attention_mask = np.ones(8, dtype=np.int64)

    features = extract_structural_features(
        hidden_states,
        attention_mask,
        max_length=64,
    )

    spectral = features[:48]
    local = features[48:54]
    shape = features[54:]

    assert len(spectral) == 48
    assert len(local) == 6
    assert len(shape) == 16


def test_local_variation_matches_structural_v1_order() -> None:
    hidden_states = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
        ],
        dtype=np.float64,
    )
    attention_mask = np.ones(3, dtype=np.int64)

    features = extract_structural_features(
        hidden_states,
        attention_mask,
        max_length=64,
    )

    expected = np.array(
        [
            10.0,
            5.0,
            0.0,
            10.0,
            8.3333333333,
            5.0,
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(features[48:54], expected, rtol=1e-10, atol=1e-10)
