from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ranksmith.confidence import ConfidenceArtifactError, ScorerMetadata
from ranksmith.confidence._scorer import (
    ARTIFACT_SCHEMA_VERSION,
    JoblibScorerWrapper,
    LightGBMScorer,
    load_lightgbm_scorer,
    metadata_from_dict,
    validate_scorer_metadata,
)


def metadata_dict(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "scorer_type": "lightgbm",
        "task_type": "answer_confidence",
        "encoder_name": "bert-base-uncased",
        "encoder_revision": None,
        "tokenizer_name": "bert-base-uncased",
        "tokenizer_revision": None,
        "input_template_version": "structural-template-v1",
        "feature_schema_version": "structural-v1",
        "feature_dim": 70,
        "feature_dtype": "float64",
        "max_length": 256,
        "granularity": "two_scale",
        "local_window_size": 5,
        "local_stride": 2,
        "score_output": "probability",
        "positive_class_index": 1,
        "extra_note": "preserved",
    }
    data.update(overrides)
    return data


def metadata(**overrides: object) -> ScorerMetadata:
    return metadata_from_dict(metadata_dict(**overrides))


def install_fake_joblib(monkeypatch: pytest.MonkeyPatch, artifact: object) -> None:
    module = ModuleType("joblib")

    def load(path: str | Path) -> object:
        del path
        return artifact

    module.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "joblib", module)


def install_fake_lightgbm(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    loaded_paths: list[str] = []
    module = ModuleType("lightgbm")

    class FakeBooster:
        def __init__(self, *, model_file: str) -> None:
            loaded_paths.append(model_file)

        def predict(self, rows: list[list[float]]) -> list[list[float]]:
            assert len(rows) == 1
            return [[0.25, 0.75]]

    module.Booster = FakeBooster  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lightgbm", module)
    return loaded_paths


def test_metadata_from_dict_preserves_unknown_fields() -> None:
    parsed = metadata()

    assert parsed.extra == {"extra_note": "preserved"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_schema_version", "other"),
        ("feature_schema_version", "other"),
        ("feature_dim", 69),
        ("feature_dtype", "float32"),
        ("score_output", "margin"),
        ("positive_class_index", -1),
        ("max_length", 33),
        ("granularity", "global"),
        ("local_window_size", 6),
        ("local_stride", 1),
        ("task_type", "other"),
        ("input_template_version", "other"),
    ],
)
def test_metadata_from_dict_rejects_invalid_contract_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ConfidenceArtifactError):
        metadata_from_dict(metadata_dict(**{field: value}))


def test_metadata_from_dict_requires_json_object_and_serializable_values() -> None:
    with pytest.raises(ConfidenceArtifactError):
        metadata_from_dict(metadata_dict(extra_note={1, 2}))

    with pytest.raises(ConfidenceArtifactError):
        metadata_from_dict({1: "not a string key"})  # type: ignore[dict-item]


def test_metadata_from_dict_requires_exact_field_types() -> None:
    with pytest.raises(ConfidenceArtifactError):
        metadata_from_dict(metadata_dict(feature_dim=True))

    with pytest.raises(ConfidenceArtifactError):
        metadata_from_dict(metadata_dict(encoder_revision=123))


def test_validate_scorer_metadata_accepts_matching_metadata() -> None:
    validate_scorer_metadata(
        metadata(),
        encoder_name="bert-base-uncased",
        encoder_revision=None,
        tokenizer_name="bert-base-uncased",
        tokenizer_revision=None,
        task_type="answer_confidence",
        max_length=256,
        input_template_version="structural-template-v1",
        feature_schema_version="structural-v1",
        feature_dim=70,
    )


def test_validate_scorer_metadata_rejects_mismatch() -> None:
    with pytest.raises(ConfidenceArtifactError):
        validate_scorer_metadata(
            metadata(encoder_name="other"),
            encoder_name="bert-base-uncased",
            encoder_revision=None,
            tokenizer_name="bert-base-uncased",
            tokenizer_revision=None,
            task_type="answer_confidence",
            max_length=256,
            input_template_version="structural-template-v1",
            feature_schema_version="structural-v1",
            feature_dim=70,
        )


class FakeProbaModel:
    def predict_proba(self, rows: list[list[float]]) -> list[list[float]]:
        assert len(rows) == 1
        return [[0.2, 0.8]]


class FakePredictVectorModel:
    def predict(self, rows: list[list[float]]) -> list[float]:
        assert len(rows) == 1
        return [0.6]


class FakePredictMatrixClassModel:
    def predict(self, rows: list[list[float]]) -> list[list[float]]:
        assert len(rows) == 1
        return [[0.3, 0.7]]


class FakePredictMatrixSingleModel:
    def predict(self, rows: list[list[float]]) -> list[list[float]]:
        assert len(rows) == 1
        return [[0.4]]


def test_lightgbm_scorer_uses_predict_proba_positive_class() -> None:
    scorer = LightGBMScorer(model=FakeProbaModel(), metadata=metadata())

    assert scorer.predict_confidence([0.0] * 70) == 0.8


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (FakePredictVectorModel(), 0.6),
        (FakePredictMatrixClassModel(), 0.7),
        (FakePredictMatrixSingleModel(), 0.4),
    ],
)
def test_lightgbm_scorer_accepts_probability_predict_shapes(
    model: object,
    expected: float,
) -> None:
    scorer = LightGBMScorer(model=model, metadata=metadata())

    assert scorer.predict_confidence([0.0] * 70) == expected


def test_lightgbm_scorer_accepts_numpy_outputs() -> None:
    import numpy as np

    class NumpyProbaModel:
        def predict_proba(self, rows: list[list[float]]) -> object:
            assert len(rows) == 1
            return np.asarray([[0.1, 0.9]], dtype=np.float64)

    scorer = LightGBMScorer(model=NumpyProbaModel(), metadata=metadata())

    assert scorer.predict_confidence([0.0] * 70) == 0.9


@pytest.mark.parametrize(
    "raw",
    [
        [0.2, 0.8],
        [[0.1, 0.9], [0.2, 0.8]],
        [[[0.8]]],
    ],
)
def test_lightgbm_scorer_rejects_bad_predict_shapes(raw: object) -> None:
    class BadPredictModel:
        def predict(self, rows: list[list[float]]) -> object:
            del rows
            return raw

    scorer = LightGBMScorer(model=BadPredictModel(), metadata=metadata())

    with pytest.raises(ConfidenceArtifactError):
        scorer.predict_confidence([0.0] * 70)


def test_lightgbm_scorer_rejects_bad_predict_proba_shape() -> None:
    class BadShapeProbaModel:
        def predict_proba(self, rows: list[list[float]]) -> list[float]:
            del rows
            return [0.2, 0.8]

    scorer = LightGBMScorer(model=BadShapeProbaModel(), metadata=metadata())

    with pytest.raises(ConfidenceArtifactError):
        scorer.predict_confidence([0.0] * 70)


@pytest.mark.parametrize("score", [math.nan, math.inf, -0.1, 1.1])
def test_lightgbm_scorer_rejects_non_probability_scores(score: float) -> None:
    class BadScoreModel:
        def predict(self, rows: list[list[float]]) -> list[float]:
            del rows
            return [score]

    scorer = LightGBMScorer(model=BadScoreModel(), metadata=metadata())

    with pytest.raises(ConfidenceArtifactError):
        scorer.predict_confidence([0.0] * 70)


def test_lightgbm_scorer_validates_feature_vector() -> None:
    scorer = LightGBMScorer(model=FakePredictVectorModel(), metadata=metadata())

    with pytest.raises(ConfidenceArtifactError):
        scorer.predict_confidence([0.0] * 69)

    with pytest.raises(ConfidenceArtifactError):
        scorer.predict_confidence([0.0] * 69 + [math.nan])


def test_joblib_wrapper_validates_predict_confidence_contract() -> None:
    class FakeWrapper:
        metadata = metadata_dict(scorer_type="joblib-wrapper")

        def predict_confidence(self, features: list[float]) -> float:
            assert len(features) == 70
            return 0.55

    scorer = JoblibScorerWrapper(
        scorer=FakeWrapper(),
        metadata=metadata(scorer_type="joblib-wrapper"),
    )

    assert scorer.predict_confidence([0.0] * 70) == 0.55


def test_load_lightgbm_scorer_loads_joblib_dict_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_joblib(
        monkeypatch,
        {"model": FakePredictVectorModel(), "metadata": metadata_dict()},
    )

    scorer = load_lightgbm_scorer(tmp_path / "artifact.joblib")

    assert scorer.predict_confidence([0.0] * 70) == 0.6


def test_load_lightgbm_scorer_loads_joblib_wrapper_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeWrapper:
        metadata = metadata_dict(scorer_type="joblib-wrapper")

        def predict_confidence(self, features: list[float]) -> float:
            assert len(features) == 70
            return 0.45

    install_fake_joblib(monkeypatch, FakeWrapper())

    scorer = load_lightgbm_scorer(tmp_path / "wrapper.joblib")

    assert scorer.predict_confidence([0.0] * 70) == 0.45


def test_load_lightgbm_scorer_uses_default_metadata_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded_paths = install_fake_lightgbm(monkeypatch)
    model_path = tmp_path / "model.txt"
    model_path.write_text("fake booster", encoding="utf-8")
    metadata_path = model_path.with_suffix(model_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata_dict()), encoding="utf-8")

    scorer = load_lightgbm_scorer(model_path)

    assert loaded_paths == [str(model_path)]
    assert scorer.predict_confidence([0.0] * 70) == 0.75


def test_load_lightgbm_scorer_uses_explicit_metadata_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded_paths = install_fake_lightgbm(monkeypatch)
    model_path = tmp_path / "model.bin"
    metadata_path = tmp_path / "scorer-metadata.json"
    metadata_path.write_text(json.dumps(metadata_dict()), encoding="utf-8")

    scorer = load_lightgbm_scorer(model_path, metadata_path=metadata_path)

    assert loaded_paths == [str(model_path)]
    assert scorer.predict_confidence([0.0] * 70) == 0.75


def test_load_lightgbm_scorer_rejects_non_object_metadata_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_lightgbm(monkeypatch)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(["not", "object"]), encoding="utf-8")

    with pytest.raises(ConfidenceArtifactError):
        load_lightgbm_scorer(tmp_path / "model.txt", metadata_path=metadata_path)


def test_load_lightgbm_scorer_rejects_invalid_joblib_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_joblib(monkeypatch, {"metadata": metadata_dict()})

    with pytest.raises(ConfidenceArtifactError):
        load_lightgbm_scorer(tmp_path / "artifact.joblib")
