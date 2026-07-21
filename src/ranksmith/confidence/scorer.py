from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

from ranksmith.confidence.dependencies import import_optional_dependency
from ranksmith.confidence.errors import ConfidenceArtifactError
from ranksmith.confidence.features import (
    FEATURE_DIM,
    FEATURE_DTYPE,
    FEATURE_SCHEMA_VERSION,
    GRANULARITY,
    LOCAL_STRIDE,
    LOCAL_WINDOW_SIZE,
    MIN_MAX_LENGTH,
)
from ranksmith.confidence.templates import INPUT_TEMPLATE_VERSION
from ranksmith.confidence.types import (
    ScoreOutput,
    ScorerMetadata,
    StructuralConfidenceScorer,
    TaskType,
)

ARTIFACT_SCHEMA_VERSION = "structural-artifact-v1"
SUPPORTED_SCORE_OUTPUT = "probability"


def metadata_from_dict(data: Mapping[str, Any]) -> ScorerMetadata:
    _ensure_json_object(data)
    constructor_names = {field.name for field in fields(ScorerMetadata)} - {"extra"}
    missing = sorted(name for name in constructor_names if name not in data)
    if missing:
        raise ConfidenceArtifactError(f"missing scorer metadata fields: {missing}")

    extra = {key: value for key, value in data.items() if key not in constructor_names}
    metadata = ScorerMetadata(
        artifact_schema_version=_required_str(
            data, "artifact_schema_version", allow_empty=False
        ),
        scorer_type=_required_str(data, "scorer_type", allow_empty=False),
        task_type=_task_type(data["task_type"]),
        encoder_name=_required_str(data, "encoder_name", allow_empty=False),
        encoder_revision=_optional_str(data["encoder_revision"]),
        tokenizer_name=_required_str(data, "tokenizer_name", allow_empty=False),
        tokenizer_revision=_optional_str(data["tokenizer_revision"]),
        input_template_version=_required_str(
            data, "input_template_version", allow_empty=False
        ),
        feature_schema_version=_required_str(
            data, "feature_schema_version", allow_empty=False
        ),
        feature_dim=_required_int(data, "feature_dim"),
        feature_dtype=_required_str(data, "feature_dtype", allow_empty=False),
        max_length=_required_int(data, "max_length"),
        granularity=_required_str(data, "granularity", allow_empty=False),
        local_window_size=_required_int(data, "local_window_size"),
        local_stride=_required_int(data, "local_stride"),
        score_output=_score_output(data["score_output"]),
        positive_class_index=_required_int(data, "positive_class_index"),
        extra=extra,
    )
    _validate_metadata_values(metadata)
    return metadata


def validate_scorer_metadata(
    metadata: ScorerMetadata,
    *,
    encoder_name: str,
    encoder_revision: str | None,
    tokenizer_name: str,
    tokenizer_revision: str | None,
    task_type: TaskType,
    max_length: int,
    input_template_version: str,
    feature_schema_version: str,
    feature_dim: int,
) -> None:
    _validate_metadata_values(metadata)
    expected: dict[str, object] = {
        "encoder_name": encoder_name,
        "encoder_revision": encoder_revision,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "task_type": task_type,
        "max_length": max_length,
        "input_template_version": input_template_version,
        "feature_schema_version": feature_schema_version,
        "feature_dim": feature_dim,
        "feature_dtype": FEATURE_DTYPE,
        "granularity": GRANULARITY,
        "local_window_size": LOCAL_WINDOW_SIZE,
        "local_stride": LOCAL_STRIDE,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "score_output": SUPPORTED_SCORE_OUTPUT,
    }
    for name, value in expected.items():
        if getattr(metadata, name) != value:
            raise ConfidenceArtifactError(f"scorer metadata mismatch: {name}")


class LightGBMScorer:
    def __init__(self, *, model: object, metadata: ScorerMetadata) -> None:
        _validate_metadata_values(metadata)
        self.model = model
        self.metadata = metadata

    def predict_confidence(self, features: Sequence[float]) -> float:
        rows = [_validate_features(features, feature_dim=self.metadata.feature_dim)]
        if hasattr(self.model, "predict_proba"):
            raw = self.model.predict_proba(rows)
            score = _extract_predict_proba_score(
                raw,
                positive_class_index=self.metadata.positive_class_index,
            )
        elif hasattr(self.model, "predict"):
            raw = self.model.predict(rows)
            score = _extract_predict_score(
                raw,
                positive_class_index=self.metadata.positive_class_index,
            )
        else:
            raise ConfidenceArtifactError(
                "scorer model must provide predict_proba() or predict()"
            )
        return _validate_probability(score)


class JoblibScorerWrapper:
    def __init__(self, *, scorer: object, metadata: ScorerMetadata) -> None:
        if not hasattr(scorer, "predict_confidence"):
            raise ConfidenceArtifactError(
                "joblib scorer must provide predict_confidence()"
            )
        _validate_metadata_values(metadata)
        self._scorer = scorer
        self.metadata = metadata

    def predict_confidence(self, features: Sequence[float]) -> float:
        validated = _validate_features(features, feature_dim=self.metadata.feature_dim)
        raw = self._scorer.predict_confidence(validated)
        try:
            score = float(raw)
        except (TypeError, ValueError) as exc:
            raise ConfidenceArtifactError("confidence score must be numeric") from exc
        return _validate_probability(score)


def load_lightgbm_scorer(
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> StructuralConfidenceScorer:
    # Route by artifact content, not by whether metadata_path was passed: the
    # training pipeline exports a self-contained joblib dict, so a caller who
    # also has a metadata sidecar (write_metadata_json) and passes metadata_path
    # must still load correctly. Only a raw LightGBM Booster file — which
    # joblib cannot unpickle — needs the sidecar/explicit metadata.
    artifact_path = Path(path)
    artifact = _try_load_joblib(artifact_path)
    if artifact is not _JOBLIB_LOAD_FAILED:
        return _joblib_scorer_from_artifact(artifact)

    resolved_metadata_path = _resolve_metadata_path(artifact_path, metadata_path)
    if resolved_metadata_path is None:
        raise ConfidenceArtifactError(
            "raw LightGBM model file requires a metadata sidecar or metadata_path"
        )
    return _load_lightgbm_booster_scorer(
        artifact_path,
        metadata_path=resolved_metadata_path,
    )


_JOBLIB_LOAD_FAILED = object()


def _try_load_joblib(path: Path) -> object:
    joblib = import_optional_dependency("joblib")
    try:
        return joblib.load(path)
    except Exception:
        return _JOBLIB_LOAD_FAILED


def _joblib_scorer_from_artifact(artifact: object) -> StructuralConfidenceScorer:
    if _has_predict_confidence(artifact) and hasattr(artifact, "metadata"):
        return JoblibScorerWrapper(
            scorer=artifact,
            metadata=_coerce_metadata(artifact.metadata),
        )

    if not isinstance(artifact, Mapping):
        raise ConfidenceArtifactError(
            "joblib artifact must be a mapping or scorer wrapper"
        )
    if "metadata" not in artifact:
        raise ConfidenceArtifactError("joblib artifact requires metadata")
    metadata = metadata_from_dict(_require_mapping(artifact["metadata"], "metadata"))

    if "scorer" in artifact:
        return JoblibScorerWrapper(scorer=artifact["scorer"], metadata=metadata)
    if "model" in artifact:
        return LightGBMScorer(model=artifact["model"], metadata=metadata)
    raise ConfidenceArtifactError("joblib artifact requires model or scorer")


def _load_lightgbm_booster_scorer(
    path: Path,
    *,
    metadata_path: Path,
) -> LightGBMScorer:
    lightgbm = import_optional_dependency("lightgbm")
    metadata = metadata_from_dict(_read_metadata_json(metadata_path))
    return LightGBMScorer(
        model=lightgbm.Booster(model_file=str(path)),
        metadata=metadata,
    )


def _resolve_metadata_path(
    path: Path,
    metadata_path: str | Path | None,
) -> Path | None:
    if metadata_path is not None:
        return Path(metadata_path)
    default_metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if default_metadata_path.exists():
        return default_metadata_path
    return None


def _read_metadata_json(path: Path) -> Mapping[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfidenceArtifactError("metadata JSON must be valid JSON") from exc
    except OSError as exc:
        raise ConfidenceArtifactError("metadata JSON could not be read") from exc
    return _require_mapping(loaded, "metadata JSON")


def _coerce_metadata(value: object) -> ScorerMetadata:
    if isinstance(value, ScorerMetadata):
        _validate_metadata_values(value)
        return value
    return metadata_from_dict(_require_mapping(value, "metadata"))


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfidenceArtifactError(f"{name} must be a JSON object")
    return value


def _ensure_json_object(data: Mapping[str, Any]) -> None:
    if any(not isinstance(key, str) for key in data):
        raise ConfidenceArtifactError("metadata keys must be strings")
    try:
        encoded = json.dumps(dict(data))
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ConfidenceArtifactError("metadata must be JSON-serializable") from exc
    if not isinstance(decoded, dict):
        raise ConfidenceArtifactError("metadata must be a JSON object")


def _validate_metadata_values(metadata: ScorerMetadata) -> None:
    if metadata.artifact_schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ConfidenceArtifactError("unsupported artifact_schema_version")
    if metadata.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ConfidenceArtifactError("unsupported feature_schema_version")
    if metadata.feature_dim != FEATURE_DIM:
        raise ConfidenceArtifactError("unsupported feature_dim")
    if metadata.feature_dtype != FEATURE_DTYPE:
        raise ConfidenceArtifactError("unsupported feature_dtype")
    if metadata.score_output != SUPPORTED_SCORE_OUTPUT:
        raise ConfidenceArtifactError("score_output must be probability")
    if metadata.positive_class_index < 0:
        raise ConfidenceArtifactError("positive_class_index must be non-negative")
    if metadata.max_length < MIN_MAX_LENGTH:
        raise ConfidenceArtifactError("max_length is too small")
    if metadata.granularity != GRANULARITY:
        raise ConfidenceArtifactError("unsupported granularity")
    if metadata.local_window_size != LOCAL_WINDOW_SIZE:
        raise ConfidenceArtifactError("unsupported local_window_size")
    if metadata.local_stride != LOCAL_STRIDE:
        raise ConfidenceArtifactError("unsupported local_stride")
    if metadata.input_template_version != INPUT_TEMPLATE_VERSION:
        raise ConfidenceArtifactError("unsupported input_template_version")
    if metadata.task_type not in {
        "answer_confidence",
        "judgment_confidence",
        "query_answerability_confidence",
        "query_context_answerability_confidence",
    }:
        raise ConfidenceArtifactError("invalid task_type")
    if not metadata.scorer_type.strip():
        raise ConfidenceArtifactError("scorer_type must be non-empty")


def _required_str(
    data: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool,
) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise ConfidenceArtifactError(f"{key} must be a string")
    if not allow_empty and not value.strip():
        raise ConfidenceArtifactError(f"{key} must be non-empty")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfidenceArtifactError("optional metadata string must be string or null")
    return value


def _required_int(data: Mapping[str, Any], key: str) -> int:
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfidenceArtifactError(f"{key} must be an integer")
    return value


def _task_type(value: object) -> TaskType:
    if value == "answer_confidence":
        return "answer_confidence"
    if value == "judgment_confidence":
        return "judgment_confidence"
    if value == "query_answerability_confidence":
        return "query_answerability_confidence"
    if value == "query_context_answerability_confidence":
        return "query_context_answerability_confidence"
    raise ConfidenceArtifactError("invalid task_type")


def _score_output(value: object) -> ScoreOutput:
    if value == SUPPORTED_SCORE_OUTPUT:
        return "probability"
    raise ConfidenceArtifactError("score_output must be probability")


def _validate_features(features: Sequence[float], *, feature_dim: int) -> list[float]:
    if len(features) != feature_dim:
        raise ConfidenceArtifactError("feature length does not match metadata")
    values: list[float] = []
    for feature in features:
        try:
            value = float(feature)
        except (TypeError, ValueError) as exc:
            raise ConfidenceArtifactError("feature vector must be numeric") from exc
        if not math.isfinite(value):
            raise ConfidenceArtifactError("feature vector must be finite")
        values.append(value)
    return values


def _extract_predict_proba_score(
    raw: object,
    *,
    positive_class_index: int,
) -> float:
    np = import_optional_dependency("numpy")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 1:
        raise ConfidenceArtifactError("predict_proba output must have shape (1, n)")
    if positive_class_index >= array.shape[1]:
        raise ConfidenceArtifactError("positive_class_index is out of range")
    return float(array[0, positive_class_index])


def _extract_predict_score(
    raw: object,
    *,
    positive_class_index: int,
) -> float:
    np = import_optional_dependency("numpy")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    if array.ndim == 1:
        if array.shape[0] != 1:
            raise ConfidenceArtifactError("predict output has ambiguous shape")
        return float(array[0])
    if array.ndim == 2:
        if array.shape[0] != 1:
            raise ConfidenceArtifactError("predict output has ambiguous rows")
        if array.shape[1] == 1:
            return float(array[0, 0])
        if positive_class_index >= array.shape[1]:
            raise ConfidenceArtifactError("positive_class_index is out of range")
        return float(array[0, positive_class_index])
    raise ConfidenceArtifactError("predict output has invalid shape")


def _validate_probability(score: float) -> float:
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ConfidenceArtifactError("confidence score must be a probability")
    return score


def _has_predict_confidence(value: object) -> bool:
    method = getattr(value, "predict_confidence", None)
    return callable(method)
