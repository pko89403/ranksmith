from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ranksmith.confidence._encoder import FrozenAutoEncoder
from ranksmith.confidence._errors import ConfidenceArtifactError, ConfidenceInputError
from ranksmith.confidence._features import (
    FEATURE_DIM,
    FEATURE_DTYPE,
    FEATURE_SCHEMA_VERSION,
    GRANULARITY,
    LOCAL_STRIDE,
    LOCAL_WINDOW_SIZE,
    extract_structural_features,
)
from ranksmith.confidence._scorer import (
    ARTIFACT_SCHEMA_VERSION,
    load_lightgbm_scorer,
    validate_scorer_metadata,
)
from ranksmith.confidence._templates import (
    INPUT_TEMPLATE_VERSION,
    format_confidence_input,
)
from ranksmith.confidence._types import (
    StructuralConfidenceInput,
    StructuralConfidenceResult,
    StructuralConfidenceScorer,
    TaskType,
)


class StructuralConfidenceEncoder(Protocol):
    @property
    def encoder_name(self) -> str:
        """Encoder model name."""

    @property
    def encoder_revision(self) -> str | None:
        """Encoder model revision."""

    @property
    def tokenizer_name(self) -> str:
        """Tokenizer name."""

    @property
    def tokenizer_revision(self) -> str | None:
        """Tokenizer revision."""

    @property
    def max_length(self) -> int:
        """Maximum token length used by the encoder."""

    def encode(self, text: str) -> tuple[object, object]:
        """Return final hidden states and attention mask for one text input."""


@dataclass(frozen=True)
class StructuralConfidenceEstimator:
    """Single-item structural confidence estimator."""

    encoder: StructuralConfidenceEncoder
    scorer: StructuralConfidenceScorer
    task_type: TaskType

    def __post_init__(self) -> None:
        _validate_metadata_for_encoder(
            scorer=self.scorer,
            encoder=self.encoder,
            task_type=self.task_type,
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        scorer: StructuralConfidenceScorer,
        task_type: TaskType,
        encoder_name: str = "bert-base-uncased",
        encoder_revision: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        hf_token: str | None = None,
        local_files_only: bool = False,
        cache_dir: str | None = None,
        device: str = "cpu",
        max_length: int = 256,
        allow_truncation: bool = False,
    ) -> StructuralConfidenceEstimator:
        encoder = FrozenAutoEncoder.from_pretrained(
            encoder_name=encoder_name,
            encoder_revision=encoder_revision,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            hf_token=hf_token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            device=device,
            max_length=max_length,
            allow_truncation=allow_truncation,
        )
        return cls(encoder=encoder, scorer=scorer, task_type=task_type)

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        metadata_path: str | Path | None = None,
        encoder_name: str | None = None,
        encoder_revision: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        task_type: TaskType | None = None,
        hf_token: str | None = None,
        local_files_only: bool = False,
        cache_dir: str | None = None,
        device: str = "cpu",
        max_length: int | None = None,
        allow_truncation: bool = False,
    ) -> StructuralConfidenceEstimator:
        scorer = load_lightgbm_scorer(
            artifact_path,
            metadata_path=metadata_path,
        )
        metadata = scorer.metadata
        resolved_task_type = metadata.task_type if task_type is None else task_type
        resolved_encoder_name = (
            metadata.encoder_name if encoder_name is None else encoder_name
        )
        resolved_encoder_revision = (
            metadata.encoder_revision if encoder_revision is None else encoder_revision
        )
        resolved_tokenizer_name = (
            metadata.tokenizer_name if tokenizer_name is None else tokenizer_name
        )
        resolved_tokenizer_revision = (
            metadata.tokenizer_revision
            if tokenizer_revision is None
            else tokenizer_revision
        )
        resolved_max_length = metadata.max_length if max_length is None else max_length
        validate_scorer_metadata(
            metadata,
            encoder_name=resolved_encoder_name,
            encoder_revision=resolved_encoder_revision,
            tokenizer_name=resolved_tokenizer_name,
            tokenizer_revision=resolved_tokenizer_revision,
            task_type=resolved_task_type,
            max_length=resolved_max_length,
            input_template_version=INPUT_TEMPLATE_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_dim=FEATURE_DIM,
        )
        return cls.from_pretrained(
            scorer=scorer,
            task_type=resolved_task_type,
            encoder_name=resolved_encoder_name,
            encoder_revision=resolved_encoder_revision,
            tokenizer_name=resolved_tokenizer_name,
            tokenizer_revision=resolved_tokenizer_revision,
            hf_token=hf_token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            device=device,
            max_length=resolved_max_length,
            allow_truncation=allow_truncation,
        )

    def score(self, item: StructuralConfidenceInput) -> StructuralConfidenceResult:
        text = format_confidence_input(self.task_type, item)
        hidden_states, mask = self.encoder.encode(text)
        features = extract_structural_features(
            hidden_states,
            mask,
            max_length=self.encoder.max_length,
        )
        score = _validate_confidence_score(self.scorer.predict_confidence(features))
        return StructuralConfidenceResult(
            score=score,
            task_type=self.task_type,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            metadata=_result_metadata(
                encoder=self.encoder,
                scorer=self.scorer,
            ),
        )

    def score_batch(
        self,
        items: Sequence[StructuralConfidenceInput],
        *,
        batch_size: int = 8,
        max_workers: int = 1,
        max_batch_items: int | None = None,
    ) -> list[StructuralConfidenceResult]:
        _validate_batch_options(
            items,
            batch_size=batch_size,
            max_workers=max_workers,
            max_batch_items=max_batch_items,
        )
        results: list[StructuralConfidenceResult] = []
        for chunk in _chunked(items, batch_size):
            results.extend(self._score_chunk(chunk, max_workers=max_workers))
        return results

    def _score_chunk(
        self,
        items: Sequence[StructuralConfidenceInput],
        *,
        max_workers: int,
    ) -> list[StructuralConfidenceResult]:
        if max_workers == 1:
            return [self.score(item) for item in items]
        return _score_chunk_parallel(self, items, max_workers=max_workers)


def _validate_batch_options(
    items: Sequence[StructuralConfidenceInput],
    *,
    batch_size: int,
    max_workers: int,
    max_batch_items: int | None,
) -> None:
    if not items:
        raise ConfidenceInputError("items must not be empty.")
    _validate_positive_int_option("batch_size", batch_size)
    _validate_positive_int_option("max_workers", max_workers)
    if max_batch_items is not None:
        _validate_positive_int_option("max_batch_items", max_batch_items)
    if max_batch_items is not None and len(items) > max_batch_items:
        raise ConfidenceInputError("items exceeds max_batch_items.")


def _validate_positive_int_option(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfidenceInputError(f"{name} must be an int.")
    if value < 1:
        raise ConfidenceInputError(f"{name} must be >= 1.")


def _chunked(
    items: Sequence[StructuralConfidenceInput],
    size: int,
) -> Iterator[Sequence[StructuralConfidenceInput]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _score_chunk_parallel(
    estimator: StructuralConfidenceEstimator,
    items: Sequence[StructuralConfidenceInput],
    *,
    max_workers: int,
) -> list[StructuralConfidenceResult]:
    indexed_items = list(enumerate(items))

    def score_indexed(
        value: tuple[int, StructuralConfidenceInput],
    ) -> tuple[int, StructuralConfidenceResult]:
        index, item = value
        return index, estimator.score(item)

    results: list[StructuralConfidenceResult | None] = [None] * len(indexed_items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(score_indexed, indexed): indexed[0]
            for indexed in indexed_items
        }
        try:
            for future in as_completed(future_to_index):
                index, result = future.result()
                results[index] = result
        except Exception:
            for future in future_to_index:
                future.cancel()
            raise

    return [_require_result(result) for result in results]


def _require_result(
    result: StructuralConfidenceResult | None,
) -> StructuralConfidenceResult:
    if result is None:
        raise ConfidenceArtifactError("parallel confidence result is missing.")
    return result


def _validate_metadata_for_encoder(
    *,
    scorer: StructuralConfidenceScorer,
    encoder: StructuralConfidenceEncoder,
    task_type: TaskType,
) -> None:
    validate_scorer_metadata(
        scorer.metadata,
        encoder_name=encoder.encoder_name,
        encoder_revision=encoder.encoder_revision,
        tokenizer_name=encoder.tokenizer_name,
        tokenizer_revision=encoder.tokenizer_revision,
        task_type=task_type,
        max_length=encoder.max_length,
        input_template_version=INPUT_TEMPLATE_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_dim=FEATURE_DIM,
    )


def _validate_confidence_score(score: object) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ConfidenceArtifactError("confidence score must be numeric")
    value = float(score)
    if not math.isfinite(value):
        raise ConfidenceArtifactError("confidence score must be finite")
    if value < 0.0 or value > 1.0:
        raise ConfidenceArtifactError("confidence score must be in [0, 1]")
    return value


def _result_metadata(
    *,
    encoder: StructuralConfidenceEncoder,
    scorer: StructuralConfidenceScorer,
) -> dict[str, object]:
    return {
        "encoder_name": encoder.encoder_name,
        "encoder_revision": encoder.encoder_revision,
        "tokenizer_name": encoder.tokenizer_name,
        "tokenizer_revision": encoder.tokenizer_revision,
        "max_length": encoder.max_length,
        "feature_dim": FEATURE_DIM,
        "feature_dtype": FEATURE_DTYPE,
        "granularity": GRANULARITY,
        "local_window_size": LOCAL_WINDOW_SIZE,
        "local_stride": LOCAL_STRIDE,
        "input_template_version": INPUT_TEMPLATE_VERSION,
        "scorer_type": scorer.metadata.scorer_type,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    }
