from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ranksmith.confidence._dependencies import import_optional_dependency
from ranksmith.confidence._encoder import FrozenAutoEncoder
from ranksmith.confidence_training._artifact import (
    export_scorer_artifact,
    write_metadata_json,
)
from ranksmith.confidence_training._calibration import (
    CalibratedConfidenceScorer,
    PredictiveModel,
)
from ranksmith.confidence_training._dataset import load_canonical_dataset
from ranksmith.confidence_training._features import EncoderLike, extract_feature_rows
from ranksmith.confidence_training._manifest import (
    build_dataset_manifest,
    build_split_manifest,
    json_hash,
    training_config_hash,
)
from ranksmith.confidence_training._report import generate_training_report
from ranksmith.confidence_training._split import split_dataset
from ranksmith.confidence_training._train import (
    calibrate_classifier,
    train_lightgbm_classifier,
)
from ranksmith.confidence_training._types import (
    CanonicalConfidenceSample,
    ConfidenceDatasetSplit,
    ConfidenceFeatureRow,
    ConfidenceTrainingConfig,
    ConfidenceTrainingReport,
    ConfidenceTrainingResult,
)


def train_confidence_scorer(
    config: ConfidenceTrainingConfig,
) -> ConfidenceTrainingResult:
    dataset_path = Path(config.dataset_path)
    output_dir = Path(config.output_dir)
    export_path = Path(config.export_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_canonical_dataset(dataset_path, task_type=config.task_type)
    split = _split_samples(samples, config=config)
    train_rows, valid_rows, test_rows = _extract_split_features(
        split,
        encoder=_build_encoder(config),
        output_dir=output_dir,
    )

    model = train_lightgbm_classifier(train_rows, seed=config.seed)
    scorer = calibrate_classifier(model, valid_rows)
    _write_training_report(
        model,
        scorer,
        valid_rows=valid_rows,
        test_rows=test_rows,
        output_dir=output_dir,
    )

    dataset_manifest = build_dataset_manifest(
        dataset_path,
        task_type=config.task_type,
        sample_count=len(samples),
    )
    split_manifest = build_split_manifest(
        split,
        seed=config.seed,
        task_type=config.task_type,
    )
    _write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    _write_json(output_dir / "split_manifest.json", split_manifest)
    _write_model(output_dir / "model.joblib", scorer)

    metadata = export_scorer_artifact(
        scorer,
        config=config,
        train_count=len(split.train),
        valid_count=len(split.valid),
        test_count=len(split.test),
        dataset_manifest_hash=json_hash(dataset_manifest),
        training_config_hash=training_config_hash(config),
    )
    metadata_path = output_dir / "metadata.json"
    write_metadata_json(metadata_path, metadata)

    return ConfidenceTrainingResult(
        output_dir=output_dir,
        export_path=export_path,
        report_path=output_dir / "report.json",
        metadata_path=metadata_path,
    )


def _split_samples(
    samples: list[CanonicalConfidenceSample],
    *,
    config: ConfidenceTrainingConfig,
) -> ConfidenceDatasetSplit:
    return split_dataset(
        samples,
        seed=config.seed,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
    )


def _build_encoder(config: ConfidenceTrainingConfig) -> FrozenAutoEncoder:
    return FrozenAutoEncoder.from_pretrained(
        encoder_name=config.encoder_name,
        encoder_revision=config.encoder_revision,
        tokenizer_name=config.tokenizer_name,
        tokenizer_revision=config.tokenizer_revision,
        hf_token=None,
        local_files_only=config.local_files_only,
        cache_dir=config.cache_dir,
        device="cpu",
        max_length=config.max_length,
        allow_truncation=config.allow_truncation,
    )


def _extract_split_features(
    split: ConfidenceDatasetSplit,
    *,
    encoder: EncoderLike,
    output_dir: Path,
) -> tuple[
    list[ConfidenceFeatureRow],
    list[ConfidenceFeatureRow],
    list[ConfidenceFeatureRow],
]:
    train_rows = extract_feature_rows(split.train, encoder=encoder)
    valid_rows = extract_feature_rows(split.valid, encoder=encoder)
    test_rows = extract_feature_rows(split.test, encoder=encoder)
    _write_feature_rows(output_dir / "features_train.jsonl", train_rows)
    _write_feature_rows(output_dir / "features_valid.jsonl", valid_rows)
    _write_feature_rows(output_dir / "features_test.jsonl", test_rows)
    return train_rows, valid_rows, test_rows


def _write_training_report(
    model: PredictiveModel,
    scorer: CalibratedConfidenceScorer,
    *,
    valid_rows: list[ConfidenceFeatureRow],
    test_rows: list[ConfidenceFeatureRow],
    output_dir: Path,
) -> None:
    report = generate_training_report(
        model,
        scorer,
        valid_rows=valid_rows,
        test_rows=test_rows,
    )
    _write_json(output_dir / "report.json", report.to_dict())
    _write_report_markdown(output_dir / "report.md", report)


def _write_model(path: Path, scorer: object) -> None:
    joblib = import_optional_dependency("joblib", extra="confidence-train")
    joblib.dump(scorer, path)


def _write_feature_rows(path: Path, rows: list[ConfidenceFeatureRow]) -> None:
    path.write_text(
        "".join(json.dumps(_feature_row_json(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _feature_row_json(row: ConfidenceFeatureRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "task_type": row.task_type,
        "label": row.label,
        "features": list(row.features),
        "feature_schema_version": row.feature_schema_version,
        "metadata": dict(row.metadata),
    }


def _write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_report_markdown(path: Path, report: ConfidenceTrainingReport) -> None:
    path.write_text(
        "# Confidence Training Report\n\n"
        f"```json\n{json.dumps(report.to_dict(), indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
