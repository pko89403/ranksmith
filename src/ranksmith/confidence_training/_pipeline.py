from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ranksmith.confidence._encoder import FrozenAutoEncoder
from ranksmith.confidence_training._artifact import (
    export_scorer_artifact,
    write_metadata_json,
)
from ranksmith.confidence_training._dataset import load_canonical_dataset
from ranksmith.confidence_training._features import extract_feature_rows
from ranksmith.confidence_training._report import generate_training_report
from ranksmith.confidence_training._split import split_dataset
from ranksmith.confidence_training._train import (
    calibrate_classifier,
    train_lightgbm_classifier,
)
from ranksmith.confidence_training._types import (
    ConfidenceFeatureRow,
    ConfidenceTrainingConfig,
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
    split = split_dataset(
        samples,
        seed=config.seed,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
    )
    encoder = FrozenAutoEncoder.from_pretrained(
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

    train_rows = extract_feature_rows(split.train, encoder=encoder)
    valid_rows = extract_feature_rows(split.valid, encoder=encoder)
    test_rows = extract_feature_rows(split.test, encoder=encoder)
    _write_feature_rows(output_dir / "features_train.jsonl", train_rows)
    _write_feature_rows(output_dir / "features_valid.jsonl", valid_rows)
    _write_feature_rows(output_dir / "features_test.jsonl", test_rows)

    model = train_lightgbm_classifier(train_rows, seed=config.seed)
    scorer = calibrate_classifier(model, valid_rows)
    report = generate_training_report(
        model,
        scorer,
        valid_rows=valid_rows,
        test_rows=test_rows,
    )
    _write_json(output_dir / "report.json", report)
    _write_report_markdown(output_dir / "report.md", report)

    dataset_manifest = {
        "dataset_path": str(dataset_path),
        "dataset_hash": _file_sha256(dataset_path),
        "task_type": config.task_type,
        "sample_count": len(samples),
    }
    split_manifest = {
        "seed": config.seed,
        "train_count": len(split.train),
        "valid_count": len(split.valid),
        "test_count": len(split.test),
        "task_type": config.task_type,
    }
    _write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    _write_json(output_dir / "split_manifest.json", split_manifest)

    joblib = __import__("joblib")
    joblib.dump(scorer, output_dir / "model.joblib")

    metadata = export_scorer_artifact(
        scorer,
        config=config,
        train_count=len(split.train),
        valid_count=len(split.valid),
        test_count=len(split.test),
        dataset_manifest_hash=_json_hash(dataset_manifest),
        training_config_hash=_json_hash(_config_for_hash(config)),
    )
    metadata_path = output_dir / "metadata.json"
    write_metadata_json(metadata_path, metadata)

    return ConfidenceTrainingResult(
        output_dir=output_dir,
        export_path=export_path,
        report_path=output_dir / "report.json",
        metadata_path=metadata_path,
    )


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


def _write_report_markdown(path: Path, report: dict[str, object]) -> None:
    path.write_text(
        "# Confidence Training Report\n\n"
        f"```json\n{json.dumps(report, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(data: object) -> str:
    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_for_hash(config: ConfidenceTrainingConfig) -> dict[str, object]:
    data = asdict(config)
    data.pop("dataset_path", None)
    data.pop("output_dir", None)
    data.pop("export_path", None)
    return data
