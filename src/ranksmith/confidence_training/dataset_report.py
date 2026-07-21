from __future__ import annotations

from pathlib import Path
from typing import Any

from ranksmith.confidence import TaskType
from ranksmith.confidence_training.dataset import load_canonical_dataset
from ranksmith.confidence_training.types import CanonicalConfidenceSample

MISSING_SOURCE_KEY = "__MISSING__"


def build_dataset_report(path: str | Path, task_type: TaskType) -> dict[str, Any]:
    samples = load_canonical_dataset(path, task_type=task_type)
    positive_count = sum(sample.label for sample in samples)
    negative_count = len(samples) - positive_count
    present_sources = {sample.source for sample in samples if sample.source is not None}
    present_groups = {
        sample.group_id for sample in samples if sample.group_id is not None
    }

    sample_count = len(samples)
    missing_source_count = sum(1 for sample in samples if sample.source is None)
    missing_group_id_count = sum(1 for sample in samples if sample.group_id is None)

    return {
        "task_type": task_type,
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": _rate(positive_count, sample_count),
        "source_count": len(present_sources),
        "group_count": len(present_groups),
        "missing_source_count": missing_source_count,
        "missing_source_rate": _rate(missing_source_count, sample_count),
        "missing_group_id_count": missing_group_id_count,
        "missing_group_id_rate": _rate(missing_group_id_count, sample_count),
        "sources": _source_reports(samples),
    }


def _source_reports(
    samples: list[CanonicalConfidenceSample],
) -> dict[str, dict[str, int | float]]:
    buckets: dict[str, list[CanonicalConfidenceSample]] = {}
    for sample in samples:
        source = sample.source if sample.source is not None else MISSING_SOURCE_KEY
        buckets.setdefault(source, []).append(sample)

    return {
        source: _source_report(source_samples)
        for source, source_samples in sorted(buckets.items())
    }


def _source_report(
    samples: list[CanonicalConfidenceSample],
) -> dict[str, int | float]:
    positive_count = sum(sample.label for sample in samples)
    sample_count = len(samples)
    return {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": sample_count - positive_count,
        "positive_rate": _rate(positive_count, sample_count),
    }


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
