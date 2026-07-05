from __future__ import annotations

import pytest

from ranksmith.confidence_training import ConfidenceDatasetError
from ranksmith.confidence_training.split import split_dataset
from ranksmith.confidence_training.types import CanonicalConfidenceSample


def _samples(count: int = 40) -> list[CanonicalConfidenceSample]:
    return [
        CanonicalConfidenceSample(
            id=f"s{i}",
            task_type="answer_confidence",
            label=i % 2,
            context=f"context {i}",
            answer=f"answer {i}",
        )
        for i in range(count)
    ]


def test_split_dataset_is_deterministic() -> None:
    samples = _samples()

    first = split_dataset(samples, seed=7)
    second = split_dataset(samples, seed=7)

    assert [sample.id for sample in first.train] == [
        sample.id for sample in second.train
    ]
    assert [sample.id for sample in first.valid] == [
        sample.id for sample in second.valid
    ]
    assert [sample.id for sample in first.test] == [sample.id for sample in second.test]
    assert len(first.train) == 32
    assert len(first.valid) == 4
    assert len(first.test) == 4


def test_split_dataset_changes_with_seed() -> None:
    samples = _samples()

    first = split_dataset(samples, seed=7)
    second = split_dataset(samples, seed=8)

    assert [sample.id for sample in first.train] != [
        sample.id for sample in second.train
    ]


def test_split_preserves_group_id_boundaries() -> None:
    samples = [
        CanonicalConfidenceSample(
            id=f"s{i}",
            task_type="answer_confidence",
            label=i % 2,
            context=f"context {i}",
            answer=f"answer {i}",
            group_id=f"g{i // 2}",
        )
        for i in range(40)
    ]

    split = split_dataset(samples, seed=7)
    partitions = {
        sample.id: name
        for name, part in (
            ("train", split.train),
            ("valid", split.valid),
            ("test", split.test),
        )
        for sample in part
    }

    for i in range(0, 40, 2):
        assert partitions[f"s{i}"] == partitions[f"s{i + 1}"]


def test_split_fails_when_dataset_is_too_small() -> None:
    with pytest.raises(ConfidenceDatasetError, match="at least 30 samples"):
        split_dataset(_samples(10), seed=7)


def test_split_fails_when_a_split_loses_a_class() -> None:
    samples = [
        CanonicalConfidenceSample(
            id=f"s{i}",
            task_type="answer_confidence",
            label=1 if i == 39 else 0,
            context=f"context {i}",
            answer=f"answer {i}",
        )
        for i in range(40)
    ]

    with pytest.raises(ConfidenceDatasetError, match="positive and negative"):
        split_dataset(samples, seed=7)


def test_split_fails_for_invalid_ratios() -> None:
    with pytest.raises(ConfidenceDatasetError, match="ratios must sum to 1.0"):
        split_dataset(
            _samples(),
            seed=7,
            train_ratio=0.7,
            valid_ratio=0.2,
            test_ratio=0.2,
        )
