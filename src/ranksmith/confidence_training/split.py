from __future__ import annotations

import random
from collections import defaultdict

from ranksmith.confidence_training.errors import ConfidenceDatasetError
from ranksmith.confidence_training.types import (
    CanonicalConfidenceSample,
    ConfidenceDatasetSplit,
)

MIN_TOTAL_SAMPLES = 30
MIN_SPLIT_SAMPLES = 2


def split_dataset(
    samples: list[CanonicalConfidenceSample],
    *,
    seed: int,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> ConfidenceDatasetSplit:
    _validate_ratios(train_ratio, valid_ratio, test_ratio)
    _validate_sample_count(samples)
    if any(sample.group_id is not None for sample in samples):
        split = _split_grouped(
            samples,
            seed=seed,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )
    else:
        split = _split_stratified(
            samples,
            seed=seed,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )
    _validate_split(split)
    return split


def _split_stratified(
    samples: list[CanonicalConfidenceSample],
    *,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> ConfidenceDatasetSplit:
    rng = random.Random(seed)
    by_label: dict[int, list[CanonicalConfidenceSample]] = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)

    train: list[CanonicalConfidenceSample] = []
    valid: list[CanonicalConfidenceSample] = []
    test: list[CanonicalConfidenceSample] = []
    for label_samples in by_label.values():
        shuffled = list(label_samples)
        rng.shuffle(shuffled)
        train_count, valid_count = _target_counts(
            len(shuffled),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )
        train.extend(shuffled[:train_count])
        valid.extend(shuffled[train_count : train_count + valid_count])
        test.extend(shuffled[train_count + valid_count :])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return ConfidenceDatasetSplit(tuple(train), tuple(valid), tuple(test))


def _split_grouped(
    samples: list[CanonicalConfidenceSample],
    *,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> ConfidenceDatasetSplit:
    rng = random.Random(seed)
    groups_by_id: dict[str, list[CanonicalConfidenceSample]] = defaultdict(list)
    for sample in samples:
        group_key = sample.group_id if sample.group_id is not None else sample.id
        groups_by_id[group_key].append(sample)

    groups = list(groups_by_id.values())
    rng.shuffle(groups)
    train_target, valid_target = _target_counts(
        len(samples),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )
    train_groups: list[list[CanonicalConfidenceSample]] = []
    valid_groups: list[list[CanonicalConfidenceSample]] = []
    test_groups: list[list[CanonicalConfidenceSample]] = []
    train_size = 0
    valid_size = 0

    for group in groups:
        if train_size + len(group) <= train_target:
            train_groups.append(group)
            train_size += len(group)
        elif valid_size + len(group) <= valid_target:
            valid_groups.append(group)
            valid_size += len(group)
        else:
            test_groups.append(group)

    return ConfidenceDatasetSplit(
        tuple(sample for group in train_groups for sample in group),
        tuple(sample for group in valid_groups for sample in group),
        tuple(sample for group in test_groups for sample in group),
    )


def _target_counts(
    total: int,
    *,
    train_ratio: float,
    valid_ratio: float,
) -> tuple[int, int]:
    train_count = round(total * train_ratio)
    valid_count = round(total * valid_ratio)
    return train_count, valid_count


def _validate_ratios(
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> None:
    if train_ratio <= 0 or valid_ratio <= 0 or test_ratio <= 0:
        raise ConfidenceDatasetError("split ratios must be positive")
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-9:
        raise ConfidenceDatasetError("split ratios must sum to 1.0")


def _validate_sample_count(samples: list[CanonicalConfidenceSample]) -> None:
    if len(samples) < MIN_TOTAL_SAMPLES:
        raise ConfidenceDatasetError("dataset must contain at least 30 samples")


def _validate_split(split: ConfidenceDatasetSplit) -> None:
    for name, part in (
        ("train", split.train),
        ("valid", split.valid),
        ("test", split.test),
    ):
        if len(part) < MIN_SPLIT_SAMPLES:
            raise ConfidenceDatasetError(f"{name} split has too few samples")
        labels = {sample.label for sample in part}
        if labels != {0, 1}:
            raise ConfidenceDatasetError(
                f"{name} split must contain positive and negative labels"
            )
