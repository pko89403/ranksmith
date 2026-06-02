from ranksmith.confidence_training._errors import (
    ConfidenceDatasetError,
    ConfidenceLabelError,
    ConfidenceTrainingConfigError,
    ConfidenceTrainingError,
)
from ranksmith.confidence_training._types import (
    ConfidenceTrainingConfig,
    ConfidenceTrainingResult,
)


def train_confidence_scorer(
    config: ConfidenceTrainingConfig,
) -> ConfidenceTrainingResult:
    raise ConfidenceTrainingError("confidence training is not implemented yet")


__all__ = [
    "ConfidenceDatasetError",
    "ConfidenceLabelError",
    "ConfidenceTrainingConfig",
    "ConfidenceTrainingConfigError",
    "ConfidenceTrainingError",
    "ConfidenceTrainingResult",
    "train_confidence_scorer",
]
