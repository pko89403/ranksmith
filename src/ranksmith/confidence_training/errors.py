class ConfidenceTrainingError(Exception):
    """Base error for confidence training."""


class ConfidenceDatasetError(ConfidenceTrainingError):
    """Raised when a confidence training dataset is invalid."""


class ConfidenceLabelError(ConfidenceTrainingError):
    """Raised when confidence training labels are invalid."""


class ConfidenceTrainingConfigError(ConfidenceTrainingError):
    """Raised when confidence training configuration is invalid."""
