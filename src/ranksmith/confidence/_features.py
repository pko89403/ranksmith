from __future__ import annotations

from typing import Any

from ranksmith.confidence._dependencies import import_optional_dependency
from ranksmith.confidence._errors import ConfidenceArtifactError, ConfidenceInputError

FEATURE_SCHEMA_VERSION = "structural-v1"
FEATURE_DIM = 70
FEATURE_DTYPE = "float64"
GRANULARITY = "two_scale"
LOCAL_WINDOW_SIZE = 5
LOCAL_STRIDE = 2
MIN_MAX_LENGTH = 34
EIGENVALUE_TOLERANCE = 1e-12

_EPSILON = 1e-12
_FFT_FREQUENCIES = 16
_SHAPE_BINS = 16


def extract_structural_features(
    hidden_states: object,
    attention_mask: object,
    *,
    max_length: int,
) -> list[float]:
    np: Any = import_optional_dependency("numpy")

    if max_length < MIN_MAX_LENGTH:
        raise ConfidenceInputError(
            f"max_length must be >= {MIN_MAX_LENGTH} for {FEATURE_SCHEMA_VERSION}."
        )

    hidden = np.asarray(hidden_states, dtype=np.float64)
    mask = np.asarray(attention_mask)
    _validate_inputs(np, hidden, mask)

    token_hidden = hidden[mask.astype(bool)]
    if token_hidden.shape[0] == 0:
        raise ConfidenceInputError(
            "attention_mask must include at least one non-padding token."
        )

    _ensure_finite(np, token_hidden, "hidden_states")
    global_descriptor = _descriptor(np, token_hidden, max_length=max_length)
    local_descriptor = _local_descriptor(np, token_hidden, max_length=max_length)
    features = (global_descriptor + local_descriptor) / 2.0

    validated = _validate_feature_vector(np, features)
    return [float(value) for value in validated.tolist()]


def _validate_inputs(np: Any, hidden: Any, mask: Any) -> None:
    if hidden.ndim != 2:
        raise ConfidenceInputError("hidden_states must be a 2D array.")
    if mask.ndim != 1:
        raise ConfidenceInputError("attention_mask must be a 1D array.")
    if hidden.shape[0] != mask.shape[0]:
        raise ConfidenceInputError(
            "attention_mask length must match hidden_states rows."
        )
    _ensure_finite(np, hidden, "hidden_states")


def _descriptor(np: Any, trajectory: Any, *, max_length: int) -> Any:
    spectral = np.concatenate(
        [
            _frequency_domain_smoothness(np, trajectory, max_length=max_length),
            _graph_spectral_diffusion(np, trajectory),
        ]
    )
    local = _local_variation(np, trajectory)
    shape = _shape_coherence(np, trajectory)
    descriptor = np.concatenate([spectral, local, shape]).astype(np.float64, copy=False)
    return _validate_feature_vector(np, descriptor)


def _frequency_domain_smoothness(np: Any, trajectory: Any, *, max_length: int) -> Any:
    padded = np.zeros((max_length, trajectory.shape[1]), dtype=np.float64)
    token_count = min(trajectory.shape[0], max_length)
    padded[:token_count] = trajectory[:token_count]

    fft_values = np.fft.rfft(padded, axis=0) / float(max_length)
    powers = np.abs(fft_values[1 : _FFT_FREQUENCIES + 1]) ** 2
    if powers.shape[0] < _FFT_FREQUENCIES:
        missing = _FFT_FREQUENCIES - powers.shape[0]
        powers = np.pad(powers, ((0, missing), (0, 0)), mode="constant")

    means = np.mean(powers, axis=1)
    maxes = np.max(powers, axis=1)
    features = np.empty(_FFT_FREQUENCIES * 2, dtype=np.float64)
    features[0::2] = means
    features[1::2] = maxes
    _ensure_finite(np, features, "frequency_domain_smoothness")
    return features


def _graph_spectral_diffusion(np: Any, trajectory: Any) -> Any:
    norms = np.linalg.norm(trajectory, axis=1, keepdims=True)
    normalized = np.divide(
        trajectory,
        norms,
        out=np.zeros_like(trajectory, dtype=np.float64),
        where=norms > _EPSILON,
    )
    similarity = normalized @ normalized.T
    similarity = np.maximum(similarity, 0.0)
    np.fill_diagonal(similarity, 0.0)
    _ensure_finite(np, similarity, "similarity_graph")

    degree = similarity.sum(axis=1)
    inv_sqrt_degree = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree, dtype=np.float64),
        where=degree > _EPSILON,
    )
    normalized_adjacency = (
        similarity * inv_sqrt_degree[:, None] * inv_sqrt_degree[None, :]
    )
    laplacian = np.eye(trajectory.shape[0], dtype=np.float64) - normalized_adjacency
    _ensure_finite(np, laplacian, "normalized_laplacian")

    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = _sanitize_eigenvalues(np, eigenvalues)
    lowest = np.sort(eigenvalues)[:_FFT_FREQUENCIES]
    if lowest.shape[0] < _FFT_FREQUENCIES:
        lowest = np.pad(
            lowest,
            (0, _FFT_FREQUENCIES - lowest.shape[0]),
            mode="constant",
        )
    _ensure_finite(np, lowest, "graph_spectral_diffusion")
    return lowest.astype(np.float64, copy=False)


def _sanitize_eigenvalues(np: Any, eigenvalues: Any) -> Any:
    if not bool(np.all(np.isfinite(eigenvalues))):
        raise ConfidenceArtifactError(
            "normalized_laplacian eigenvalues contain NaN or Inf."
        )

    too_negative = eigenvalues < -EIGENVALUE_TOLERANCE
    if bool(np.any(too_negative)):
        raise ConfidenceArtifactError(
            "normalized_laplacian eigenvalues are below the negative tolerance."
        )

    return np.where(
        (eigenvalues < 0.0) & (np.abs(eigenvalues) <= EIGENVALUE_TOLERANCE),
        0.0,
        eigenvalues,
    )


def _local_variation(np: Any, trajectory: Any) -> Any:
    if trajectory.shape[0] < 2:
        return np.zeros(6, dtype=np.float64)

    deltas = np.diff(trajectory, axis=0)
    step_norms = np.linalg.norm(deltas, axis=1)
    path_length = float(np.sum(step_norms))
    start_end_distance = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
    embedding_variance = float(np.mean(np.var(trajectory, axis=0)))
    centroid_norm = float(np.linalg.norm(np.mean(trajectory, axis=0)))
    features = np.array(
        [
            path_length,
            float(np.mean(step_norms)),
            float(np.var(step_norms)),
            start_end_distance,
            embedding_variance,
            centroid_norm,
        ],
        dtype=np.float64,
    )
    _ensure_finite(np, features, "local_variation")
    return features


def _shape_coherence(np: Any, trajectory: Any) -> Any:
    if trajectory.shape[0] < 2:
        return np.zeros(_SHAPE_BINS, dtype=np.float64)

    distances = _pairwise_distances(np, trajectory)
    if distances.size == 0:
        return np.zeros(_SHAPE_BINS, dtype=np.float64)

    max_distance = float(np.max(distances))
    if max_distance <= _EPSILON:
        return np.zeros(_SHAPE_BINS, dtype=np.float64)

    normalized = distances / max_distance
    histogram, _ = np.histogram(normalized, bins=_SHAPE_BINS, range=(0.0, 1.0))
    total = float(np.sum(histogram))
    if total <= 0.0:
        return np.zeros(_SHAPE_BINS, dtype=np.float64)

    features = histogram.astype(np.float64) / total
    _ensure_finite(np, features, "shape_coherence")
    return features


def _pairwise_distances(np: Any, trajectory: Any) -> Any:
    row_count = trajectory.shape[0]
    distances = []
    for left in range(row_count - 1):
        diff = trajectory[left + 1 :] - trajectory[left]
        distances.extend(np.linalg.norm(diff, axis=1).tolist())
    return np.asarray(distances, dtype=np.float64)


def _local_descriptor(np: Any, trajectory: Any, *, max_length: int) -> Any:
    if trajectory.shape[0] < LOCAL_WINDOW_SIZE:
        return _descriptor(np, trajectory, max_length=max_length)

    descriptors = []
    for start in range(0, trajectory.shape[0] - LOCAL_WINDOW_SIZE + 1, LOCAL_STRIDE):
        window = trajectory[start : start + LOCAL_WINDOW_SIZE]
        descriptors.append(_descriptor(np, window, max_length=max_length))

    if not descriptors:
        return _descriptor(np, trajectory, max_length=max_length)

    local_mean = np.mean(np.vstack(descriptors), axis=0)
    return _validate_feature_vector(np, local_mean)


def _ensure_finite(np: Any, values: Any, name: str) -> None:
    if not bool(np.all(np.isfinite(values))):
        raise ConfidenceArtifactError(f"{name} contains NaN or Inf.")


def _validate_feature_vector(np: Any, features: Any) -> Any:
    if features.shape != (FEATURE_DIM,):
        raise ConfidenceArtifactError(
            f"{FEATURE_SCHEMA_VERSION} must produce exactly {FEATURE_DIM} features."
        )
    if features.dtype != np.dtype(FEATURE_DTYPE):
        features = features.astype(np.float64)
    _ensure_finite(np, features, "structural_features")
    return features
