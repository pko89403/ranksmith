from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ranksmith.confidence.dependencies import import_optional_dependency
from ranksmith.confidence.errors import (
    ConfidenceDependencyError,
    ConfidenceInputError,
)
from ranksmith.confidence.features import MIN_MAX_LENGTH

HF_LIVE_TEST_ENV = "RANKSMITH_RUN_HF_TESTS"


def validate_device(device: str) -> None:
    if device != "cpu":
        raise ConfidenceInputError('Phase 1 supports only device="cpu".')


def build_hf_from_pretrained_kwargs(
    *,
    revision: str | None,
    hf_token: str | None,
    local_files_only: bool,
    cache_dir: str | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"local_files_only": local_files_only}
    if revision is not None:
        kwargs["revision"] = revision
    if hf_token is not None:
        kwargs["token"] = hf_token
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return kwargs


@dataclass(frozen=True, kw_only=True)
class FrozenAutoEncoder:
    encoder_name: str
    encoder_revision: str | None
    tokenizer_name: str
    tokenizer_revision: str | None
    max_length: int
    allow_truncation: bool
    local_files_only: bool = False
    cache_dir: str | None = None
    device: str = "cpu"
    tokenizer: Any = field(repr=False)
    model: Any = field(repr=False)
    torch_module: Any = field(repr=False)

    def __post_init__(self) -> None:
        validate_device(self.device)
        _validate_max_length(self.max_length)
        if hasattr(self.model, "to"):
            self.model.to("cpu")
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        *,
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
    ) -> FrozenAutoEncoder:
        validate_device(device)
        _validate_max_length(max_length)
        transformers = import_optional_dependency("transformers")
        torch = import_optional_dependency("torch")

        resolved_tokenizer_name = tokenizer_name or encoder_name
        tokenizer_kwargs = build_hf_from_pretrained_kwargs(
            revision=tokenizer_revision,
            hf_token=hf_token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        model_kwargs = build_hf_from_pretrained_kwargs(
            revision=encoder_revision,
            hf_token=hf_token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                resolved_tokenizer_name,
                **tokenizer_kwargs,
            )
            model = transformers.AutoModel.from_pretrained(
                encoder_name,
                **model_kwargs,
            )
        except Exception:
            raise ConfidenceDependencyError(
                "Failed to load HuggingFace tokenizer or model."
            ) from None

        return cls(
            encoder_name=encoder_name,
            encoder_revision=encoder_revision,
            tokenizer_name=resolved_tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            max_length=max_length,
            allow_truncation=allow_truncation,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            device=device,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
        )

    def encode(self, text: str) -> tuple[list[list[float]], list[int]]:
        _validate_text(text)
        if not self.allow_truncation:
            preflight_tokens = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            token_count = _token_count(_mapping_get(preflight_tokens, "input_ids"))
            if token_count > self.max_length:
                raise ConfidenceInputError(
                    f"input token length {token_count} exceeds max_length "
                    f"{self.max_length}."
                )

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=self.allow_truncation,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        with self.torch_module.no_grad():
            output = self.model(**encoded)

        hidden = _as_python(output.last_hidden_state)
        mask = _as_python(_mapping_get(encoded, "attention_mask"))
        return _to_hidden_rows(hidden), _to_mask_values(mask)


def _validate_max_length(max_length: int) -> None:
    if max_length < MIN_MAX_LENGTH:
        raise ConfidenceInputError(
            f"max_length must be >= {MIN_MAX_LENGTH} for structural-v1."
        )


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise ConfidenceInputError("text must be a string.")
    if not text.strip():
        raise ConfidenceInputError("text must not be empty.")


def _mapping_get(values: object, key: str) -> object:
    if not isinstance(values, Mapping) or key not in values:
        raise ConfidenceInputError(f"tokenizer output must include {key!r}.")
    return values[key]


def _token_count(input_ids: object) -> int:
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        dimensions = [int(value) for value in shape]
        if len(dimensions) == 1:
            return dimensions[0]
        if len(dimensions) >= 2:
            return dimensions[-1]

    if isinstance(input_ids, Sequence) and not isinstance(input_ids, (str, bytes)):
        if not input_ids:
            return 0
        first = input_ids[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            return len(first)
        return len(input_ids)

    raise ConfidenceInputError("tokenizer input_ids must be a sequence.")


def _as_python(value: object) -> object:
    current = value
    for method_name in ("detach", "cpu", "numpy"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        return tolist()
    return current


def _to_hidden_rows(value: object) -> list[list[float]]:
    rows = _first_hidden_batch(value)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ConfidenceInputError("encoder hidden states must be a 2D sequence.")

    converted: list[list[float]] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ConfidenceInputError("encoder hidden states must be a 2D sequence.")
        converted.append([float(item) for item in row])

    if not converted:
        raise ConfidenceInputError(
            "encoder hidden states must include at least one row."
        )
    return converted


def _to_mask_values(value: object) -> list[int]:
    values = _first_mask_batch(value)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ConfidenceInputError("attention_mask must be a 1D sequence.")

    converted = [int(item) for item in values]
    if not converted:
        raise ConfidenceInputError("attention_mask must include at least one value.")
    return converted


def _first_hidden_batch(value: object) -> object:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return value
    if not value:
        return value
    first = value[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
        return value
    if not first:
        return value
    first_inner = first[0]
    if isinstance(first_inner, Sequence) and not isinstance(first_inner, (str, bytes)):
        return first
    return value


def _first_mask_batch(value: object) -> object:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return value
    if not value:
        return value
    first = value[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        return first
    return value
