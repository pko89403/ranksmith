from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ranksmith.confidence import ConfidenceDependencyError, ConfidenceInputError
from ranksmith.confidence._encoder import FrozenAutoEncoder


@dataclass
class FakeTensor:
    value: object
    detach_called: bool = False
    cpu_called: bool = False
    numpy_called: bool = False

    def detach(self) -> FakeTensor:
        self.detach_called = True
        return self

    def cpu(self) -> FakeTensor:
        self.cpu_called = True
        return self

    def numpy(self) -> object:
        self.numpy_called = True
        return self.value


class FakeNoGrad:
    def __init__(self, torch_module: FakeTorch) -> None:
        self._torch_module = torch_module

    def __enter__(self) -> None:
        self._torch_module.entered_no_grad = True
        return None

    def __exit__(self, *args: object) -> None:
        self._torch_module.exited_no_grad = True
        return None


class FakeTorch:
    def __init__(self) -> None:
        self.entered_no_grad = False
        self.exited_no_grad = False

    def no_grad(self) -> FakeNoGrad:
        return FakeNoGrad(self)


class FakeTokenizer:
    def __init__(
        self,
        *,
        preflight_ids: object,
        encoded_ids: object,
        encoded_mask: object,
    ) -> None:
        self.preflight_ids = preflight_ids
        self.encoded_ids = encoded_ids
        self.encoded_mask = encoded_mask
        self.calls: list[dict[str, object]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        self.calls.append({"text": text, **kwargs})
        if kwargs["return_tensors"] is None:
            return {"input_ids": self.preflight_ids}
        return {
            "input_ids": self.encoded_ids,
            "attention_mask": self.encoded_mask,
        }


@dataclass
class FakeOutput:
    last_hidden_state: FakeTensor


class FakeModel:
    def __init__(self, hidden: FakeTensor) -> None:
        self.hidden = hidden
        self.eval_called = False
        self.to_calls: list[str] = []
        self.calls: list[dict[str, object]] = []

    def to(self, device: str) -> FakeModel:
        self.to_calls.append(device)
        return self

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, **kwargs: object) -> FakeOutput:
        self.calls.append(kwargs)
        return FakeOutput(last_hidden_state=self.hidden)


def test_encoder_sets_eval_no_grad_and_encodes_to_python_values() -> None:
    hidden_tensor = FakeTensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask_tensor = FakeTensor([[1, 1]])
    tokenizer = FakeTokenizer(
        preflight_ids=[[101, 102]],
        encoded_ids=FakeTensor([[101, 102]]),
        encoded_mask=mask_tensor,
    )
    model = FakeModel(hidden_tensor)
    torch = FakeTorch()
    encoder = FrozenAutoEncoder(
        encoder_name="bert-base-uncased",
        encoder_revision=None,
        tokenizer_name="bert-base-uncased",
        tokenizer_revision=None,
        max_length=256,
        allow_truncation=False,
        tokenizer=tokenizer,
        model=model,
        torch_module=torch,
    )

    hidden, mask = encoder.encode("hello")

    assert model.to_calls == ["cpu"]
    assert model.eval_called is True
    assert torch.entered_no_grad is True
    assert torch.exited_no_grad is True
    assert hidden_tensor.detach_called is True
    assert hidden_tensor.cpu_called is True
    assert hidden_tensor.numpy_called is True
    assert mask_tensor.detach_called is True
    assert mask_tensor.cpu_called is True
    assert mask_tensor.numpy_called is True
    assert hidden == [[1.0, 2.0], [3.0, 4.0]]
    assert mask == [1, 1]


def test_encoder_preflight_rejects_long_input_before_forward() -> None:
    tokenizer = FakeTokenizer(
        preflight_ids=[1] * 35,
        encoded_ids=FakeTensor([[1] * 35]),
        encoded_mask=FakeTensor([[1] * 35]),
    )
    model = FakeModel(FakeTensor([[[1.0]] * 35]))
    encoder = FrozenAutoEncoder(
        encoder_name="bert-base-uncased",
        encoder_revision=None,
        tokenizer_name="bert-base-uncased",
        tokenizer_revision=None,
        max_length=34,
        allow_truncation=False,
        tokenizer=tokenizer,
        model=model,
        torch_module=FakeTorch(),
    )

    with pytest.raises(ConfidenceInputError, match="exceeds max_length"):
        encoder.encode("too long")

    assert model.calls == []


def test_encoder_allows_explicit_truncation_without_preflight() -> None:
    tokenizer = FakeTokenizer(
        preflight_ids=[1] * 100,
        encoded_ids=FakeTensor([[1] * 34]),
        encoded_mask=FakeTensor([[1] * 34]),
    )
    model = FakeModel(FakeTensor([[[1.0]] * 34]))
    encoder = FrozenAutoEncoder(
        encoder_name="bert-base-uncased",
        encoder_revision=None,
        tokenizer_name="bert-base-uncased",
        tokenizer_revision=None,
        max_length=34,
        allow_truncation=True,
        tokenizer=tokenizer,
        model=model,
        torch_module=FakeTorch(),
    )

    _, mask = encoder.encode("truncate")

    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0]["truncation"] is True
    assert tokenizer.calls[0]["max_length"] == 34
    assert len(mask) == 34


def test_encoder_rejects_empty_or_non_string_text() -> None:
    encoder = FrozenAutoEncoder(
        encoder_name="bert-base-uncased",
        encoder_revision=None,
        tokenizer_name="bert-base-uncased",
        tokenizer_revision=None,
        max_length=34,
        allow_truncation=False,
        tokenizer=FakeTokenizer(
            preflight_ids=[1],
            encoded_ids=FakeTensor([[1]]),
            encoded_mask=FakeTensor([[1]]),
        ),
        model=FakeModel(FakeTensor([[[1.0]]])),
        torch_module=FakeTorch(),
    )

    with pytest.raises(ConfidenceInputError, match="must not be empty"):
        encoder.encode(" ")
    with pytest.raises(ConfidenceInputError, match="must be a string"):
        encoder.encode(123)  # type: ignore[arg-type]


def test_from_pretrained_uses_lazy_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(name: str, *, extra: str = "confidence") -> Any:
        raise ConfidenceDependencyError(f"missing {name} {extra}")

    monkeypatch.setattr(
        "ranksmith.confidence._encoder.import_optional_dependency",
        missing_dependency,
    )

    with pytest.raises(ConfidenceDependencyError, match="missing transformers"):
        FrozenAutoEncoder.from_pretrained(max_length=34)


def test_from_pretrained_sanitizes_hf_loader_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "hf_secret_123"

    class BrokenAutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            raise RuntimeError(f"bad token {secret}")

    class FakeTransformers:
        AutoTokenizer = BrokenAutoTokenizer
        AutoModel = object()

    def fake_import(name: str, *, extra: str = "confidence") -> object:
        if name == "transformers":
            return FakeTransformers()
        return FakeTorch()

    monkeypatch.setattr(
        "ranksmith.confidence._encoder.import_optional_dependency",
        fake_import,
    )

    with pytest.raises(ConfidenceDependencyError) as exc_info:
        FrozenAutoEncoder.from_pretrained(hf_token=secret, max_length=34)

    assert secret not in str(exc_info.value)
    assert secret not in repr(exc_info.value)
