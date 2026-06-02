from __future__ import annotations

from dataclasses import dataclass, fields

import pytest

from ranksmith.confidence import ConfidenceDependencyError
from ranksmith.confidence._encoder import (
    FrozenAutoEncoder,
    build_hf_from_pretrained_kwargs,
)


@dataclass
class FakeTensor:
    value: object

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> object:
        return self.value


class FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class FakeTorch:
    def no_grad(self) -> FakeNoGrad:
        return FakeNoGrad()


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

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        if kwargs["return_tensors"] is None:
            return {"input_ids": self.preflight_ids}
        return {
            "input_ids": self.encoded_ids,
            "attention_mask": self.encoded_mask,
        }


class FakeModel:
    def __init__(self, hidden: FakeTensor) -> None:
        self.hidden = hidden

    def to(self, device: str) -> FakeModel:
        return self

    def eval(self) -> None:
        return None

    def __call__(self, **kwargs: object) -> object:
        return type("FakeOutput", (), {"last_hidden_state": self.hidden})()


def test_hf_token_is_forwarded_but_not_returned_as_metadata() -> None:
    kwargs = build_hf_from_pretrained_kwargs(
        revision="main",
        hf_token="secret-token",
        local_files_only=False,
        cache_dir=None,
    )

    assert kwargs["token"] == "secret-token"
    assert "secret-token" not in repr({k: v for k, v in kwargs.items() if k != "token"})


def test_hf_token_is_not_a_frozen_encoder_field_or_repr() -> None:
    secret = "hf_secret_123"
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

    assert "hf_token" not in {field.name for field in fields(FrozenAutoEncoder)}
    assert secret not in repr(encoder)


def test_hf_token_is_passed_to_tokenizer_and_model_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "hf_secret_123"
    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs: object) -> FakeTokenizer:
            calls.append(("tokenizer", name, kwargs))
            return FakeTokenizer(
                preflight_ids=[1],
                encoded_ids=FakeTensor([[1]]),
                encoded_mask=FakeTensor([[1]]),
            )

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name: str, **kwargs: object) -> FakeModel:
            calls.append(("model", name, kwargs))
            return FakeModel(FakeTensor([[[1.0]]]))

    class FakeTransformers:
        AutoTokenizer = FakeAutoTokenizer
        AutoModel = FakeAutoModel

    def fake_import(name: str, *, extra: str = "confidence") -> object:
        if name == "transformers":
            return FakeTransformers()
        return FakeTorch()

    monkeypatch.setattr(
        "ranksmith.confidence._encoder.import_optional_dependency",
        fake_import,
    )

    encoder = FrozenAutoEncoder.from_pretrained(
        encoder_name="encoder",
        encoder_revision="model-rev",
        tokenizer_name=None,
        tokenizer_revision="tok-rev",
        hf_token=secret,
        local_files_only=True,
        cache_dir="/tmp/hf-cache",
        device="cpu",
        max_length=34,
        allow_truncation=False,
    )

    assert encoder.tokenizer_name == "encoder"
    assert calls[0][2]["token"] == secret
    assert calls[1][2]["token"] == secret
    assert calls[0][2]["revision"] == "tok-rev"
    assert calls[1][2]["revision"] == "model-rev"
    assert secret not in repr(encoder)


def test_hf_token_is_not_exposed_when_model_loader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "hf_secret_123"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs: object) -> FakeTokenizer:
            return FakeTokenizer(
                preflight_ids=[1],
                encoded_ids=FakeTensor([[1]]),
                encoded_mask=FakeTensor([[1]]),
            )

    class BrokenAutoModel:
        @staticmethod
        def from_pretrained(name: str, **kwargs: object) -> object:
            raise RuntimeError(f"token={secret}")

    class FakeTransformers:
        AutoTokenizer = FakeAutoTokenizer
        AutoModel = BrokenAutoModel

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
