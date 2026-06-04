from __future__ import annotations

import os
from dataclasses import FrozenInstanceError, dataclass

import pytest

from ranksmith.confidence import ConfidenceInputError
from ranksmith.confidence._encoder import (
    HF_LIVE_TEST_ENV,
    FrozenAutoEncoder,
    build_hf_from_pretrained_kwargs,
    validate_device,
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


def test_local_files_only_and_cache_dir_are_forwarded() -> None:
    kwargs = build_hf_from_pretrained_kwargs(
        revision=None,
        hf_token=None,
        local_files_only=True,
        cache_dir="/tmp/hf-cache",
    )

    assert kwargs["local_files_only"] is True
    assert kwargs["cache_dir"] == "/tmp/hf-cache"
    assert "token" not in kwargs


def test_only_cpu_device_is_supported() -> None:
    validate_device("cpu")

    with pytest.raises(ConfidenceInputError):
        validate_device("mps")


def test_hf_live_tests_are_opt_in() -> None:
    assert HF_LIVE_TEST_ENV == "RANKSMITH_RUN_HF_TESTS"
    assert os.environ.get(HF_LIVE_TEST_ENV) is None


def test_encoder_stores_hf_options_as_read_only_attributes() -> None:
    encoder = FrozenAutoEncoder(
        encoder_name="encoder",
        encoder_revision="model-rev",
        tokenizer_name="tokenizer",
        tokenizer_revision="tok-rev",
        max_length=34,
        allow_truncation=True,
        local_files_only=True,
        cache_dir="/tmp/hf-cache",
        device="cpu",
        tokenizer=FakeTokenizer(
            preflight_ids=[1],
            encoded_ids=FakeTensor([[1]]),
            encoded_mask=FakeTensor([[1]]),
        ),
        model=FakeModel(FakeTensor([[[1.0]]])),
        torch_module=FakeTorch(),
    )

    assert encoder.encoder_name == "encoder"
    assert encoder.encoder_revision == "model-rev"
    assert encoder.tokenizer_name == "tokenizer"
    assert encoder.tokenizer_revision == "tok-rev"
    assert encoder.local_files_only is True
    assert encoder.cache_dir == "/tmp/hf-cache"
    with pytest.raises(FrozenInstanceError):
        encoder.cache_dir = "changed"  # type: ignore[misc]


def test_max_length_respects_structural_minimum() -> None:
    with pytest.raises(ConfidenceInputError, match="max_length"):
        FrozenAutoEncoder(
            encoder_name="encoder",
            encoder_revision=None,
            tokenizer_name="tokenizer",
            tokenizer_revision=None,
            max_length=33,
            allow_truncation=False,
            tokenizer=FakeTokenizer(
                preflight_ids=[1],
                encoded_ids=FakeTensor([[1]]),
                encoded_mask=FakeTensor([[1]]),
            ),
            model=FakeModel(FakeTensor([[[1.0]]])),
            torch_module=FakeTorch(),
        )
