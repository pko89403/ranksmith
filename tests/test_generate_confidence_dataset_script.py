from __future__ import annotations

import importlib.util
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path("scripts/generate_confidence_dataset.py")


@dataclass(frozen=True)
class FakeGenerationResult:
    output_path: Path
    input_count: int = 2
    generated_count: int = 2
    skipped_count: int = 0
    positive_count: int = 1
    negative_count: int = 1


class FakeLMStudioProvider:
    instances: list[FakeLMStudioProvider] = []

    def __init__(
        self,
        *,
        base_url: str | None,
        model: str | None,
        api_key: str | None,
        max_tokens: int,
        timeout: float | None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.instances.append(self)

    def complete(self, request: object) -> object:
        raise AssertionError("live provider must not be called in CLI tests")


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "generate_confidence_dataset",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_confidence_dataset_help_exits_successfully() -> None:
    result = subprocess.run(
        ["uv", "run", "python", str(SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--task" in result.stdout


def test_query_answerability_cli_builds_lmstudio_provider_and_config(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    script = _load_script()
    calls: list[Any] = []

    def fake_generate(config: Any) -> FakeGenerationResult:
        calls.append(config)
        return FakeGenerationResult(output_path=Path(config.output_path))

    monkeypatch.setattr(script, "LMStudioModelProvider", FakeLMStudioProvider)
    monkeypatch.setattr(
        script,
        "generate_query_answerability_confidence_dataset",
        fake_generate,
    )

    status = script.main(
        [
            "--task",
            "query_answerability_confidence",
            "--provider",
            "lmstudio",
            "--lmstudio-base-url",
            "http://localhost:1234/v1",
            "--lmstudio-model",
            "local-model",
            "--lmstudio-api-key",
            "key",
            "--timeout",
            "15",
            "--input",
            str(tmp_path / "raw.jsonl"),
            "--output",
            str(tmp_path / "canonical.jsonl"),
            "--overwrite",
            "--max-items",
            "3",
            "--source",
            "unit",
        ]
    )

    assert status == 0
    provider = FakeLMStudioProvider.instances[-1]
    assert provider.base_url == "http://localhost:1234/v1"
    assert provider.model == "local-model"
    assert provider.api_key == "key"
    assert provider.max_tokens == 128
    assert provider.timeout == 15
    config = calls[0]
    assert config.overwrite is True
    assert config.resume is False
    assert config.max_items == 3
    assert config.source == "unit"
    summary = json.loads(capsys.readouterr().out)
    assert summary["output_path"] == str(tmp_path / "canonical.jsonl")
    assert summary["generated_count"] == 2


def test_query_context_cli_passes_max_context_chars_and_resume(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    script = _load_script()
    calls: list[Any] = []

    def fake_generate(config: Any) -> FakeGenerationResult:
        calls.append(config)
        return FakeGenerationResult(output_path=Path(config.output_path))

    monkeypatch.setattr(script, "LMStudioModelProvider", FakeLMStudioProvider)
    monkeypatch.setattr(
        script,
        "generate_query_context_answerability_confidence_dataset",
        fake_generate,
    )

    status = script.main(
        [
            "--task",
            "query_context_answerability_confidence",
            "--provider",
            "lmstudio",
            "--lmstudio-model",
            "local-model",
            "--lmstudio-max-tokens",
            "64",
            "--input",
            str(tmp_path / "raw.jsonl"),
            "--output",
            str(tmp_path / "canonical.jsonl"),
            "--resume",
            "--max-context-chars",
            "1234",
        ]
    )

    assert status == 0
    assert FakeLMStudioProvider.instances[-1].max_tokens == 64
    config = calls[0]
    assert config.resume is True
    assert config.max_context_chars == 1234
    assert json.loads(capsys.readouterr().out)["output_path"] == str(
        tmp_path / "canonical.jsonl"
    )


def test_query_context_cli_defaults_max_context_chars(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    script = _load_script()
    calls: list[Any] = []

    def fake_generate(config: Any) -> FakeGenerationResult:
        calls.append(config)
        return FakeGenerationResult(output_path=Path(config.output_path))

    monkeypatch.setattr(script, "LMStudioModelProvider", FakeLMStudioProvider)
    monkeypatch.setattr(
        script,
        "generate_query_context_answerability_confidence_dataset",
        fake_generate,
    )

    status = script.main(
        [
            "--task",
            "query_context_answerability_confidence",
            "--provider",
            "lmstudio",
            "--input",
            str(tmp_path / "raw.jsonl"),
            "--output",
            str(tmp_path / "canonical.jsonl"),
        ]
    )

    assert status == 0
    assert calls[0].max_context_chars == 8000


def test_query_answerability_cli_rejects_max_context_chars(tmp_path: Path) -> None:
    script = _load_script()

    with pytest.raises(SystemExit) as exc_info:
        script._parse_args(
            [
                "--task",
                "query_answerability_confidence",
                "--provider",
                "lmstudio",
                "--input",
                str(tmp_path / "raw.jsonl"),
                "--output",
                str(tmp_path / "canonical.jsonl"),
                "--max-context-chars",
                "1234",
            ]
        )

    assert exc_info.value.code != 0
