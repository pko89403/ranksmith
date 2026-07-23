from __future__ import annotations

import importlib.util
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path("scripts/train_confidence_scorer.py")


@dataclass(frozen=True)
class FakeTrainingResult:
    output_dir: Path
    export_path: Path
    report_path: Path
    metadata_path: Path


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "train_confidence_scorer",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_confidence_scorer_help_exits_successfully() -> None:
    result = subprocess.run(
        ["uv", "run", "python", str(SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--task" in result.stdout


def test_training_cli_builds_config_and_prints_result_paths(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    script = _load_script()
    calls: list[Any] = []

    def fake_train(config: Any) -> FakeTrainingResult:
        calls.append(config)
        return FakeTrainingResult(
            output_dir=Path(config.output_dir),
            export_path=Path(config.export_path),
            report_path=Path(config.output_dir) / "report.json",
            metadata_path=Path(config.output_dir) / "metadata.json",
        )

    monkeypatch.setattr(script, "train_confidence_scorer", fake_train)

    status = script.main(
        [
            "--task",
            "query_context_answerability_confidence",
            "--dataset",
            str(tmp_path / "dataset.jsonl"),
            "--output-dir",
            str(tmp_path / "training"),
            "--export-path",
            str(tmp_path / "artifact.joblib"),
            "--encoder-name",
            "encoder",
            "--encoder-revision",
            "rev1",
            "--tokenizer-name",
            "tokenizer",
            "--tokenizer-revision",
            "rev2",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--local-files-only",
            "--max-length",
            "384",
            "--allow-truncation",
            "--seed",
            "7",
            "--train-ratio",
            "0.6",
            "--valid-ratio",
            "0.2",
            "--test-ratio",
            "0.2",
            "--calibration-method",
            "sigmoid",
        ]
    )

    assert status == 0
    config = calls[0]
    assert config.task_type == "query_context_answerability_confidence"
    assert config.dataset_path == tmp_path / "dataset.jsonl"
    assert config.output_dir == tmp_path / "training"
    assert config.export_path == tmp_path / "artifact.joblib"
    assert config.encoder_name == "encoder"
    assert config.encoder_revision == "rev1"
    assert config.tokenizer_name == "tokenizer"
    assert config.tokenizer_revision == "rev2"
    assert config.cache_dir == str(tmp_path / "cache")
    assert config.local_files_only is True
    assert config.max_length == 384
    assert config.allow_truncation is True
    assert config.seed == 7
    assert config.train_ratio == 0.6
    assert config.valid_ratio == 0.2
    assert config.test_ratio == 0.2
    assert config.calibration_method == "sigmoid"
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "output_dir": str(tmp_path / "training"),
        "export_path": str(tmp_path / "artifact.joblib"),
        "report_path": str(tmp_path / "training" / "report.json"),
        "metadata_path": str(tmp_path / "training" / "metadata.json"),
    }
