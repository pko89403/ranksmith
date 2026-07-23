from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ranksmith.confidence_generation.io import load_answer_generation_samples

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_answer_confidence_training_data",
    ROOT / "scripts" / "build_answer_confidence_training_data.py",
)
assert SPEC is not None
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
# Dataclass processing looks the module up in sys.modules by name.
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _squad_payload() -> dict[str, object]:
    return {
        "version": "1.1",
        "data": [
            {
                "title": "Scurvy",
                "paragraphs": [
                    {
                        "context": (
                            "Scurvy is a disease resulting from a lack of "
                            "vitamin C. Early symptoms include weakness and "
                            "gum disease."
                        ),
                        "qas": [
                            {
                                "id": "q-scurvy-1",
                                "question": "What deficiency causes scurvy?",
                                "answers": [
                                    {"text": "a lack of vitamin C"},
                                    {"text": "a lack of vitamin C"},
                                    {"text": "vitamin C"},
                                ],
                            }
                        ],
                    },
                    {
                        "context": (
                            "Treatment of scurvy consists of vitamin C "
                            "supplementation and a diet rich in fresh fruit."
                        ),
                        "qas": [
                            {
                                "id": "q-scurvy-2",
                                "question": "How is scurvy treated?",
                                "answers": [{"text": "vitamin C supplementation"}],
                            }
                        ],
                    },
                ],
            },
            {
                "title": "Sleep",
                "paragraphs": [
                    {
                        "context": (
                            "Sleep deprivation weakens the immune system and "
                            "impairs memory consolidation in humans."
                        ),
                        "qas": [
                            {
                                "id": "q-sleep-1",
                                "question": "What does sleep deprivation weaken?",
                                "answers": [{"text": "the immune system"}],
                            }
                        ],
                    },
                    {
                        "context": (
                            "Good sleep hygiene includes regular schedules "
                            "and avoiding caffeine before bed."
                        ),
                        "qas": [],
                    },
                ],
            },
        ],
    }


def _write_squad(path: Path) -> Path:
    path.write_text(json.dumps(_squad_payload()), encoding="utf-8")
    return path


def _run_builder(tmp_path: Path, *extra: str) -> Path:
    squad_path = _write_squad(tmp_path / "train-v1.1.json")
    output = tmp_path / "answer_train.jsonl"
    argv = [
        "build_answer_confidence_training_data.py",
        "--squad-train",
        str(squad_path),
        "--output",
        str(output),
        "--questions",
        "3",
        "--negatives-per-question",
        "1",
        "--negative-pool",
        "2",
        *extra,
    ]
    original = sys.argv
    sys.argv = argv
    try:
        builder.main()
    finally:
        sys.argv = original
    return output


def test_builder_output_loads_with_the_real_generation_loader(
    tmp_path: Path,
) -> None:
    output = _run_builder(tmp_path)

    samples = load_answer_generation_samples(output, max_context_chars=4000)

    assert len(samples) == 6  # 3 questions x (1 gold + 1 hard negative)
    by_id = {sample.id: sample for sample in samples}
    gold = by_id["q-scurvy-1::gold"]
    negative = by_id["q-scurvy-1::neg1"]
    assert gold.query == negative.query
    assert gold.gold_answer == negative.gold_answer
    assert gold.gold_answer == ["a lack of vitamin C", "vitamin C"]
    assert gold.context.startswith("Scurvy\n\n")
    assert gold.context != negative.context
    assert gold.metadata["context_kind"] == "gold"
    assert negative.metadata["context_kind"] == "bm25_hard_negative"
    assert gold.group_id == "Scurvy"


def test_builder_hard_negative_is_the_top_non_gold_bm25_paragraph(
    tmp_path: Path,
) -> None:
    output = _run_builder(tmp_path)

    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    negative = next(row for row in rows if row["id"] == "q-scurvy-1::neg1")

    # The other Scurvy paragraph shares "scurvy"/"vitamin" terms, so it must
    # beat the Sleep paragraph as the hard negative.
    assert negative["context"].startswith("Scurvy\n\n")
    assert "Treatment of scurvy" in negative["context"]


def test_builder_is_deterministic(tmp_path: Path) -> None:
    first = _run_builder(tmp_path).read_bytes()
    second = _run_builder(tmp_path, "--overwrite").read_bytes()

    assert first == second


def test_builder_writes_a_report(tmp_path: Path) -> None:
    output = _run_builder(tmp_path)

    report = json.loads(
        (tmp_path / "answer_train.report.json").read_text(encoding="utf-8")
    )

    assert report["source"] == "squad-v1.1-train"
    assert report["statistics"]["total_rows"] == 6
    assert report["statistics"]["gold_rows"] == 3
    assert report["statistics"]["negative_rows"] == 3
    assert report["parameters"]["questions"] == 3
    assert len(report["squad_sha256"]) == 64
    assert output.exists()


def test_builder_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    _run_builder(tmp_path)

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        _run_builder(tmp_path)


def test_builder_fails_when_not_enough_questions_qualify(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Only .* usable"):
        _run_builder(tmp_path, "--overwrite", "--questions", "4")


def test_builder_rejects_squad_v2(tmp_path: Path) -> None:
    payload = _squad_payload()
    payload["version"] = "v2.0"
    squad_path = tmp_path / "train-v2.0.json"
    squad_path.write_text(json.dumps(payload), encoding="utf-8")
    original = sys.argv
    sys.argv = [
        "build_answer_confidence_training_data.py",
        "--squad-train",
        str(squad_path),
        "--output",
        str(tmp_path / "out.jsonl"),
    ]
    try:
        with pytest.raises(SystemExit, match="version 1.1"):
            builder.main()
    finally:
        sys.argv = original


def test_builder_requires_an_input_source(tmp_path: Path) -> None:
    original = sys.argv
    sys.argv = [
        "build_answer_confidence_training_data.py",
        "--output",
        str(tmp_path / "out.jsonl"),
    ]
    try:
        with pytest.raises(SystemExit, match="--squad-train <path> or --download"):
            builder.main()
    finally:
        sys.argv = original
