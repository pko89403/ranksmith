from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ranksmith._providers import _build_selection_prompt
from ranksmith.types import Document

ROOT = Path(__file__).resolve().parents[1]


def test_pairwise_prp_example_runs_without_live_provider() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "pairwise_prp.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Pairwise PRP-Sliding-K example" in result.stdout
    assert "rank=01 id=vitamin_c" in result.stdout
    assert "rank=02 id=vitamin_b12" in result.stdout
    assert "compare_calls=18" in result.stdout


def test_custom_strategy_example_runs_without_live_provider() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "custom_strategy.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Custom Strategy example" in result.stdout
    assert "length_strategy rank=01 id=vitamin_c" in result.stdout
    assert "provider_strategy rank=01 id=vitamin_b12" in result.stdout
    assert "provider_error=provider timeout" in result.stdout


def test_tourrank_example_runs_without_live_provider() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "tourrank.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "TourRank-r example" in result.stdout
    assert "tourrank rank=01 id=vitamin_c" in result.stdout
    assert "tourrank rank=02 id=vitamin_b12" in result.stdout
    assert "select_calls=4" in result.stdout


def test_selection_prompt_example_matches_top_m() -> None:
    prompt = _build_selection_prompt(
        "query",
        [Document(text=f"doc {index}") for index in range(10)],
        top_m=4,
    )

    assert '{"selected": [1, 2, 3, 4]}' in prompt
    assert "Return exactly 4 candidate numbers" in prompt
