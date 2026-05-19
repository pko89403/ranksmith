from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
