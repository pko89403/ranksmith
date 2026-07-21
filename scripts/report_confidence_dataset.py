from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksmith.confidence import TaskType  # noqa: E402
from ranksmith.confidence_training._dataset_report import (  # noqa: E402
    build_dataset_report,
)

TASK_QUERY = "query_answerability_confidence"
TASK_QUERY_CONTEXT = "query_context_answerability_confidence"
SUPPORTED_TASKS = (TASK_QUERY, TASK_QUERY_CONTEXT)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_dataset_report(args.dataset, cast(TaskType, args.task))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report confidence dataset balance.")
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--dataset", required=True, type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
