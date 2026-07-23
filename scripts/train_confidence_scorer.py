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
from ranksmith.confidence_training import (  # noqa: E402
    ConfidenceTrainingConfig,
    ConfidenceTrainingResult,
    train_confidence_scorer,
)

TASK_QUERY = "query_answerability_confidence"
TASK_QUERY_CONTEXT = "query_context_answerability_confidence"
SUPPORTED_TASKS = (TASK_QUERY, TASK_QUERY_CONTEXT)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = train_confidence_scorer(
        ConfidenceTrainingConfig(
            task_type=cast(TaskType, args.task),
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            export_path=args.export_path,
            encoder_name=args.encoder_name,
            encoder_revision=args.encoder_revision,
            tokenizer_name=args.tokenizer_name,
            tokenizer_revision=args.tokenizer_revision,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            allow_truncation=args.allow_truncation,
            seed=args.seed,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            calibration_method=args.calibration_method,
        )
    )

    print(json.dumps(_training_summary(result), ensure_ascii=False, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a confidence scorer artifact.")
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--export-path", required=True, type=Path)
    parser.add_argument("--encoder-name", default="bert-base-uncased")
    parser.add_argument("--encoder-revision", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--calibration-method", choices=("sigmoid",), default="sigmoid")
    return parser.parse_args(argv)


def _training_summary(result: ConfidenceTrainingResult) -> dict[str, str]:
    return {
        "output_dir": str(result.output_dir),
        "export_path": str(result.export_path),
        "report_path": str(result.report_path),
        "metadata_path": str(result.metadata_path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
