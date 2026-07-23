from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksmith.confidence_generation import (  # noqa: E402
    ConfidenceGenerationResult,
    QueryAnswerabilityGenerationConfig,
    QueryContextAnswerabilityGenerationConfig,
    generate_query_answerability_confidence_dataset,
    generate_query_context_answerability_confidence_dataset,
)
from ranksmith.integrations import LMStudioModelProvider  # noqa: E402

TASK_QUERY = "query_answerability_confidence"
TASK_QUERY_CONTEXT = "query_context_answerability_confidence"
SUPPORTED_TASKS = (TASK_QUERY, TASK_QUERY_CONTEXT)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    provider = LMStudioModelProvider(
        base_url=args.lmstudio_base_url,
        model=args.lmstudio_model,
        api_key=args.lmstudio_api_key,
        max_tokens=args.lmstudio_max_tokens,
        timeout=args.timeout,
    )

    if args.task == TASK_QUERY:
        result = generate_query_answerability_confidence_dataset(
            QueryAnswerabilityGenerationConfig(
                input_path=args.input,
                output_path=args.output,
                provider=provider,
                overwrite=args.overwrite,
                resume=args.resume,
                max_items=args.max_items,
                source=args.source,
            )
        )
    else:
        result = generate_query_context_answerability_confidence_dataset(
            QueryContextAnswerabilityGenerationConfig(
                input_path=args.input,
                output_path=args.output,
                provider=provider,
                overwrite=args.overwrite,
                resume=args.resume,
                max_items=args.max_items,
                max_context_chars=args.max_context_chars,
                source=args.source,
            )
        )

    print(json.dumps(_generation_summary(result), ensure_ascii=False, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate answerability confidence canonical JSONL datasets."
    )
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--provider", required=True, choices=("lmstudio",))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument("--lmstudio-base-url", default=None)
    parser.add_argument("--lmstudio-model", default=None)
    parser.add_argument("--lmstudio-api-key", default=None)
    parser.add_argument("--lmstudio-max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args(argv)
    if args.task == TASK_QUERY and args.max_context_chars is not None:
        parser.error("--max-context-chars is only valid for query-context tasks")
    if args.task == TASK_QUERY_CONTEXT and args.max_context_chars is None:
        args.max_context_chars = 8000
    return args


def _generation_summary(result: ConfidenceGenerationResult) -> dict[str, int | str]:
    return {
        "output_path": str(result.output_path),
        "input_count": result.input_count,
        "generated_count": result.generated_count,
        "skipped_count": result.skipped_count,
        "positive_count": result.positive_count,
        "negative_count": result.negative_count,
    }


if __name__ == "__main__":
    raise SystemExit(main())
