#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksmith._benchmark import (  # noqa: E402
    SCHEMA_VERSION,
    BenchmarkCase,
    CandidateStrategy,
    aggregate_evaluations,
    aggregate_to_dict,
    evaluate_ranked_ids,
    evaluation_to_dict,
    load_beir_cases,
    load_fixture_cases,
)

Algorithm = Literal[
    "rankgpt_sliding_window",
    "prp_sliding_k",
]
Dataset = Literal["fixture", "beir-scifact"]
DEFAULT_FIXTURE = ROOT / "tests/fixtures/reranking_smoke_fixture.jsonl"
ALGORITHMS: tuple[Algorithm, ...] = (
    "rankgpt_sliding_window",
    "prp_sliding_k",
)


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    _load_env_file(args.env_file)
    if not args.allow_live:
        raise SystemExit("Refusing live Azure calls without --allow-live.")

    algorithms = (
        ALGORITHMS if args.algorithm == "all" else (cast(Algorithm, args.algorithm),)
    )
    cases = _load_cases(args)
    call_estimates = {
        algorithm: sum(
            _estimate_provider_calls(
                len(case.documents),
                algorithm,
                args.window_size,
                args.stride,
                args.passes,
            )
            for case in cases
        )
        for algorithm in algorithms
    }
    print(
        "Live Azure comparison will run "
        f"{sum(call_estimates.values())} provider calls: {call_estimates}",
        file=sys.stderr,
    )

    evaluations = [
        evaluate_ranked_ids(
            case=case,
            algorithm=algorithm,
            ranked_ids=_rank_case(
                case=case,
                algorithm=algorithm,
                window_size=args.window_size,
                stride=args.stride,
                passes=args.passes,
            ),
            top_k=args.top_k,
        )
        for algorithm in algorithms
        for case in cases
    ]
    report = _build_report(
        args=args,
        algorithms=algorithms,
        cases=cases,
        call_estimates=call_estimates,
        per_query=[evaluation_to_dict(evaluation) for evaluation in evaluations],
        aggregate=[
            aggregate_to_dict(aggregate)
            for aggregate in aggregate_evaluations(evaluations)
        ],
    )
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(output)
    else:
        args.output.write_text(f"{output}\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ranksmith reranking algorithms on qrels-backed benchmark cases."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("fixture", "beir-scifact"),
        default="fixture",
        help="Benchmark source. fixture uses the committed smoke fixture.",
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "BEIR/SciFact cache directory containing corpus.jsonl, queries.jsonl, "
            "and qrels/<split>.tsv."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--candidates",
        type=Path,
        help=(
            "First-stage candidate TSV for BEIR mode. Each row must start with "
            "query_id and document_id."
        ),
    )
    parser.add_argument(
        "--candidate-strategy",
        choices=("candidate_file", "oracle_plus_random"),
        default="candidate_file",
        help=(
            "candidate_file is required for benchmark-style reranking. "
            "oracle_plus_random is diagnostic only."
        ),
    )
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--algorithm",
        choices=("all", *ALGORITHMS),
        default="all",
    )
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help="Path to a .env file. Existing process environment values win.",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Required because this script sends live Azure OpenAI requests.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("window_size", "stride", "passes", "top_k", "candidate_count"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than 0.")
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be greater than 0.")
    if args.dataset == "fixture":
        if args.cache_dir is not None:
            raise SystemExit("--cache-dir is only valid with --dataset beir-scifact.")
        if args.candidates is not None:
            raise SystemExit("--candidates is only valid with --dataset beir-scifact.")
        if args.candidate_strategy != "candidate_file":
            raise SystemExit(
                "--candidate-strategy is only valid with --dataset beir-scifact."
            )
    if args.dataset == "beir-scifact" and args.cache_dir is None:
        raise SystemExit("--cache-dir is required with --dataset beir-scifact.")
    if (
        args.dataset == "beir-scifact"
        and args.candidate_strategy == "candidate_file"
        and args.candidates is None
    ):
        raise SystemExit(
            "--candidates is required for BEIR benchmark mode. "
            "Use --candidate-strategy oracle_plus_random only for diagnostics."
        )


def _load_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    if args.dataset == "fixture":
        return load_fixture_cases(args.fixture)
    return load_beir_cases(
        args.cache_dir,
        split=args.split,
        candidates_path=args.candidates,
        candidate_strategy=cast(CandidateStrategy, args.candidate_strategy),
        candidate_count=args.candidate_count,
        max_cases=args.max_cases,
        seed=args.seed,
    )


def _rank_case(
    *,
    case: BenchmarkCase,
    algorithm: Algorithm,
    window_size: int,
    stride: int,
    passes: int,
) -> tuple[str, ...]:
    from ranksmith import (
        AzureOpenAIReranker,
        Document,
        ListwiseStrategy,
        PairwiseStrategy,
    )

    if algorithm == "prp_sliding_k":
        strategy = PairwiseStrategy(passes=passes)
    else:
        strategy = ListwiseStrategy(
            algorithm=algorithm,
            window_size=window_size,
            stride=stride,
        )

    reranker = AzureOpenAIReranker(
        api_key=_required_env("AZURE_OPENAI_API_KEY"),
        azure_endpoint=_required_env("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=_required_env(
            "AZURE_OPENAI_LLM_DEPLOYMENT",
            fallback="AZURE_OPENAI_DEPLOYMENT",
        ),
        api_version=_env_value(
            "AZURE_OPENAI_LLM_API_VERSION",
            fallback="AZURE_OPENAI_API_VERSION",
            default="2024-08-01-preview",
        ),
        timeout=_env_float("AZURE_OPENAI_LLM_TIMEOUT"),
        strategy=strategy,
    )
    documents = [
        Document(
            id=document.id,
            text=f"{document.title}\n\n{document.text}",
        )
        for document in case.documents
    ]
    return tuple(
        result.document.id or "" for result in reranker.rerank(case.query, documents)
    )


def _build_report(
    *,
    args: argparse.Namespace,
    algorithms: Sequence[Algorithm],
    cases: Sequence[BenchmarkCase],
    call_estimates: Mapping[str, int],
    per_query: Sequence[dict[str, object]],
    aggregate: Sequence[dict[str, object]],
) -> dict[str, object]:
    benchmark_type = (
        "diagnostic_not_retrieval"
        if args.dataset == "beir-scifact"
        and args.candidate_strategy == "oracle_plus_random"
        else "reranking_with_first_stage_candidates"
    )
    if args.dataset == "fixture":
        benchmark_type = "smoke_fixture"
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": benchmark_type,
        "dataset": args.dataset,
        "fixture": str(args.fixture) if args.dataset == "fixture" else None,
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        "candidates": str(args.candidates) if args.candidates is not None else None,
        "candidate_strategy": args.candidate_strategy,
        "candidate_count": args.candidate_count,
        "seed": args.seed,
        "algorithm": list(algorithms),
        "top_k": args.top_k,
        "window_size": args.window_size,
        "stride": args.stride,
        "passes": args.passes,
        "case_count": len(cases),
        "call_estimates": dict(call_estimates),
        "aggregate": list(aggregate),
        "per_query": list(per_query),
    }


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if separator == "":
            raise SystemExit(f"Invalid .env line without '=': {line}")
        key = key.strip()
        if key == "":
            raise SystemExit(f"Invalid .env line with empty key: {line}")
        os.environ.setdefault(key, _clean_env_value(value))


def _clean_env_value(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith(("'", '"')):
        quote = stripped[0]
        end = stripped.find(quote, 1)
        if end == -1:
            raise SystemExit("Invalid .env quoted value.")
        return stripped[1:end]
    return stripped.split("#", maxsplit=1)[0].strip()


def _required_env(name: str, *, fallback: str | None = None) -> str:
    value = _env_value(name, fallback=fallback)
    if value is None or value == "":
        names = name if fallback is None else f"{name} or {fallback}"
        raise SystemExit(f"Missing required environment variable: {names}")
    return value


def _env_value(
    name: str,
    *,
    fallback: str | None = None,
    default: str | None = None,
) -> str | None:
    value = os.environ.get(name)
    if value is not None and value != "":
        return value
    if fallback is not None:
        fallback_value = os.environ.get(fallback)
        if fallback_value is not None and fallback_value != "":
            return fallback_value
    return default


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


def _estimate_provider_calls(
    document_count: int,
    algorithm: Algorithm,
    window_size: int,
    stride: int,
    passes: int = 10,
) -> int:
    if algorithm == "prp_sliding_k":
        return 2 * passes * max(document_count - 1, 0)
    if document_count <= window_size:
        return 1
    start_pos = document_count - window_size
    calls = 0
    while True:
        start_pos = max(0, start_pos)
        calls += 1
        if start_pos == 0:
            return calls
        start_pos -= stride


if __name__ == "__main__":
    main()
