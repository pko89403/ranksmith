#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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
    EvaluationResult,
    aggregate_evaluations,
    aggregate_to_dict,
    evaluate_ranked_ids,
    evaluation_to_dict,
    load_beir_cases,
    load_fixture_cases,
)
from ranksmith._mteb_eval import (  # noqa: E402
    tourrank_stage_configs_for_candidate_count,
)

Algorithm = Literal[
    "original_bm25",
    "single_call_listwise@20",
    "rankgpt_sw_w5",
    "acurank_k5_b1",
    "acurank_k5_b4",
    "acurank_b1",
    "acurank_b4",
    "tourrank_r2",
    "tourrank_r10",
    "prp_sliding_p1",
    "prp_sliding_p3",
    "rankgpt_sliding_window",
    "rankgpt_sw",
    "prp_sliding_k",
    "tourrank_r",
    "acurank",
]
Dataset = Literal["fixture", "benchmark-cache", "beir-scifact"]
DEFAULT_FIXTURE = ROOT / "tests/fixtures/reranking_smoke_fixture.jsonl"
ALGORITHMS: tuple[Algorithm, ...] = (
    "original_bm25",
    "single_call_listwise@20",
    "rankgpt_sw_w5",
    "acurank_k5_b1",
    "tourrank_r2",
    "prp_sliding_p1",
)
OPTIONAL_ALGORITHMS: tuple[Algorithm, ...] = (
    "acurank_k5_b4",
    "acurank_b4",
    "tourrank_r10",
    "prp_sliding_p3",
)
LEGACY_ALGORITHMS: tuple[Algorithm, ...] = (
    "rankgpt_sliding_window",
    "rankgpt_sw",
    "prp_sliding_k",
    "tourrank_r",
    "acurank",
)


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    _load_env_file(args.env_file)
    cases = _load_cases(args)
    cases = _filter_cases_by_query_id(cases, args.query_id)
    algorithms = _selected_algorithms(args, cases)
    needs_live = any(algorithm != "original_bm25" for algorithm in algorithms)
    if needs_live and not args.allow_live:
        raise SystemExit("Refusing live Azure calls without --allow-live.")
    call_estimates = {
        algorithm: sum(
            _estimate_provider_calls(
                len(case.documents),
                algorithm,
                args.window_size,
                args.stride,
                args.passes,
                args.tourrank_rounds,
            )
            for case in cases
        )
        for algorithm in algorithms
    }
    if needs_live:
        print(
            "Live Azure comparison will run "
            f"{sum(call_estimates.values())} provider calls: {call_estimates}",
            file=sys.stderr,
        )
    else:
        print(
            f"Offline comparison will run {call_estimates}",
            file=sys.stderr,
        )

    evaluations, per_query = _evaluate_cases(
        args=args,
        algorithms=algorithms,
        cases=cases,
    )
    report = _build_report(
        args=args,
        algorithms=algorithms,
        cases=cases,
        call_estimates=call_estimates,
        per_query=per_query,
        aggregate=_aggregate_with_validity(evaluations, per_query),
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
        choices=("fixture", "benchmark-cache", "beir-scifact"),
        default="fixture",
        help="Benchmark source. fixture uses the committed smoke fixture.",
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Benchmark cache directory containing corpus.jsonl, queries.jsonl, "
            "and qrels/<split>.tsv. Use with --dataset benchmark-cache."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        help=(
            "Human-readable dataset label for benchmark-cache reports. "
            "Defaults to the cache directory name."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--candidates",
        type=Path,
        help=(
            "First-stage candidate file. Supports simple "
            "query/document TSV and Pyserini TREC run format."
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
        choices=("all", *ALGORITHMS, *OPTIONAL_ALGORITHMS, *LEGACY_ALGORITHMS),
        default="all",
    )
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--tourrank-rounds", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--query-id",
        action="append",
        default=[],
        help="Restrict the run to one or more query ids. Repeat for multiple ids.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Override Azure OpenAI request timeout in seconds for this run.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        help="Optional JSONL path for per-query rows written during long live runs.",
    )
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
    for name in (
        "window_size",
        "stride",
        "passes",
        "tourrank_rounds",
        "top_k",
        "candidate_count",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than 0.")
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be greater than 0.")
    if args.timeout is not None and args.timeout <= 0:
        raise SystemExit("--timeout must be greater than 0.")
    if args.checkpoint_output is not None and args.checkpoint_output == args.output:
        raise SystemExit("--checkpoint-output must differ from --output.")
    if args.dataset == "fixture":
        if args.cache_dir is not None:
            raise SystemExit(
                "--cache-dir is only valid with --dataset benchmark-cache."
            )
        if args.dataset_name is not None:
            raise SystemExit(
                "--dataset-name is only valid with --dataset benchmark-cache."
            )
        if args.candidates is not None:
            raise SystemExit(
                "--candidates is only valid with --dataset benchmark-cache."
            )
        if args.candidate_strategy != "candidate_file":
            raise SystemExit(
                "--candidate-strategy is only valid with --dataset benchmark-cache."
            )
    if args.dataset != "fixture" and args.cache_dir is None:
        raise SystemExit("--cache-dir is required with --dataset benchmark-cache.")
    if (
        args.dataset != "fixture"
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
    dataset_name = _dataset_name(args)
    return load_beir_cases(
        args.cache_dir,
        split=args.split,
        candidates_path=args.candidates,
        candidate_strategy=cast(CandidateStrategy, args.candidate_strategy),
        candidate_count=args.candidate_count,
        max_cases=args.max_cases,
        seed=args.seed,
        dataset_name=dataset_name,
        fixture_prefix=_fixture_prefix(dataset_name),
    )


def _filter_cases_by_query_id(
    cases: Sequence[BenchmarkCase],
    query_ids: Sequence[str],
) -> list[BenchmarkCase]:
    if not query_ids:
        return list(cases)
    wanted = set(query_ids)
    filtered = [case for case in cases if case.query_id in wanted]
    found = {case.query_id for case in filtered}
    missing = sorted(wanted - found)
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"--query-id not found in loaded cases: {joined}")
    return filtered


def _selected_algorithms(
    args: argparse.Namespace,
    cases: Sequence[BenchmarkCase],
) -> tuple[Algorithm, ...]:
    if args.algorithm != "all":
        return (cast(Algorithm, args.algorithm),)
    return ALGORITHMS


def _evaluate_cases(
    *,
    args: argparse.Namespace,
    algorithms: Sequence[Algorithm],
    cases: Sequence[BenchmarkCase],
) -> tuple[list[EvaluationResult], list[dict[str, object]]]:
    from ranksmith import RerankError

    if args.checkpoint_output is not None:
        args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint_output.write_text("", encoding="utf-8")

    evaluations: list[EvaluationResult] = []
    per_query: list[dict[str, object]] = []
    for algorithm in algorithms:
        print(f"Running {algorithm} on {len(cases)} cases.", file=sys.stderr)
        for index, case in enumerate(cases, 1):
            try:
                ranked_ids = _rank_case(
                    case=case,
                    algorithm=algorithm,
                    window_size=args.window_size,
                    stride=args.stride,
                    passes=args.passes,
                    tourrank_rounds=args.tourrank_rounds,
                    timeout=getattr(args, "timeout", None),
                )
                evaluation = evaluate_ranked_ids(
                    case=case,
                    algorithm=algorithm,
                    ranked_ids=ranked_ids,
                    top_k=args.top_k,
                )
                row = evaluation_to_dict(evaluation)
                row["valid"] = True
            except RerankError as exc:
                evaluation = evaluate_ranked_ids(
                    case=case,
                    algorithm=algorithm,
                    ranked_ids=(),
                    top_k=args.top_k,
                )
                row = evaluation_to_dict(evaluation)
                row["valid"] = False
                row["error_type"] = exc.__class__.__name__
                row["error"] = str(exc)
                print(
                    f"{algorithm} invalid query {index}/{len(cases)} "
                    f"query_id={case.query_id}: {exc}",
                    file=sys.stderr,
                )
            evaluations.append(evaluation)
            per_query.append(row)
            if args.checkpoint_output is not None:
                _append_checkpoint_row(args.checkpoint_output, row)
            if index == len(cases) or index % 25 == 0:
                print(
                    f"{algorithm} progress {index}/{len(cases)}",
                    file=sys.stderr,
                )
    return evaluations, per_query


def _rank_case(
    *,
    case: BenchmarkCase,
    algorithm: Algorithm,
    window_size: int,
    stride: int,
    passes: int,
    tourrank_rounds: int,
    timeout: float | None = None,
) -> tuple[str, ...]:
    from ranksmith import (
        AcuRankStrategy,
        AzureOpenAIReranker,
        Document,
        ListwiseStrategy,
        PairwiseStrategy,
        TourRankStrategy,
    )

    if algorithm == "original_bm25":
        return tuple(document.id for document in case.documents)
    if algorithm in {"prp_sliding_k", "prp_sliding_p1", "prp_sliding_p3"}:
        strategy = PairwiseStrategy(passes=_prp_passes_for_algorithm(algorithm, passes))
    elif algorithm in {"tourrank_r", "tourrank_r2", "tourrank_r10"}:
        strategy = TourRankStrategy(
            rounds=_tourrank_rounds_for_algorithm(algorithm, tourrank_rounds),
            stage_configs=tourrank_stage_configs_for_candidate_count(
                len(case.documents)
            ),
        )
    elif algorithm in {
        "acurank",
        "acurank_k5_b1",
        "acurank_k5_b4",
        "acurank_b1",
        "acurank_b4",
    }:
        target_rank = min(
            _acurank_target_rank_for_algorithm(algorithm),
            len(case.documents),
        )
        strategy = AcuRankStrategy(
            target_rank=target_rank,
            window_size=window_size,
            max_adaptive_reranker_calls=_acurank_budget_for_algorithm(algorithm),
        )
    else:
        listwise_window_size, listwise_stride = _listwise_window_stride_for_algorithm(
            algorithm,
            window_size,
            stride,
        )
        strategy = ListwiseStrategy(
            algorithm="rankgpt_sliding_window",
            window_size=listwise_window_size,
            stride=listwise_stride,
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
        timeout=timeout or _env_float("AZURE_OPENAI_LLM_TIMEOUT"),
        strategy=strategy,
    )
    documents = [
        Document(
            id=document.id,
            text=f"{document.title}\n\n{document.text}",
            metadata={"score": document.score} if document.score is not None else {},
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
        if args.dataset != "fixture" and args.candidate_strategy == "oracle_plus_random"
        else "reranking_with_first_stage_candidates"
    )
    if args.dataset == "fixture":
        benchmark_type = "smoke_fixture"
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": benchmark_type,
        "dataset": args.dataset,
        "dataset_name": None if args.dataset == "fixture" else _dataset_name(args),
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
        "tourrank_rounds": args.tourrank_rounds,
        "query_id_filter": list(args.query_id),
        "timeout": args.timeout,
        "tourrank_stage_policy": "paper_top_100_else_single_group_halving",
        "case_count": len(cases),
        "call_estimates": dict(call_estimates),
        "checkpoint_output": (
            str(args.checkpoint_output) if args.checkpoint_output is not None else None
        ),
        "aggregate": list(aggregate),
        "per_query": list(per_query),
    }


def _append_checkpoint_row(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _aggregate_with_validity(
    evaluations: Sequence[EvaluationResult],
    per_query: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    aggregates = [
        aggregate_to_dict(aggregate) for aggregate in aggregate_evaluations(evaluations)
    ]
    validity: dict[str, list[bool]] = {}
    for row in per_query:
        algorithm = str(row["algorithm"])
        validity.setdefault(algorithm, []).append(bool(row.get("valid", False)))
    for aggregate in aggregates:
        states = validity.get(str(aggregate["algorithm"]), [])
        valid_count = sum(1 for state in states if state)
        aggregate["valid_count"] = valid_count
        aggregate["invalid_rate"] = (
            0.0 if not states else (len(states) - valid_count) / len(states)
        )
    return aggregates


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


def _dataset_name(args: argparse.Namespace) -> str:
    if args.dataset == "beir-scifact":
        return args.dataset_name or "BEIR/SciFact"
    if args.dataset_name is not None and args.dataset_name.strip() != "":
        return args.dataset_name.strip()
    return args.cache_dir.name


def _fixture_prefix(dataset_name: str) -> str:
    prefix = "".join(
        character.lower() if character.isalnum() else "-" for character in dataset_name
    ).strip("-")
    return prefix or "benchmark-cache"


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
    tourrank_rounds: int = 2,
) -> int:
    if algorithm == "original_bm25":
        return 0
    if algorithm in {"prp_sliding_k", "prp_sliding_p1", "prp_sliding_p3"}:
        return (
            2
            * _prp_passes_for_algorithm(algorithm, passes)
            * max(document_count - 1, 0)
        )
    if algorithm in {"tourrank_r", "tourrank_r2", "tourrank_r10"}:
        rounds = _tourrank_rounds_for_algorithm(algorithm, tourrank_rounds)
        return rounds * sum(
            stage.group_count
            for stage in tourrank_stage_configs_for_candidate_count(document_count)
        )
    if algorithm in {
        "acurank",
        "acurank_k5_b1",
        "acurank_k5_b4",
        "acurank_b1",
        "acurank_b4",
    }:
        adaptive_budget = _acurank_budget_for_algorithm(algorithm)
        if adaptive_budget is None:
            adaptive_budget = 1
        return max(1, math.ceil(document_count / window_size) + adaptive_budget)
    window_size, stride = _listwise_window_stride_for_algorithm(
        algorithm,
        window_size,
        stride,
    )
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


def _prp_passes_for_algorithm(algorithm: Algorithm, passes: int) -> int:
    if algorithm == "prp_sliding_p1":
        return 1
    if algorithm == "prp_sliding_p3":
        return 3
    return passes


def _tourrank_rounds_for_algorithm(
    algorithm: Algorithm,
    tourrank_rounds: int,
) -> int:
    if algorithm == "tourrank_r2":
        return 2
    if algorithm == "tourrank_r10":
        return 10
    return tourrank_rounds


def _acurank_budget_for_algorithm(algorithm: Algorithm) -> int | None:
    if algorithm in {"acurank_k5_b1", "acurank_b1"}:
        return 1
    if algorithm in {"acurank_k5_b4", "acurank_b4"}:
        return 4
    return None


def _acurank_target_rank_for_algorithm(algorithm: Algorithm) -> int:
    if algorithm in {"acurank_k5_b1", "acurank_k5_b4"}:
        return 5
    return 10


def _listwise_window_stride_for_algorithm(
    algorithm: Algorithm,
    window_size: int,
    stride: int,
) -> tuple[int, int]:
    if algorithm == "rankgpt_sw_w5":
        return 5, 2
    return window_size, stride


if __name__ == "__main__":
    main()
