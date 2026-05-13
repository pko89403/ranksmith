#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypedDict, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

Algorithm = Literal["direct", "sliding_window", "rankgpt_sliding_window"]
DEFAULT_FIXTURE = ROOT / "tests/fixtures/reranking_smoke_fixture.jsonl"
ALGORITHMS: tuple[Algorithm, ...] = (
    "direct",
    "sliding_window",
    "rankgpt_sliding_window",
)


class FixtureDocument(TypedDict):
    id: str
    title: str
    text: str


class FixtureCase(TypedDict):
    schema_version: int
    fixture_id: str
    query: str
    documents: list[FixtureDocument]
    qrels: dict[str, int]


def main() -> None:
    args = _parse_args()
    _load_env_file(args.env_file)
    if not args.allow_live:
        raise SystemExit("Refusing live Azure calls without --allow-live.")

    algorithms = (
        ALGORITHMS if args.algorithm == "all" else (cast(Algorithm, args.algorithm),)
    )
    cases = _load_cases(args.fixture)
    call_estimates = {
        algorithm: sum(
            _estimate_provider_calls(
                len(case["documents"]), algorithm, args.window_size, args.stride
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

    results = [
        _evaluate_case(
            case=case,
            algorithm=algorithm,
            window_size=args.window_size,
            stride=args.stride,
            top_k=args.top_k,
        )
        for algorithm in algorithms
        for case in cases
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ranksmith reranking algorithms on a qrels-backed fixture."
        )
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--algorithm",
        choices=("all", *ALGORITHMS),
        default="all",
    )
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
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


def _evaluate_case(
    *,
    case: FixtureCase,
    algorithm: Algorithm,
    window_size: int,
    stride: int,
    top_k: int,
) -> dict[str, object]:
    from ranksmith import AzureOpenAIReranker, Document, ListwiseStrategy
    from ranksmith._metrics import mrr_at_k, ndcg_at_k, recall_at_k

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
        strategy=ListwiseStrategy(
            algorithm=algorithm,
            window_size=window_size,
            stride=stride,
        ),
    )
    documents = [
        Document(
            id=document["id"],
            text=f"{document['title']}\n\n{document['text']}",
        )
        for document in case["documents"]
    ]
    ranked_ids = [
        result.document.id or "" for result in reranker.rerank(case["query"], documents)
    ]
    return {
        "fixture_id": case["fixture_id"],
        "algorithm": algorithm,
        "ranked_ids": ranked_ids,
        f"ndcg@{top_k}": ndcg_at_k(ranked_ids, case["qrels"], top_k),
        f"mrr@{top_k}": mrr_at_k(ranked_ids, case["qrels"], top_k),
        f"recall@{top_k}": recall_at_k(ranked_ids, case["qrels"], top_k),
    }


def _load_cases(path: Path) -> list[FixtureCase]:
    return [
        cast(FixtureCase, json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


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
) -> int:
    if algorithm == "direct" or document_count <= window_size:
        return 1
    if algorithm == "sliding_window":
        return len(tuple(_window_starts(document_count, window_size, stride)))
    start_pos = document_count - window_size
    calls = 0
    while True:
        start_pos = max(0, start_pos)
        calls += 1
        if start_pos == 0:
            return calls
        start_pos -= stride


def _window_starts(
    document_count: int,
    window_size: int,
    stride: int,
) -> Iterable[int]:
    last_start = document_count - window_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


if __name__ == "__main__":
    main()
