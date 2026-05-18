#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksmith._mteb_eval import (  # noqa: E402
    MtebRerankingCandidate,
    MtebRerankingSample,
    PriceConfig,
    compute_query_metrics,
    estimate_cost,
    mean,
    normalize_method_name,
    percentile,
    stable_seed,
)


def main() -> None:
    args = _parse_args()
    if args.overwrite and args.resume:
        raise SystemExit("--overwrite and --resume cannot be used together.")
    if (args.input_token_price_per_1m is None) != (
        args.output_token_price_per_1m is None
    ):
        raise SystemExit(
            "--input-token-price-per-1m and --output-token-price-per-1m "
            "must be provided together."
        )
    if args.input_token_price_per_1m is not None and (
        args.input_token_price_per_1m < 0 or args.output_token_price_per_1m < 0
    ):
        raise SystemExit("Token prices must be greater than or equal to 0.")

    if args.list_tasks:
        _list_tasks()
        return
    if args.inspect_task_schema is not None:
        _inspect_task_schema(args.inspect_task_schema)
        return

    normalized_methods = [normalize_method_name(method) for method in args.methods]

    needs_live = any(method != "original" for method in normalized_methods)
    if not args.allow_live and needs_live:
        raise SystemExit("Refusing live Azure calls without --allow-live.")

    print(
        "Strict validation is enabled.\n"
        "Invalid LLM outputs are not repaired.\n"
        "Invalid query-method results will receive zero scores.",
        file=sys.stderr,
    )

    _load_env_file(args.env_file)

    if args.output_dir is None:
        print(
            "No --output-dir provided; validating arguments only. "
            "Pass --output-dir to run evaluation.",
            file=sys.stderr,
        )
        return

    output_dir: Path = args.output_dir
    query_results_path = output_dir / "query_results.jsonl"
    if query_results_path.exists() and not (args.overwrite or args.resume):
        raise SystemExit(
            f"{query_results_path} already exists. Pass --overwrite or --resume."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and query_results_path.exists():
        query_results_path.unlink()

    price_config = (
        PriceConfig(args.input_token_price_per_1m, args.output_token_price_per_1m)
        if args.input_token_price_per_1m is not None
        else None
    )

    samples = list(
        _load_samples(
            task_names=args.tasks,
            all_english=args.all_english_reranking_tasks,
            split=args.split,
            max_queries=args.max_queries,
            max_document_chars=args.max_document_chars,
            shuffle_candidates=args.shuffle_candidates,
            shuffle_seed=args.shuffle_seed,
        )
    )

    rows: list[dict[str, Any]] = []
    already_done: set[tuple[str, str, str, str]] = set()
    if args.resume and query_results_path.exists():
        for line in query_results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            already_done.add(
                (
                    str(row["task"]),
                    str(row["split"]),
                    str(row["query_id"]),
                    str(row["method"]),
                )
            )

    total_pairs = len(samples) * len(normalized_methods)
    done_pairs = 0
    with query_results_path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            for method in normalized_methods:
                done_pairs += 1
                key = (sample.task_name, sample.split, sample.query_id, method)
                if key in already_done:
                    continue
                row = _evaluate_sample_method(
                    sample=sample,
                    method=method,
                    price_config=price_config,
                    rankgpt_window_size=args.rankgpt_window_size,
                    rankgpt_step=args.rankgpt_step,
                )
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                print(
                    f"[{done_pairs}/{total_pairs}] {sample.task_name} "
                    f"q={sample.query_id} method={method} "
                    f"valid={row['valid']} "
                    f"failure={row['failure_type']} "
                    f"latency={row['latency_ms']:.0f}ms",
                    file=sys.stderr,
                    flush=True,
                )

    _write_summaries(
        output_dir=output_dir,
        rows=rows,
        methods=normalized_methods,
        args=args,
    )


def _evaluate_sample_method(
    *,
    sample: MtebRerankingSample,
    method: str,
    price_config: PriceConfig | None,
    rankgpt_window_size: int,
    rankgpt_step: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    ranked_doc_ids: tuple[str, ...] | None
    valid = True
    failure_type: str | None = None
    usage: tuple[int, int] | None = None
    error: str | None = None

    if method == "original":
        ranked_doc_ids = tuple(candidate.doc_id for candidate in sample.candidates)
    else:
        ranked_doc_ids, valid, failure_type, usage, error = _run_llm_method(
            sample=sample,
            method=method,
            rankgpt_window_size=rankgpt_window_size,
            rankgpt_step=rankgpt_step,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    metrics = compute_query_metrics(
        sample=sample,
        ranked_doc_ids=ranked_doc_ids,
        valid=valid,
    )
    return {
        "task": sample.task_name,
        "split": sample.split,
        "query_id": sample.query_id,
        "method": method,
        "metrics": metrics,
        "valid": valid,
        "failure_type": failure_type,
        "error": error,
        "latency_ms": elapsed_ms,
        "usage": list(usage) if usage is not None else None,
        "cost": estimate_cost(usage, price_config),
        "ranked_doc_ids": list(ranked_doc_ids) if ranked_doc_ids is not None else None,
    }


def _run_llm_method(
    *,
    sample: MtebRerankingSample,
    method: str,
    rankgpt_window_size: int,
    rankgpt_step: int,
) -> tuple[
    tuple[str, ...] | None,
    bool,
    str | None,
    tuple[int, int] | None,
    str | None,
]:
    from openai import APIError

    from ranksmith import AzureOpenAIReranker, Document, ListwiseStrategy
    from ranksmith.errors import RerankParseError, RerankProviderError

    if method.startswith("direct@"):
        algorithm = "direct"
        rank_end = int(method.removeprefix("direct@"))
        window_size = rank_end
        stride = rank_end
    else:
        algorithm = "rankgpt_sliding_window"
        rank_end = int(method.removeprefix("rankgpt_sliding_window@"))
        window_size = rankgpt_window_size
        stride = rankgpt_step

    totals = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def on_usage(usage: Any) -> None:
        totals["prompt_tokens"] += usage.prompt_tokens
        totals["completion_tokens"] += usage.completion_tokens
        totals["calls"] += 1

    reranker = AzureOpenAIReranker(
        api_key=_required_env("AZURE_OPENAI_API_KEY"),
        azure_endpoint=_required_env("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=_required_env(
            "AZURE_OPENAI_LLM_DEPLOYMENT",
            fallback="AZURE_OPENAI_DEPLOYMENT",
        ),
        api_version=os.environ.get(
            "AZURE_OPENAI_LLM_API_VERSION",
            os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        ),
        strategy=ListwiseStrategy(
            algorithm=algorithm,
            window_size=window_size,
            stride=stride,
        ),
        on_usage=on_usage,
    )
    documents = [
        Document(id=candidate.doc_id, text=candidate.text)
        for candidate in sample.candidates[:rank_end]
    ]
    try:
        results = reranker.rerank(sample.query, documents)
    except RerankParseError as exc:
        return None, False, "llm_output_invalid", _read_usage(totals), str(exc)
    except (RerankProviderError, APIError) as exc:
        return None, False, "provider_error", _read_usage(totals), str(exc)
    ranked = tuple(result.document.id or "" for result in results)
    return ranked, True, None, _read_usage(totals), None


def _read_usage(totals: dict[str, int]) -> tuple[int, int] | None:
    if totals["calls"] == 0:
        return None
    return totals["prompt_tokens"], totals["completion_tokens"]


def _write_summaries(
    *,
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    methods: Sequence[str],
    args: argparse.Namespace,
) -> None:
    task_summary: dict[str, dict[str, dict[str, float]]] = {}
    overall: dict[str, dict[str, float]] = {}

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        overall[method] = _aggregate(method_rows)
        for task in sorted({row["task"] for row in method_rows}):
            task_rows = [row for row in method_rows if row["task"] == task]
            task_summary.setdefault(task, {})[method] = _aggregate(task_rows)

    (output_dir / "task_summary.json").write_text(
        json.dumps(task_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "overall_summary.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            _collect_metadata(args=args, methods=methods),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_result_tables(
        output_dir / "result_tables.md", overall=overall, task_summary=task_summary
    )


def _collect_metadata(
    *, args: argparse.Namespace, methods: Sequence[str]
) -> dict[str, Any]:
    return {
        "tasks": args.tasks,
        "methods": list(methods),
        "split": args.split,
        "max_queries": args.max_queries,
        "max_document_chars": args.max_document_chars,
        "strict_validation": True,
        "zero_score_on_invalid": True,
        "shuffle_candidates": args.shuffle_candidates,
        "shuffle_seed": args.shuffle_seed,
        "rankgpt_window_size": args.rankgpt_window_size,
        "rankgpt_step": args.rankgpt_step,
        "azure_deployment": os.environ.get("AZURE_OPENAI_LLM_DEPLOYMENT")
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        "azure_api_version": os.environ.get("AZURE_OPENAI_LLM_API_VERSION")
        or os.environ.get("AZURE_OPENAI_API_VERSION"),
        "ranksmith_version": _read_ranksmith_version(),
        "openai_sdk_version": _read_module_version("openai"),
        "mteb_version": _read_module_version("mteb"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _read_git_commit(),
        "started_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def _get_split_field(split_data: Any, name: str) -> Any:
    if hasattr(split_data, "get"):
        value = split_data.get(name)
        if value is not None:
            return value
    columns = getattr(split_data, "column_names", None)
    if columns is not None and name in columns:
        return split_data[name]
    try:
        if name in split_data:
            return split_data[name]
    except TypeError:
        pass
    return None


def _read_module_version(module_name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return None
    try:
        return version(module_name)
    except PackageNotFoundError:
        return None


def _read_ranksmith_version() -> str | None:
    return _read_module_version("ranksmith")


def _read_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    latencies = [float(row["latency_ms"]) for row in rows]
    metric_keys = list(rows[0]["metrics"].keys())
    valid_rows = [row for row in rows if row["valid"]]
    aggregate: dict[str, Any] = {}
    for key in metric_keys:
        aggregate[key] = mean([float(row["metrics"][key]) for row in rows])
        aggregate[f"{key}_valid_only"] = mean(
            [float(row["metrics"][key]) for row in valid_rows]
        )
    aggregate["latency_ms_p50"] = percentile(latencies, 50.0)
    aggregate["latency_ms_p95"] = percentile(latencies, 95.0)
    aggregate["invalid_rate"] = mean([0.0 if row["valid"] else 1.0 for row in rows])
    aggregate["valid_count"] = float(len(valid_rows))
    aggregate["query_count"] = float(len(rows))
    failure_counter: Counter[str] = Counter()
    for row in rows:
        failure_type = row.get("failure_type")
        if failure_type:
            failure_counter[failure_type] += 1
    aggregate["failure_type_counts"] = dict(failure_counter)
    usage_rows = [row for row in rows if row.get("usage") is not None]
    if usage_rows:
        aggregate["prompt_tokens_mean"] = mean(
            [float(row["usage"][0]) for row in usage_rows]
        )
        aggregate["completion_tokens_mean"] = mean(
            [float(row["usage"][1]) for row in usage_rows]
        )
    cost_rows = [row for row in rows if row.get("cost") is not None]
    if cost_rows:
        total = sum(float(row["cost"]) for row in cost_rows)
        aggregate["cost_total"] = total
        aggregate["cost_mean_costed_only"] = total / len(cost_rows)
        aggregate["cost_mean_per_query"] = total / len(rows)
    return aggregate


def _write_result_tables(
    path: Path,
    *,
    overall: dict[str, dict[str, Any]],
    task_summary: dict[str, dict[str, dict[str, Any]]],
) -> None:
    lines = [
        "# MTEB Reranking Results",
        "",
        "Strict validation: invalid LLM outputs receive zero scores.",
        "Zero-score policy applies to main metrics; `valid-only` columns "
        "report metrics computed over the valid subset only.",
        "",
        "## Overall",
        "",
    ]
    lines.extend(_render_metric_table(overall))
    for task_name in sorted(task_summary):
        lines.append("")
        lines.append(f"## Task: {task_name}")
        lines.append("")
        lines.extend(_render_metric_table(task_summary[task_name]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_metric_table(rollup: dict[str, dict[str, Any]]) -> list[str]:
    header = (
        "| Method | ndcg@10 | ndcg@10 (valid-only) | mrr@10 | map | "
        "recall@10 | p50_ms | p95_ms | invalid_rate | n |"
    )
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, separator]
    for method, agg in sorted(rollup.items()):
        lines.append(
            "| {method} | {ndcg:.4f} | {ndcg_v:.4f} | {mrr:.4f} | {map_:.4f} | "
            "{recall:.4f} | {p50:.1f} | {p95:.1f} | {invalid:.3f} | {n:.0f} |".format(
                method=method,
                ndcg=agg.get("ndcg@10", 0.0),
                ndcg_v=agg.get("ndcg@10_valid_only", 0.0),
                mrr=agg.get("mrr@10", 0.0),
                map_=agg.get("map", 0.0),
                recall=agg.get("recall@10", 0.0),
                p50=agg.get("latency_ms_p50", 0.0),
                p95=agg.get("latency_ms_p95", 0.0),
                invalid=agg.get("invalid_rate", 0.0),
                n=agg.get("query_count", 0.0),
            )
        )
    return lines


def _load_samples(
    *,
    task_names: Sequence[str],
    all_english: bool,
    split: str,
    max_queries: int | None,
    max_document_chars: int,
    shuffle_candidates: bool = False,
    shuffle_seed: int = 13,
) -> Iterable[MtebRerankingSample]:
    try:
        import mteb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "mteb is not installed. Install the optional MTEB dependency."
        ) from exc

    if all_english:
        tasks = mteb.get_tasks(task_types=["Reranking"], languages=["eng"])
        names = [
            name
            for name in (
                getattr(getattr(task, "metadata", task), "name", "") for task in tasks
            )
            if name
        ]
    else:
        names = list(task_names)

    for name in names:
        task = mteb.get_task(name)
        task.load_data()
        yield from _samples_from_task(
            task_name=name,
            split=split,
            task_dataset=task.dataset,
            max_queries=max_queries,
            max_document_chars=max_document_chars,
            shuffle_candidates=shuffle_candidates,
            shuffle_seed=shuffle_seed,
        )


def _samples_from_task(
    *,
    task_name: str,
    split: str,
    task_dataset: Any,
    max_queries: int | None,
    max_document_chars: int,
    shuffle_candidates: bool = False,
    shuffle_seed: int = 13,
) -> Iterable[MtebRerankingSample]:
    subset = next(iter(task_dataset.values()))
    split_data = subset[split]
    corpus = {row["id"]: row for row in split_data["corpus"]}
    queries = {row["id"]: row["text"] for row in split_data["queries"]}
    relevant = split_data["relevant_docs"]
    top_ranked = _get_split_field(split_data, "top_ranked")
    if top_ranked is None:
        raise SystemExit(
            f"Task {task_name} does not expose a 'top_ranked' candidate field; "
            "this CLI requires the BEIR-style MTEB reranking schema."
        )

    query_ids = sorted(top_ranked.keys())
    for index, query_id in enumerate(query_ids):
        if max_queries is not None and index >= max_queries:
            break
        query_text = queries.get(query_id, "")
        qrels = relevant.get(query_id, {})
        candidate_ids = list(top_ranked[query_id])
        candidates: list[MtebRerankingCandidate] = []
        for doc_id in candidate_ids:
            doc = corpus.get(doc_id, {})
            title = doc.get("title", "")
            text = doc.get("text", "")
            combined = f"{title}\n\n{text}".strip() if title else str(text)
            candidates.append(
                MtebRerankingCandidate(
                    doc_id=str(doc_id),
                    text=combined[:max_document_chars],
                    label=float(qrels.get(doc_id, 0)),
                )
            )
        if shuffle_candidates:
            rng = random.Random(stable_seed(shuffle_seed, task_name, query_id))
            rng.shuffle(candidates)
        yield MtebRerankingSample(
            task_name=task_name,
            split=split,
            query_id=str(query_id),
            query=str(query_text),
            candidates=tuple(candidates),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ranksmith methods on native MTEB reranking tasks."
    )
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--all-english-reranking-tasks", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["original", "direct@20", "rankgpt_sliding_window@100"],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--max-document-chars", type=int, default=4000)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--input-token-price-per-1m", type=float)
    parser.add_argument("--output-token-price-per-1m", type=float)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--inspect-task-schema")
    parser.add_argument("--shuffle-candidates", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=13)
    parser.add_argument("--rankgpt-window-size", type=int, default=20)
    parser.add_argument("--rankgpt-step", type=int, default=10)
    return parser.parse_args()


def _list_tasks() -> None:
    try:
        import mteb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "mteb is not installed. Install the optional MTEB dependency."
        ) from exc
    _hint_ssl_if_needed()
    tasks = mteb.get_tasks(task_types=["Reranking"], languages=["eng"])
    print(
        json.dumps(
            [getattr(task, "metadata", task).__dict__ for task in tasks],
            default=str,
        )
    )


def _inspect_task_schema(task_name: str) -> None:
    try:
        import mteb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "mteb is not installed. Install the optional MTEB dependency."
        ) from exc
    _hint_ssl_if_needed()
    task = mteb.get_task(task_name)
    task.load_data()
    print(json.dumps({"task": task_name, "dataset": str(task.dataset)}, default=str))


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from ``path``. Existing process env wins (setdefault)."""
    expanded = path.expanduser()
    if not expanded.exists():
        return
    for line_number, line in enumerate(
        expanded.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if separator == "":
            continue
        key = key.strip()
        if key == "" or any(character.isspace() for character in key):
            raise SystemExit(f"Invalid key on {expanded}:{line_number}: {key!r}")
        os.environ.setdefault(key, _strip_matched_quotes(value.strip()))


def _hint_ssl_if_needed() -> None:
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE"):
        return
    print(
        "Note: if HuggingFace downloads fail with SSL certificate errors "
        "(self-signed CA), export SSL_CERT_FILE pointing to your system CA bundle.",
        file=sys.stderr,
    )


def _strip_matched_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _required_env(name: str, *, fallback: str | None = None) -> str:
    value = os.environ.get(name)
    if value:
        return value
    if fallback:
        fallback_value = os.environ.get(fallback)
        if fallback_value:
            return fallback_value
    raise SystemExit(f"Missing required environment variable: {name}")


if __name__ == "__main__":
    main()
