# Native MTEB Reranking Evaluation CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI that evaluates ranksmith reranking methods on native MTEB Reranking tasks and produces reference-quality quality, latency, token, cost, and reliability reports.

**Architecture:** Keep the public `ranksmith` API unchanged. Add private evaluation helpers in `src/ranksmith/_mteb_eval.py`, a CLI in `scripts/evaluate_mteb_reranking.py`, and focused tests in `tests/test_mteb_eval.py`. The CLI uses an evaluation-only Azure provider to capture raw responses, latency, usage, and cost while preserving ranksmith's strict 1-based integer permutation contract.

**Tech Stack:** Python 3.10+, `uv`, `pytest`, `ruff`, `mypy`, OpenAI Python SDK, optional/dev `mteb`.

---

## File Structure

- Create `/Users/skiiwoo/Documents/New project 2/src/ranksmith/_mteb_eval.py`
  - Private schemas, method parsing, metric helpers, MTEB sample normalization hooks, strict failure policy, reporting helpers.
- Create `/Users/skiiwoo/Documents/New project 2/scripts/evaluate_mteb_reranking.py`
  - CLI, MTEB imports, Azure wiring, live guard, output file writing.
- Create `/Users/skiiwoo/Documents/New project 2/tests/test_mteb_eval.py`
  - Unit tests without MTEB downloads or Azure calls.
- Modify `/Users/skiiwoo/Documents/New project 2/src/ranksmith/_metrics.py`
  - Add `map_score`.
- Modify `/Users/skiiwoo/Documents/New project 2/pyproject.toml`
  - Add optional/dev dependency for `mteb`.
- Modify `/Users/skiiwoo/Documents/New project 2/.gitignore`
  - Ignore `/benchmark-results/`.
- Modify `/Users/skiiwoo/Documents/New project 2/README.md` or `/Users/skiiwoo/Documents/New project 2/README.ko.md`
  - Add reference benchmark usage and interpretation caveats after CLI verification.
- Modify `/Users/skiiwoo/Documents/New project 2/docs/specs/spec_mteb_reranking_evaluation.md`
  - Check completed implementation tasks and final status when verified.

---

### Task 1: Metric Helper

**Files:**
- Modify: `/Users/skiiwoo/Documents/New project 2/src/ranksmith/_metrics.py`
- Create: `/Users/skiiwoo/Documents/New project 2/tests/test_mteb_eval.py`

- [ ] **Step 1: Write failing tests for MAP**

Add these tests to `/Users/skiiwoo/Documents/New project 2/tests/test_mteb_eval.py`.

```python
from ranksmith._metrics import map_score


def test_map_score_uses_binary_relevance() -> None:
    ranking = ["d1", "d2", "d3", "d4"]
    qrels = {"d1": 0, "d2": 2, "d3": 0, "d4": 1}

    assert map_score(ranking, qrels) == (1 / 2 + 2 / 4) / 2


def test_map_score_returns_zero_when_no_relevant_documents() -> None:
    assert map_score(["d1", "d2"], {"d1": 0, "d2": 0}) == 0.0
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
uv run pytest tests/test_mteb_eval.py::test_map_score_uses_binary_relevance tests/test_mteb_eval.py::test_map_score_returns_zero_when_no_relevant_documents -q
```

Expected: fail with import error for `map_score`.

- [ ] **Step 3: Add `map_score`**

Append this function to `/Users/skiiwoo/Documents/New project 2/src/ranksmith/_metrics.py`.

```python
def map_score(ranking: Sequence[str], qrels: Mapping[str, int]) -> float:
    relevant_documents = {
        document_id for document_id, score in qrels.items() if score > 0
    }
    if not relevant_documents:
        return 0.0

    precision_sum = 0.0
    relevant_seen = 0
    for rank, document_id in enumerate(ranking, start=1):
        if document_id in relevant_documents:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    return precision_sum / len(relevant_documents)
```

- [ ] **Step 4: Verify MAP tests pass**

Run:

```bash
uv run pytest tests/test_mteb_eval.py::test_map_score_uses_binary_relevance tests/test_mteb_eval.py::test_map_score_returns_zero_when_no_relevant_documents -q
```

Expected: `2 passed`.

---

### Task 2: Private Evaluation Core

**Files:**
- Create: `/Users/skiiwoo/Documents/New project 2/src/ranksmith/_mteb_eval.py`
- Modify: `/Users/skiiwoo/Documents/New project 2/tests/test_mteb_eval.py`

- [ ] **Step 1: Write failing tests for method parsing, strict parsing, and metrics**

Add these tests to `/Users/skiiwoo/Documents/New project 2/tests/test_mteb_eval.py`.

```python
from ranksmith._mteb_eval import (
    MtebRerankingCandidate,
    MtebRerankingSample,
    apply_integer_permutation,
    compute_query_metrics,
    normalize_method_name,
    parse_ranking_with_failure_type,
    rankgpt_window_ranges,
)


def test_normalize_method_name_accepts_sliding_alias() -> None:
    assert normalize_method_name("sliding@20") == "rankgpt_sliding_window@20"
    assert normalize_method_name("rankgpt_sliding_window@50") == (
        "rankgpt_sliding_window@50"
    )


def test_parse_ranking_with_failure_type_reports_duplicate() -> None:
    parsed = parse_ranking_with_failure_type('{"ranking": [1, 1, 2]}', 3)

    assert not parsed.valid
    assert parsed.failure_type == "duplicate_rank"


def test_apply_integer_permutation_is_one_based() -> None:
    candidates = ("a", "b", "c")

    assert apply_integer_permutation(candidates, (3, 1, 2)) == ("c", "a", "b")


def test_rankgpt_window_ranges_do_not_repeat_prefix_window() -> None:
    assert rankgpt_window_ranges(
        document_count=20,
        rank_start=0,
        rank_end=20,
        window_size=20,
        step=10,
    ) == ((0, 20),)


def test_compute_query_metrics_uses_zero_score_for_invalid_result() -> None:
    sample = MtebRerankingSample(
        task_name="task",
        split="test",
        query_id="q1",
        query="query",
        candidates=(
            MtebRerankingCandidate(doc_id="d1", text="a", label=1.0),
            MtebRerankingCandidate(doc_id="d2", text="b", label=0.0),
        ),
    )

    metrics = compute_query_metrics(sample=sample, ranked_doc_ids=None, valid=False)

    assert metrics == {"ndcg@10": 0.0, "mrr@10": 0.0, "map": 0.0, "recall@10": 0.0}
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
uv run pytest tests/test_mteb_eval.py -q
```

Expected: fail because `ranksmith._mteb_eval` does not exist.

- [ ] **Step 3: Implement private core**

Create `/Users/skiiwoo/Documents/New project 2/src/ranksmith/_mteb_eval.py` with these public-private helpers.

```python
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from ranksmith._metrics import map_score, mrr_at_k, ndcg_at_k, recall_at_k

T = TypeVar("T")


@dataclass(frozen=True)
class MtebRerankingCandidate:
    doc_id: str
    text: str
    label: float


@dataclass(frozen=True)
class MtebRerankingSample:
    task_name: str
    split: str
    query_id: str
    query: str
    candidates: tuple[MtebRerankingCandidate, ...]


@dataclass(frozen=True)
class ParsedRanking:
    ranking: tuple[int, ...]
    valid: bool
    failure_type: str | None


def normalize_method_name(method: str) -> str:
    if method == "original":
        return method
    if method.startswith("direct@"):
        _parse_positive_suffix(method, "direct@")
        return method
    if method.startswith("sliding@"):
        rank_end = _parse_positive_suffix(method, "sliding@")
        return f"rankgpt_sliding_window@{rank_end}"
    if method.startswith("rankgpt_sliding_window@"):
        _parse_positive_suffix(method, "rankgpt_sliding_window@")
        return method
    raise ValueError(
        "method must be original, direct@N, sliding@N, or "
        "rankgpt_sliding_window@N"
    )


def parse_ranking_with_failure_type(
    raw_response: str,
    expected_count: int,
) -> ParsedRanking:
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        return ParsedRanking((), False, "json_parse_failure")
    if not isinstance(data, dict) or "ranking" not in data:
        return ParsedRanking((), False, "missing_ranking")
    ranking = data["ranking"]
    if not isinstance(ranking, list):
        return ParsedRanking((), False, "missing_ranking")
    if not all(isinstance(item, int) for item in ranking):
        integer_items = tuple(item for item in ranking if isinstance(item, int))
        return ParsedRanking(integer_items, False, "non_integer_rank")
    parsed = tuple(ranking)
    if len(parsed) != expected_count:
        return ParsedRanking(parsed, False, "length_mismatch")
    if len(set(parsed)) != len(parsed):
        return ParsedRanking(parsed, False, "duplicate_rank")
    expected = set(range(1, expected_count + 1))
    actual = set(parsed)
    if any(rank < 1 or rank > expected_count for rank in parsed):
        return ParsedRanking(parsed, False, "out_of_range_rank")
    if actual != expected:
        return ParsedRanking(parsed, False, "missing_rank")
    return ParsedRanking(parsed, True, None)


def apply_integer_permutation(
    items: Sequence[T],
    ranking: Sequence[int],
) -> tuple[T, ...]:
    return tuple(items[index - 1] for index in ranking)


def rankgpt_window_ranges(
    *,
    document_count: int,
    rank_start: int,
    rank_end: int,
    window_size: int,
    step: int,
) -> tuple[tuple[int, int], ...]:
    if document_count < 1:
        return ()
    if rank_start < 0 or rank_end < 1 or window_size < 1 or step < 1:
        raise ValueError("rank_start, rank_end, window_size, and step are invalid")
    if step > window_size:
        raise ValueError("step must be less than or equal to window_size")

    effective_end = min(rank_end, document_count)
    ranges: list[tuple[int, int]] = []
    end = effective_end
    while True:
        start = max(rank_start, end - window_size)
        ranges.append((start, end))
        if start == rank_start:
            return tuple(ranges)
        end -= step


def compute_query_metrics(
    *,
    sample: MtebRerankingSample,
    ranked_doc_ids: Sequence[str] | None,
    valid: bool,
) -> dict[str, float]:
    if not valid or ranked_doc_ids is None:
        return {"ndcg@10": 0.0, "mrr@10": 0.0, "map": 0.0, "recall@10": 0.0}
    qrels = {candidate.doc_id: int(candidate.label) for candidate in sample.candidates}
    binary_qrels = {
        candidate.doc_id: 1 if candidate.label > 0 else 0
        for candidate in sample.candidates
    }
    return {
        "ndcg@10": ndcg_at_k(ranked_doc_ids, qrels, 10),
        "mrr@10": mrr_at_k(ranked_doc_ids, binary_qrels, 10),
        "map": map_score(ranked_doc_ids, binary_qrels),
        "recall@10": recall_at_k(ranked_doc_ids, binary_qrels, 10),
    }


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil((percentile_value / 100) * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _parse_positive_suffix(method: str, prefix: str) -> int:
    suffix = method.removeprefix(prefix)
    if suffix == "":
        raise ValueError(f"{method} is missing a numeric suffix")
    value = int(suffix)
    if value < 1:
        raise ValueError(f"{method} suffix must be greater than 0")
    return value
```

- [ ] **Step 4: Run core tests**

Run:

```bash
uv run pytest tests/test_mteb_eval.py -q
```

Expected: all tests in `tests/test_mteb_eval.py` pass.

---

### Task 3: Provider Instrumentation and CLI Skeleton

**Files:**
- Modify: `/Users/skiiwoo/Documents/New project 2/src/ranksmith/_mteb_eval.py`
- Create: `/Users/skiiwoo/Documents/New project 2/scripts/evaluate_mteb_reranking.py`
- Modify: `/Users/skiiwoo/Documents/New project 2/tests/test_mteb_eval.py`

- [ ] **Step 1: Write failing tests for cost summary**

Add these tests to `/Users/skiiwoo/Documents/New project 2/tests/test_mteb_eval.py`.

```python
from ranksmith._mteb_eval import PriceConfig, estimate_cost


def test_estimate_cost_requires_usage_and_prices() -> None:
    assert estimate_cost(None, PriceConfig(1.0, 2.0)) is None
    assert estimate_cost((1000, 500), None) is None


def test_estimate_cost_uses_price_per_million_tokens() -> None:
    assert estimate_cost((1_000_000, 500_000), PriceConfig(2.0, 8.0)) == 6.0
```

- [ ] **Step 2: Add price helpers**

Append to `/Users/skiiwoo/Documents/New project 2/src/ranksmith/_mteb_eval.py`.

```python
@dataclass(frozen=True)
class PriceConfig:
    input_token_price_per_1m: float
    output_token_price_per_1m: float


def estimate_cost(
    usage: tuple[int, int] | None,
    price_config: PriceConfig | None,
) -> float | None:
    if usage is None or price_config is None:
        return None
    input_tokens, output_tokens = usage
    return (
        input_tokens / 1_000_000 * price_config.input_token_price_per_1m
        + output_tokens / 1_000_000 * price_config.output_token_price_per_1m
    )
```

- [ ] **Step 3: Create CLI skeleton**

Create `/Users/skiiwoo/Documents/New project 2/scripts/evaluate_mteb_reranking.py` with argument parsing, live guard, task listing, and schema inspection stubs.

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


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

    if not args.allow_live and any(method != "original" for method in args.methods):
        raise SystemExit("Refusing live Azure calls without --allow-live.")

    print(
        "Strict validation is enabled.\n"
        "Invalid LLM outputs are not repaired for main metrics.\n"
        "Invalid query-method results will receive zero scores.\n"
        "Repaired metrics, if enabled, are diagnostic only.",
        file=sys.stderr,
    )

    return


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
    parser.add_argument("--include-repaired-diagnostics", action="store_true")
    parser.add_argument("--input-token-price-per-1m", type=float)
    parser.add_argument("--output-token-price-per-1m", type=float)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--inspect-task-schema")
    return parser.parse_args()


def _list_tasks() -> None:
    try:
        import mteb
    except ModuleNotFoundError as exc:
        raise SystemExit("mteb is not installed. Install the optional MTEB dependency.") from exc
    tasks = mteb.get_tasks(task_types=["Reranking"], languages=["eng"])
    print(json.dumps([getattr(task, "metadata", task).__dict__ for task in tasks], default=str))


def _inspect_task_schema(task_name: str) -> None:
    try:
        import mteb
    except ModuleNotFoundError as exc:
        raise SystemExit("mteb is not installed. Install the optional MTEB dependency.") from exc
    task = mteb.get_task(task_name)
    task.load_data()
    print(json.dumps({"task": task_name, "dataset": str(task.dataset)}, default=str))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI validation checks**

Run:

```bash
uv run python scripts/evaluate_mteb_reranking.py --methods direct@20
```

Expected: exits with `Refusing live Azure calls without --allow-live.`

---

### Task 4: Full Runner and Reports

**Files:**
- Modify: `/Users/skiiwoo/Documents/New project 2/src/ranksmith/_mteb_eval.py`
- Modify: `/Users/skiiwoo/Documents/New project 2/scripts/evaluate_mteb_reranking.py`
- Modify: `/Users/skiiwoo/Documents/New project 2/tests/test_mteb_eval.py`

- [ ] **Step 1: Add tests for output writing and resume**

Add tests that write a synthetic `query_results.jsonl`, read completed `(task, split, query_id, method)` keys, and verify `--resume` would skip them. Use `tmp_path`.

```python
from pathlib import Path

from ranksmith._mteb_eval import completed_result_keys, write_jsonl


def test_completed_result_keys_reads_query_results(tmp_path: Path) -> None:
    path = tmp_path / "query_results.jsonl"
    write_jsonl(
        path,
        [
            {
                "task": "Task",
                "split": "test",
                "query_id": "q1",
                "method": "direct@20",
            }
        ],
    )

    assert completed_result_keys(path) == {("Task", "test", "q1", "direct@20")}
```

- [ ] **Step 2: Implement JSONL helpers**

Append to `/Users/skiiwoo/Documents/New project 2/src/ranksmith/_mteb_eval.py`.

```python
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def completed_result_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str, str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        row = json.loads(line)
        completed.add(
            (
                str(row["task"]),
                str(row["split"]),
                str(row["query_id"]),
                str(row["method"]),
            )
        )
    return completed
```

- [ ] **Step 3: Implement runner incrementally**

Expand `scripts/evaluate_mteb_reranking.py` to:
- Load `.env`.
- Validate output directory with `--overwrite` and `--resume`.
- Normalize method names.
- Load supported MTEB task rows.
- Convert rows to `MtebRerankingSample`.
- Run `original`, `direct@N`, and `rankgpt_sliding_window@N`.
- Write `query_results.jsonl`, `task_summary.json`, `overall_summary.json`, `metadata.json`, and `result_tables.md`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/test_mteb_eval.py -q
uv run ruff check src/ranksmith/_mteb_eval.py scripts/evaluate_mteb_reranking.py tests/test_mteb_eval.py
uv run mypy src tests
```

Expected: all pass.

---

### Task 5: Dependencies, Ignore Rules, and Documentation

**Files:**
- Modify: `/Users/skiiwoo/Documents/New project 2/pyproject.toml`
- Modify: `/Users/skiiwoo/Documents/New project 2/.gitignore`
- Modify: `/Users/skiiwoo/Documents/New project 2/README.md`
- Modify: `/Users/skiiwoo/Documents/New project 2/docs/specs/spec_mteb_reranking_evaluation.md`

- [ ] **Step 1: Add dependency group**

Modify `/Users/skiiwoo/Documents/New project 2/pyproject.toml` so `mteb` is not a core runtime dependency.

```toml
[dependency-groups]
dev = [
    "mypy>=1.8",
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
    "mteb>=1.0",
]
```

If dependency resolution shows a current incompatible MTEB version, pin the lowest compatible version found by `uv add --dev mteb`.

- [ ] **Step 2: Ignore local benchmark artifacts**

Add to `/Users/skiiwoo/Documents/New project 2/.gitignore`.

```gitignore
/benchmark-results/
```

- [ ] **Step 3: Document reference benchmark usage**

Add a README section titled `MTEB Reranking Reference Evaluation` with:

```markdown
These results are intended as practical reference points, not a universal ranking.
Results depend on dataset, model, candidate count, latency budget, and invalid output rate.
This benchmark measures reranking over fixed native MTEB candidate sets, not first-stage retrieval.
```

Include command:

```bash
uv run python scripts/evaluate_mteb_reranking.py \
  --tasks AskUbuntuDupQuestions SciDocsRR StackOverflowDupQuestions \
  --methods original direct@20 rankgpt_sliding_window@20 rankgpt_sliding_window@100 \
  --output-dir benchmark-results/mteb-reranking/example \
  --max-queries 50 \
  --max-document-chars 4000 \
  --input-token-price-per-1m 2.50 \
  --output-token-price-per-1m 10.00 \
  --allow-live
```

- [ ] **Step 4: Update spec checklist**

In `/Users/skiiwoo/Documents/New project 2/docs/specs/spec_mteb_reranking_evaluation.md`, check completed tasks and leave final `Completed` unchecked until `./scripts/verify.sh` passes.

---

### Task 6: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run full verification**

Run:

```bash
./scripts/verify.sh
```

Expected:
- `pytest` passes.
- `ruff check` passes.
- `ruff format --check` passes.
- `mypy src` passes.
- `uv build` succeeds.

- [ ] **Step 2: Run CLI non-live checks**

Run:

```bash
uv run python scripts/evaluate_mteb_reranking.py --methods original
uv run python scripts/evaluate_mteb_reranking.py --methods direct@20
```

Expected:
- `original` path does not require `--allow-live`.
- LLM method without `--allow-live` fails clearly.

- [ ] **Step 3: Optional live smoke**

Run only when credentials and cost are approved:

```bash
uv run python scripts/evaluate_mteb_reranking.py \
  --tasks AskUbuntuDupQuestions \
  --methods original direct@20 rankgpt_sliding_window@20 \
  --output-dir benchmark-results/mteb-reranking/live-smoke \
  --max-queries 1 \
  --allow-live
```

Expected:
- Output directory contains `query_results.jsonl`, `task_summary.json`, `overall_summary.json`, `metadata.json`, `result_tables.md`.
- `result_tables.md` displays strict validation and zero-score policy.

---

## Self-Review

- Spec coverage: covers native MTEB scope, strict validation, zero-score policy, method naming, max document length, usage/cost, latency summary, schema inspection, output files, README caveats.
- Red-flag scan: no incomplete markers are intentionally left in this plan.
- Type consistency: method names normalize to `rankgpt_sliding_window@N`; sample/candidate/result structures are reused across tests and runner.
