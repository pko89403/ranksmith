# ranksmith

<p align="center">
  <img src="https://raw.githubusercontent.com/pko89403/ranksmith/main/assets/ranksmith-icon.png" alt="ranksmith icon" width="160">
</p>

Forge better rankings from candidate documents.

[한국어 문서](README.ko.md)

`ranksmith` is a small Python package for LLM-based reranking. Version 1 focuses
on Azure OpenAI powered zero-shot reranking for candidate documents.

## Install

```bash
pip install ranksmith
```

## Quick Start

```python
from ranksmith import AzureOpenAIReranker, Document

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
)

results = reranker.rerank(
    query="What is listwise reranking?",
    documents=[
        Document(id="a", text="Listwise reranking compares candidates together."),
        Document(id="b", text="Vector search retrieves candidate documents."),
    ],
    top_k=2,
)

for result in results:
    print(result.rank, result.original_index, result.document.id)
```

`rank` is 1-based for display. `original_index` is 0-based so it maps back to
the input list.

## Supported Strategies & Algorithms

`ranksmith` separates the evaluation methodology (Strategy) from its execution
logic (Algorithm).

### Recommended Use Cases

| Method | Strategy | Recommended when | Trade-off | Details |
| --- | --- | --- | --- | --- |
| `rankgpt_sliding_window` | `ListwiseStrategy` | You need the default, lowest-friction LLM reranker for production or evaluation. | Low call count, but each prompt asks for a full ordered list and can be sensitive to output format. | [RankGPT listwise](#rankgpt-listwise) |
| `prp_sliding_k` | `PairwiseStrategy` | You need pairwise preference comparisons or want to reproduce PRP-style behavior. | Many LLM calls; default `passes=10` is expensive. | [PRP pairwise](#prp-pairwise) |
| `tourrank_r`, `rounds=2` | `TourRankStrategy` | You want stronger quality than listwise on a moderate call budget. | More calls than RankGPT, much fewer than TourRank-10. | [TourRank-r](#tourrank-r) |
| `tourrank_r`, `rounds=10` | `TourRankStrategy` | You are doing quality-focused offline reranking, paper-style evaluation, or final reranking where latency is acceptable. | Highest call cost among built-in methods in normal use. | [TourRank-r](#tourrank-r) |
| Custom strategy | `RerankStrategy` / `AsyncRerankStrategy` | You need deterministic business logic, a proprietary ranking process, or a new research method. | You own the ranking contract and validation behavior. | [Custom Strategies](#custom-strategies) |

### Strategy Details

#### RankGPT Listwise

`ListwiseStrategy` places multiple documents into one prompt and asks the LLM to
rank them together.

`rankgpt_sliding_window` is the default algorithm. It implements the
RankGPT-style back-to-first sliding window with bubble-up behavior while keeping
ranksmith's strict JSON output validation.

#### PRP Pairwise

`PairwiseStrategy` compares two documents at a time using Pairwise Ranking
Prompting.

`prp_sliding_k` starts from the bottom of the current ranking and compares
adjacent pairs. It calls the provider twice per pair, swapping A/B order to
reduce position bias. Conflicting valid comparisons are treated as ties and keep
the current order.

Default `passes=10`, matching the PRP-Sliding-10 setting from the reference
paper. Expected provider calls per query:
`2 * passes * max(document_count - 1, 0)`.

`AsyncPairwiseStrategy` can run each pair's A/B and B/A calls concurrently with
`pair_order_parallelism=2` without changing PRP traversal or call count.

#### TourRank-r

`TourRankStrategy` treats candidate documents as tournament participants. In
each stage, the provider selects the top-`m` documents from each group; selected
documents advance and earn points. The final ranking is sorted by accumulated
points.

`rounds=2` is the default practical setting. Prefer `rounds=10` for
quality-focused evaluation, paper-style reproduction, or final offline
reranking when the extra LLM calls are acceptable.

Default stages assume exactly 100 candidate documents:
`100 -> 50 -> 20 -> 10 -> 5 -> 2`. For other candidate counts, pass explicit
`stage_configs`; ranksmith fast fails instead of silently deriving or trimming
stages.

`TourRankStrategy` defaults to `group_parallelism=1` for serial sync calls.
Increase it to run groups in the same stage concurrently. If one parallel group
fails, already-started group calls may still finish.

`AsyncTourRankStrategy` runs groups concurrently by default. Set
`group_parallelism` to cap concurrent provider calls.

### How to Apply a Strategy

You can configure and inject a custom strategy into the `AzureOpenAIReranker`.

```python
from ranksmith import AzureOpenAIReranker, ListwiseStrategy, PairwiseStrategy

# 1. Configure the strategy and algorithm
strategy = ListwiseStrategy(
    algorithm="rankgpt_sliding_window",
    window_size=20,             # Number of documents evaluated at once
    stride=10,                  # Number of overlapping documents between windows
    max_document_chars=4000,    # Max characters allowed per document
)

# 2. Inject into the Reranker
reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=strategy, # <-- Inject the strategy here
)

results = reranker.rerank("query", documents)
```

Pairwise PRP can be injected the same way:

```python
strategy = PairwiseStrategy(
    algorithm="prp_sliding_k",
    passes=10,
    max_document_chars=4000,
)

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=strategy,
)
```

TourRank-r can also be injected:

```python
from ranksmith import AzureOpenAIReranker, TourRankStrategy

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=TourRankStrategy(rounds=2, group_parallelism=1),
)
```

For quality-focused runs, explicitly switch to TourRank-10:

```python
reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=TourRankStrategy(rounds=10),
)
```

> **Note**: If `strategy` is not provided, it defaults to `ListwiseStrategy(algorithm="rankgpt_sliding_window")`. Pairwise PRP and TourRank-r use more LLM calls than listwise reranking, so check call estimates before live benchmarks.

## Custom Strategies

Custom reranking methods should be implemented as new strategy classes instead
of adding new string values to `ListwiseStrategy.algorithm`. A strategy receives
the normalized `Document` objects, a provider, and optional `top_k`, then returns
`RerankResult` objects.

```python
from collections.abc import Sequence

from ranksmith import (
    AzureOpenAIReranker,
    Document,
    RerankResult,
)


class LengthStrategy:
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: object,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del query, provider
        ordered_indexes = sorted(
            range(len(documents)),
            key=lambda index: len(documents[index].text),
            reverse=True,
        )
        results = [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={"strategy": "length"},
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]
        return results if top_k is None else results[:top_k]


reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=LengthStrategy(),
)
```

A custom strategy can also use the public provider protocols directly.

```python
from collections.abc import Sequence

from ranksmith import (
    Document,
    LLMProvider,
    RerankResult,
    parse_ranking_response,
)


class ProviderBackedStrategy:
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: LLMProvider,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        ranking = parse_ranking_response(
            provider.rank(query, list(documents)),
            expected_count=len(documents),
        )
        ordered_indexes = [number - 1 for number in ranking]
        results = [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={"strategy": "provider-backed"},
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]
        return results if top_k is None else results[:top_k]
```

Async strategies use the same contract with `async def rerank(...)` and can be
typed with `AsyncRerankStrategy`. If a custom strategy fails with an unexpected
exception, `AzureOpenAIReranker` wraps it as `RerankStrategyError`. Raise
`RerankError` subclasses directly when the error category matters.

See [`examples/custom_strategy.py`](examples/custom_strategy.py) for a runnable
offline example that covers deterministic strategies, provider-backed
strategies, strict ranking parsing, and provider error classification.

For lower PRP wall time, use the async strategy. This preserves the
PRP-Sliding-K method: adjacent pairs are still processed bottom-to-top, while
only the two order-swapped calls for the same pair are concurrent.

```python
from ranksmith import AsyncAzureOpenAIReranker, AsyncPairwiseStrategy

reranker = AsyncAzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=AsyncPairwiseStrategy(
        passes=10,
        pair_order_parallelism=2,
    ),
)
```

## Async Support

`ranksmith` provides first-class asynchronous support for high-throughput environments like FastAPI.

```python
from ranksmith import AsyncAzureOpenAIReranker

reranker = AsyncAzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
)

results = await reranker.rerank("query", documents)
```

## Examples

Ready-to-use example code for integrating the **RankGPT** algorithm into your production environment can be found in the `examples/` directory.

- [`examples/rankgpt_sync.py`](examples/rankgpt_sync.py): Synchronous RankGPT integration guide
- [`examples/rankgpt_async.py`](examples/rankgpt_async.py): High-performance asynchronous RankGPT integration guide

## Benchmarking

`ranksmith` includes a qrels-backed comparison runner for reranking algorithms. It
can run against the committed smoke fixture or a local BEIR/SciFact cache. BEIR
mode requires a first-stage candidate TSV, because qrels alone are not a valid
reranking benchmark.

Expected BEIR/SciFact cache layout:

```text
.benchmark-cache/scifact/
  corpus.jsonl
  queries.jsonl
  qrels/test.tsv
```

Candidate TSV rows must start with `query_id` and `document_id`:

```text
query_id    document_id    rank
```

Run a live Azure comparison and write a JSON artifact:

```bash
python scripts/compare_reranking.py \
  --dataset beir-scifact \
  --cache-dir .benchmark-cache/scifact \
  --split test \
  --candidates path/to/candidates.tsv \
  --algorithm all \
  --top-k 10 \
  --window-size 20 \
  --stride 10 \
  --output benchmark-results/scifact.json \
  --allow-live
```

The JSON report includes per-query metrics and macro-averaged NDCG@k, MRR@k,
and Recall@k. Raw benchmark artifacts are intentionally ignored by git; publish
only reviewed summaries. The committed smoke fixture currently verifies the
deterministic offline RankGPT path at NDCG@3, MRR@3, and Recall@3 = `1.000`.

### Call accounting

`compare_reranking.py` estimates and prints the number of live LLM reranking
calls before execution. The count depends on the number of benchmark cases, the
selected algorithms, `window_size`, `stride`, `passes`, and candidate count per query:

- `rankgpt_sliding_window`: one LLM call per back-to-front RankGPT window.
- `prp_sliding_k`: `2 * passes * max(document_count - 1, 0)` pairwise LLM calls per query.
- `tourrank_r`: `tourrank_rounds * sum(stage.group_count)` selection LLM
  calls per query. The runner uses the paper top-100 stages for exactly 100
  candidates, and an explicit single-group halving stage plan for other
  candidate counts. With the paper top-100 stages, TourRank-2 uses 26 calls
  per query and TourRank-10 uses 130 calls per query.

The runner does **not** create first-stage candidates, embeddings, or
communities. If your candidate TSV is produced by an upstream retrieval or
community-building pipeline, account for those calls separately. A typical full
pipeline has two cost surfaces:

1. Candidate generation: embedding calls for corpus/query vectors, plus any LLM
   calls used to create or summarize communities.
2. Reranking: LLM calls made by `ranksmith` for the selected reranking
   algorithms.

Benchmark summaries should report both numbers when community retrieval is part
of the experiment, for example: `embedding calls=<n>`, `community LLM calls=<n>`,
and `reranking LLM calls=<n>`.

## Result Model

```python
result.document        # Document
result.rank            # 1-based rank
result.original_index  # 0-based input index
result.metadata        # strategy-specific metadata
```

## Error Handling

`ranksmith` fails fast. It does not silently truncate long documents, repair
invalid rankings, or return unvalidated LLM output.

```python
from ranksmith import (
    DocumentTooLongError,
    RerankParseError,
    RerankProviderError,
    RerankStrategyError,
)

try:
    results = reranker.rerank("query", documents)
except DocumentTooLongError:
    ...
except RerankParseError:
    ...
except RerankProviderError:
    ...
except RerankStrategyError:
    ...
```

## MTEB Reranking Reference Evaluation

This benchmark measures reranking only. It uses the fixed native MTEB candidate
sets, not first-stage retrieval results.

```bash
UV_NATIVE_TLS=true uv run python scripts/evaluate_mteb_reranking.py \
  --tasks AskUbuntuDupQuestions \
  --methods \
    original \
    rankgpt_sliding_window@20 \
    prp_sliding_k@20:p1 \
    tourrank_r@20:r2 \
    tourrank_r@20:r10 \
  --output-dir benchmark-results/mteb-reranking/askubuntu-full-tourrank-prp-20260520 \
  --max-document-chars 4000 \
  --shuffle-candidates --shuffle-seed 13 \
  --rankgpt-window-size 20 --rankgpt-step 10 \
  --concurrency 24 \
  --retry-invalid-outputs 1 \
  --input-token-price-per-1m 2.50 \
  --output-token-price-per-1m 10.00 \
  --allow-live
```

Run scope:

- Dataset: `AskUbuntuDupQuestions`, `test` split
- Queries: `361`
- Candidates: MTEB-provided `top_ranked` candidates, `20` per query
- Candidate order: shuffled with seed `13`
- Model: Azure OpenAI deployment `gpt-5.4-nano`
- Validation: strict JSON; invalid outputs are zero-scored
- Resume policy: failed rows were retried with `--resume --retry-failed-results`
- Artifact: `benchmark-results/mteb-reranking/askubuntu-full-tourrank-prp-20260520`

| Method | NDCG@10 | MRR@10 | MAP | Recall@10 | p50 latency | Invalid rate | LLM calls/query | Total calls | Queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | 0.3926 | 0.4594 | 0.3711 | 0.4993 | 0.0 ms | 0.000 | 0.0 | 0 | 361 |
| `rankgpt_sliding_window@20` | 0.6908 | 0.7470 | 0.6355 | 0.7671 | 1820.5 ms | 0.008 | 1.0 | 374 | 361 |
| `tourrank_r@20:r2` | 0.7023 | 0.7642 | 0.6421 | 0.7785 | 8297.1 ms | 0.000 | 8.0 | 2,888 | 361 |
| `tourrank_r@20:r10` | 0.7135 | 0.7734 | 0.6597 | 0.7836 | 39026.4 ms | 0.006 | 39.9 | 14,409 | 361 |

`tourrank_r@20:r10` had the strongest scores in this run, while
`tourrank_r@20:r2` stayed close with far fewer calls and lower latency. Full
`prp_sliding_k@20` with the default `passes=10` was not run in this full-query
benchmark; it would require `380` calls/query (`137,180` calls over all 361
queries), so no quality or latency metrics are reported for that setting here.

The auxiliary `prp_sliding_k@20:p1` run completed over the same 361 queries only
as a call-budget reference near `tourrank_r@20:r10`: NDCG@10 `0.5360`, MRR@10
`0.7261`, MAP `0.4983`, Recall@10 `0.5773`, p50 latency `19919.1 ms`,
invalid rate `0.000`, `38.0` calls/query, `13,718` total calls.
