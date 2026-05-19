# ranksmith

<p align="center">
  <img src="https://raw.githubusercontent.com/pko89403/ranksmith/main/assets/ranksmith-icon.png" alt="ranksmith icon" width="160">
</p>

Forge better rankings from candidate documents.

[한국어 문서](README.ko.md)

`ranksmith` is a small Python package for LLM-based reranking. Version 1 focuses
on Azure OpenAI powered zero-shot listwise reranking for candidate documents.

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

`ranksmith` separates the evaluation methodology (Strategy) from its specific execution logic (Algorithm). Version 1 supports listwise reranking and pairwise PRP reranking.

### 1. ListwiseStrategy (RankGPT)
This strategy places multiple documents into a single prompt and asks the LLM to rank them all at once.

- **`rankgpt_sliding_window` Algorithm (Default)**
  - Implements the RankGPT-style back-to-first sliding window with bubble-up behavior.
  - Useful when you want RankGPT's window traversal semantics while keeping ranksmith's strict JSON output validation.

### 2. PairwiseStrategy (PRP)
This strategy compares two documents at a time using Pairwise Ranking Prompting.

- **`prp_sliding_k` Algorithm**
  - Starts from the bottom of the current ranking and compares adjacent pairs.
  - Calls the provider twice per pair, swapping A/B order to reduce position bias.
  - Conflicting valid comparisons are treated as ties and keep the current order.
  - Default `passes=10`, matching the PRP-Sliding-10 setting from the reference paper.
  - Expected provider calls per query: `2 * passes * max(document_count - 1, 0)`.
  - `AsyncPairwiseStrategy` can run each pair's A/B and B/A calls concurrently with `pair_order_parallelism=2` without changing PRP traversal or call count.

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

> **Note**: If `strategy` is not provided, it defaults to `ListwiseStrategy(algorithm="rankgpt_sliding_window")`. Pairwise PRP uses many more LLM calls than listwise reranking, so check call estimates before live benchmarks.

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
from ranksmith import DocumentTooLongError, RerankParseError, RerankProviderError

try:
    results = reranker.rerank("query", documents)
except DocumentTooLongError:
    ...
except RerankParseError:
    ...
except RerankProviderError:
    ...
```

## MTEB Reranking Reference Evaluation

These results are intended as practical reference points, not a universal ranking.
Results depend on dataset, model, candidate count, latency budget, and invalid output rate.
This benchmark measures reranking over fixed native MTEB candidate sets, not first-stage retrieval.

```bash
uv run python scripts/evaluate_mteb_reranking.py \
  --tasks AskUbuntuDupQuestions SciDocsRR StackOverflowDupQuestions \
  --methods original rankgpt_sliding_window@20 prp_sliding_k@20 \
  --output-dir benchmark-results/mteb-reranking/example \
  --max-queries 50 \
  --max-document-chars 4000 \
  --shuffle-candidates --shuffle-seed 13 \
  --rankgpt-window-size 20 --rankgpt-step 10 \
  --prp-passes 10 \
  --concurrency 4 \
  --input-token-price-per-1m 2.50 \
  --output-token-price-per-1m 10.00 \
  --allow-live
```

### Current MTEB snapshot

The committed reference snapshot below is from
`benchmark-results/mteb-reranking/n30-ask-fixed`.

Scope:

- Task: `AskUbuntuDupQuestions`
- Split: `test`
- Queries: `30`
- Candidate order: shuffled with seed `13`
- Max document length: `4000` characters
- Validation: strict JSON validation, invalid outputs score `0`
- Measured methods: `original`, `rankgpt_sliding_window@20`

| Method | NDCG@10 | MRR@10 | MAP | Recall@10 | p50 latency | p95 latency | Invalid rate | Queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | 0.4431 | 0.5668 | 0.3895 | 0.5871 | 0.0 ms | 0.0 ms | 0.000 | 30 |
| `rankgpt_sliding_window@20` | 0.6825 | 0.6753 | 0.6424 | 0.7870 | 1953.3 ms | 2893.9 ms | 0.000 | 30 |

On this small snapshot, `rankgpt_sliding_window@20` improved NDCG@10 and
Recall@10 over the original candidate order. This is not a general claim about
all datasets; it is a smoke-sized reference result for this task and
configuration.

### PRP vs RankGPT Snapshot

The PRP comparison run below uses the same `AskUbuntuDupQuestions` setup and is
saved under `benchmark-results/mteb-reranking/n30-prp-vs-rankgpt-rerun`.
This is a native MTEB candidate-set benchmark: this task exposes 20 candidates
per query, so it is **not** the standard top-100 RankGPT setting.

| Method | NDCG@10 | MRR@10 | MAP | Recall@10 | p50 latency | p95 latency | Invalid rate | LLM calls/query | Total LLM calls | Mean cost/query | Queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | 0.4431 | 0.5668 | 0.3895 | 0.5871 | 0.0 ms | 0.0 ms | 0.000 | 0 | 0 | - | 30 |
| `rankgpt_sliding_window@20` | 0.6830 | 0.6834 | 0.6400 | 0.7706 | 1842.6 ms | 2542.6 ms | 0.033 | 1 | 30 | $0.001530 | 30 |
| `prp_sliding_k@20` | 0.6714 | 0.7837 | 0.6132 | 0.7451 | 213583.6 ms | 230670.9 ms | 0.000 | 380 | 11,400 | $0.172772 | 30 |

RankGPT listwise led on NDCG@10, MAP, Recall@10, latency, and cost. PRP led on
MRR@10, but it required about 380 pairwise LLM calls per query with `passes=10`
and 20 candidates. Strict validation is applied: the RankGPT row includes one
invalid LLM output scored as zero.

For the common top-100 RankGPT setup with `window_size=20` and `step=10`,
`rankgpt_sliding_window@100` would use 9 listwise LLM calls per query. The
matching `prp_sliding_k@100` setting would use
`2 * 10 * (100 - 1) = 1,980` pairwise LLM calls per query.
