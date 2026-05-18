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

`ranksmith` separates the evaluation methodology (Strategy) from its specific execution logic (Algorithm). Version 1 officially supports the **Listwise Reranking (RankGPT)** strategy.

### 1. ListwiseStrategy (RankGPT)
This strategy places multiple documents into a single prompt and asks the LLM to rank them all at once.

- **`direct` Algorithm**
  - Reranks all candidates in a single LLM API call.
  - Suitable when the number of documents is small enough to comfortably fit within the LLM's context window.
- **`sliding_window` Algorithm (Default)**
  - When there are too many documents, this algorithm chunks them by `window_size` and iteratively ranks them with an overlap of `stride`.
  - Essential for handling long candidate lists, preventing token limit exceedances, and avoiding the "lost in the middle" degradation in LLM ranking capabilities.
- **`rankgpt_sliding_window` Algorithm**
  - Implements the RankGPT-style back-to-first sliding window with bubble-up behavior.
  - Useful when you want RankGPT's window traversal semantics while keeping ranksmith's strict JSON output validation.

### How to Apply a Strategy

You can configure and inject a custom strategy into the `AzureOpenAIReranker`.

```python
from ranksmith import AzureOpenAIReranker, ListwiseStrategy

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

> **Note**: If `strategy` is not provided, it defaults to `ListwiseStrategy(algorithm="sliding_window")`. Version 1 supports `direct`, `sliding_window`, and `rankgpt_sliding_window`. Pointwise, Pairwise, and Tournament strategies are planned for future releases.

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
selected algorithms, `window_size`, `stride`, and candidate count per query:

- `direct`: one LLM call per query.
- `sliding_window`: one LLM call per evaluated window.
- `rankgpt_sliding_window`: one LLM call per back-to-front RankGPT window.

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
  --methods original direct@20 rankgpt_sliding_window@20 rankgpt_sliding_window@100 \
  --output-dir benchmark-results/mteb-reranking/example \
  --max-queries 50 \
  --max-document-chars 4000 \
  --shuffle-candidates --shuffle-seed 13 \
  --rankgpt-window-size 20 --rankgpt-step 10 \
  --input-token-price-per-1m 2.50 \
  --output-token-price-per-1m 10.00 \
  --allow-live
```
