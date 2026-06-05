# ranksmith

<p align="center">
  <img src="https://raw.githubusercontent.com/pko89403/ranksmith/main/assets/ranksmith-icon.png" alt="ranksmith icon" width="160">
</p>

Forge better rankings from candidate documents.

[한국어 문서](https://github.com/pko89403/ranksmith/blob/main/README.ko.md)

`ranksmith` is a small Python package for LLM-based reranking. The current
package focuses on Azure OpenAI powered zero-shot reranking for candidate
documents.

Highlights:

- Built-in listwise RankGPT, pairwise PRP, tournament-style TourRank-r, and
  uncertainty-aware AcuRank strategies
- Public strategy contracts for custom reranking methods
- `ModelClient` / `ModelProvider` boundary for vendor-independent LLM calls
- Strict JSON parsing and fast-fail error behavior
- Sync and async Azure OpenAI rerankers
- Reproducible benchmark summaries with committed evidence artifacts

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

| Method | Strategy | Use when | Cost / risk |
| --- | --- | --- | --- |
| `rankgpt_sliding_window` | `ListwiseStrategy` | You need the default, lowest-friction LLM reranker for production or evaluation. | Low call count, but each prompt asks for a full ordered list and can be sensitive to output format. With `window_size >= N`, this becomes one-shot listwise reranking. |
| `prp_sliding_k` | `PairwiseStrategy` | You need pairwise preference comparisons or want to reproduce PRP-style behavior. | Many LLM calls; default `passes=10` is expensive. |
| `setwise_heapsort` | `SetwiseStrategy` | You want top-k-oriented setwise selection with fewer calls than pairwise PRP in practical long-context settings. | Quality depends on `set_size`; larger sets reduce calls but can make the selection prompt harder. |
| `tourrank_r`, `rounds=2` | `TourRankStrategy` | You want stronger quality than listwise on a moderate call budget. | More calls than RankGPT, much fewer than TourRank-10. |
| `tourrank_r`, `rounds=10` | `TourRankStrategy` | You are doing quality-focused offline reranking, paper-style evaluation, or final reranking where latency is acceptable. | Highest call cost among built-in methods in normal use. |
| `acurank` | `AcuRankStrategy` | You want adaptive listwise reranking that spends calls on uncertain candidates near the top-k boundary. | Uses TrueSkill state and may issue more calls than basic listwise reranking unless capped. |
| Custom strategy | `RerankStrategy` / `AsyncRerankStrategy` | You need deterministic business logic, a proprietary ranking process, or a new research method. | You own the ranking contract and validation behavior. |

### Applying a Strategy

Configure a strategy and pass it to `AzureOpenAIReranker`.

```python
from ranksmith import AzureOpenAIReranker, ListwiseStrategy

strategy = ListwiseStrategy(
    algorithm="rankgpt_sliding_window",
    window_size=20,
    stride=10,
    max_document_chars=4000,
)

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=strategy,
)

results = reranker.rerank("query", documents)
```

Pairwise PRP uses the same reranker facade with a different strategy:

```python
from ranksmith import AzureOpenAIReranker, PairwiseStrategy

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=PairwiseStrategy(passes=3),
)
```

TourRank-r uses the same injection point:

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

AcuRank uses listwise reranker calls as evidence for TrueSkill-based relevance
estimates:

```python
from ranksmith import AcuRankStrategy, AzureOpenAIReranker

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=AcuRankStrategy(
        target_rank=10,
        window_size=20,
        max_adaptive_reranker_calls=20,  # Optional adaptive-phase budget cap.
        batch_parallelism=2,  # Optional; keep 1 if your provider is not thread-safe.
    ),
)
```

If every `Document` has numeric `metadata["score"]`, AcuRank uses it as the
first-stage prior. If no document has a score, it falls back to the standard
TrueSkill prior. Partial score metadata and boolean score values fail fast.

For small candidate sets, `target_rank` is clipped to the number of documents.
`max_adaptive_reranker_calls` limits only the adaptive refinement phase; the
optional initial pass is counted separately in result metadata.
`batch_parallelism` parallelizes independent batches within the same AcuRank
iteration, while posterior updates are still applied in deterministic batch
order.

> **Note**: If `strategy` is not provided, it defaults to `ListwiseStrategy(algorithm="rankgpt_sliding_window")`. Pairwise PRP, Setwise, TourRank-r, and AcuRank can use more LLM calls than basic listwise reranking, so check call estimates before live benchmarks.

## Custom Strategies

Custom reranking methods should be implemented as new strategy classes instead
of adding new string values to `ListwiseStrategy.algorithm`. A strategy receives
the normalized `Document` objects, a model client, and optional `top_k`, then
returns `RerankResult` objects.

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
        model_client: object,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del query, model_client
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

Model-backed and async strategies use the same public contract. See
the [custom strategy extension guide](https://github.com/pko89403/ranksmith/blob/main/docs/wiki/08_custom_strategy_extension.md)
and [custom strategy example](https://github.com/pko89403/ranksmith/blob/main/examples/custom_strategy.py)
for the full extension guide.

## Model Provider Architecture

`ModelClient` owns ranksmith's domain prompts and `rank` / `compare` / `select`
contracts. `ModelProvider` only executes vendor-specific JSON completion
requests.

| Layer | Responsibility | Public methods |
| --- | --- | --- |
| `Strategy` | Build the final reranking order. | `rerank(...)` |
| `ModelClient` | Build ranksmith prompts, enforce the ranking domain contract, and emit usage. | `rank(...)`, `compare(...)`, `select(...)` |
| `ModelProvider` | Call a vendor SDK and return JSON completion text. | `complete(...)` |

```python
from ranksmith import AzureAOAIProvider, ModelClient

provider = AzureAOAIProvider(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    api_version="2024-08-01-preview",
)
model_client = ModelClient(provider=provider)
```

The same `ModelClient` can power all built-in strategies:

```python
from ranksmith import AzureOpenAIReranker, PairwiseStrategy

reranker = AzureOpenAIReranker(
    model_client=model_client,
    strategy=PairwiseStrategy(passes=3),
)
```

`OpenAIProvider`, `AnthropicProvider`, and `GeminiProvider` are reserved public
stubs for future SDK-backed implementations. Calling them fails fast with
`RerankProviderError`.

## Async Support

`ranksmith` provides first-class asynchronous support for high-throughput
environments like FastAPI.

```python
from ranksmith import AsyncAzureOpenAIReranker

reranker = AsyncAzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
)

results = await reranker.rerank("query", documents)
```

## Structural Confidence

`ranksmith.confidence` provides single-item and bounded batch sync confidence
inference for closed-model outputs using a frozen HuggingFace encoder,
`structural-v1` features, and a trained compatible scorer artifact.

Install optional dependencies:

```bash
pip install "ranksmith[confidence]"
```

```python
from ranksmith.confidence import (
    AnswerConfidenceInput,
    StructuralConfidenceEstimator,
)

estimator = StructuralConfidenceEstimator.from_artifact(
    "structural-confidence.joblib",
)

result = estimator.score(
    AnswerConfidenceInput(context="...", answer="...")
)
print(result.score)

batch_results = estimator.score_batch(
    [AnswerConfidenceInput(context="...", answer="...")],
    batch_size=8,
    max_workers=1,
)
```

This module does not train a scorer, does not add a reranking Strategy, and
does not perform async inference. Parallel batch scoring shares the same
encoder and scorer instances across worker threads, so use `max_workers>1` only
with thread-safe backends. It cancels pending work on the first worker error,
but Python threads that have already started may finish in the background.

`ranksmith.confidence_generation` can create supervised canonical JSONL for
confidence training by calling a closed model over raw answer or relevance
examples. It is a data-generation utility, not a reranking Strategy.

### Training a compatible confidence scorer

`ranksmith.confidence_training` can train a Phase 1-compatible scorer artifact
from supervised canonical JSONL. It does not generate labels, call closed
models, provide dataset adapters, or report reranking benchmark numbers.

Install training dependencies:

```bash
pip install "ranksmith[confidence-train]"
```

```python
from ranksmith.confidence_training import (
    ConfidenceTrainingConfig,
    train_confidence_scorer,
)

result = train_confidence_scorer(
    ConfidenceTrainingConfig(
        task_type="answer_confidence",
        dataset_path="answer_confidence.jsonl",
        output_dir="confidence-runs/answer-v1",
        export_path="artifacts/answer_confidence.joblib",
    )
)
print(result.export_path)
```

## Examples

Runnable examples live in the `examples/` directory.

- [rankgpt_sync.py](https://github.com/pko89403/ranksmith/blob/main/examples/rankgpt_sync.py): synchronous RankGPT integration
- [rankgpt_async.py](https://github.com/pko89403/ranksmith/blob/main/examples/rankgpt_async.py): async RankGPT integration
- [pairwise_prp.py](https://github.com/pko89403/ranksmith/blob/main/examples/pairwise_prp.py): pairwise PRP strategy
- [setwise_heapsort.py](https://github.com/pko89403/ranksmith/blob/main/examples/setwise_heapsort.py): Setwise Heapsort with a fake provider
- [tourrank.py](https://github.com/pko89403/ranksmith/blob/main/examples/tourrank.py): TourRank-r with a fake provider
- [acurank.py](https://github.com/pko89403/ranksmith/blob/main/examples/acurank.py): AcuRank with first-stage score priors
- [custom_strategy.py](https://github.com/pko89403/ranksmith/blob/main/examples/custom_strategy.py): custom strategy contracts

## Claude Code Advisor

This repo ships a [Claude Code](https://code.claude.com/docs) plugin,
`ranksmith-advisor`, that helps you choose a reranking strategy for your use
case and returns working, CI-verified snippets. It encodes ranksmith-specific
guardrails, so the suggested code follows the library's real contracts (no
calls into unimplemented providers, no `algorithm` string hacks, no treating
`confidence` as a reranker).

Use it from Claude Code:

```bash
/plugin marketplace add pko89403/ranksmith
/plugin install ranksmith-advisor@ranksmith
```

Repo contributors get it automatically: the project-shared
`.claude/settings.json` registers the local marketplace and enables the plugin,
so no manual install is needed. The plugin content lives under
`skills/ranksmith-advisor/` and is excluded from the PyPI distribution.

## Benchmarking

The benchmark below measures reranking only. Pyserini BM25 provides the fixed
first-stage candidates; `ranksmith` reranks those candidates without performing
retrieval. The run uses `AskUbuntuDupQuestions` test data: `361` queries, BM25
top-20 candidates per query, and `@5` evaluation. Methods that support top-k
early stopping may emit only the evaluated top-5. Azure OpenAI deployment
`gpt-5.4-nano` was used for live LLM calls.

Invalid LLM outputs were not repaired or silently corrected. They were retried,
and any remaining invalid rows are reported as invalid.

The table separates nominal algorithm call estimates from row-level retry
attempts. Row attempts are useful for retry accounting, but they are not exact
provider-call telemetry for multi-call methods that can fail partway through an
algorithm run. The committed evidence artifacts are:

- [`benchmark-results/live/askubuntu-bm25-top20-default-live.v3.merged.json`](https://github.com/pko89403/ranksmith/blob/main/benchmark-results/live/askubuntu-bm25-top20-default-live.v3.merged.json)
- [`benchmark-results/pyserini/askubuntu-bm25-top20.trec`](https://github.com/pko89403/ranksmith/blob/main/benchmark-results/pyserini/askubuntu-bm25-top20.trec)

| Method | NDCG@5 | MRR@5 | Recall@5 | Valid rows | Invalid rate | Nominal LLM calls/query | LLM row attempts/query incl. retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original_bm25` | 0.3520 | 0.5062 | 0.2862 | 361/361 | 0.000 | 0 | N/A |
| `single_call_listwise@20` | 0.4082 | 0.5541 | 0.3345 | 359/361 | 0.006 | 1 | 1.04 |
| `rankgpt_sw_w5` | 0.3973 | 0.5283 | 0.3366 | 361/361 | 0.000 | 9 | 1.01 |
| `acurank_k5_b1` | 0.4053 | 0.5491 | 0.3377 | 356/361 | 0.014 | 2 | 1.12 |
| `tourrank_r2` | 0.4236 | 0.5725 | 0.3601 | 361/361 | 0.000 | 8 | 1.03 |
| `setwise_hs_s10` | 0.3653 | 0.5059 | 0.3005 | 361/361 | 0.000 | 12 | 1.00 |
| `prp_sliding_p1` | 0.4065 | 0.5818 | 0.3277 | 361/361 | 0.000 | 38 | 1.00 |

`tourrank_r2` had the best NDCG@5 and Recall@5, while `prp_sliding_p1` had the
best MRR@5. `single_call_listwise@20` is the one-shot listwise baseline.
`rankgpt_sw_w5` is the true sliding-window listwise baseline for this top-20
setup. `acurank_k5_b1` aligns AcuRank's uncertainty boundary with the `@5`
evaluation cutoff. `setwise_hs_s10` is a practical Setwise Heapsort setting
that extracts only the evaluated top-5 from 20 candidates.

After retries, 2 `single_call_listwise@20` rows and 5 `acurank_k5_b1` rows
remained invalid. They are included in the invalid-rate accounting instead of
being repaired.

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
