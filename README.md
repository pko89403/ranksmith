# ranksmith

<p align="center">
  <img src="https://raw.githubusercontent.com/pko89403/ranksmith/main/assets/ranksmith-icon.png" alt="ranksmith icon" width="160">
</p>

Forge better rankings from candidate documents.

[한국어 문서](https://github.com/pko89403/ranksmith/blob/main/README.ko.md)

`ranksmith` is a small Python package for LLM-based reranking. Version 1 focuses
on Azure OpenAI powered zero-shot reranking for candidate documents.

Highlights:

- Built-in listwise RankGPT, pairwise PRP, and tournament-style TourRank-r strategies
- Public strategy contracts for custom reranking methods
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

| Method | Strategy | Recommended when | Trade-off |
| --- | --- | --- | --- |
| `rankgpt_sliding_window` | `ListwiseStrategy` | You need the default, lowest-friction LLM reranker for production or evaluation. | Low call count, but each prompt asks for a full ordered list and can be sensitive to output format. |
| `prp_sliding_k` | `PairwiseStrategy` | You need pairwise preference comparisons or want to reproduce PRP-style behavior. | Many LLM calls; default `passes=10` is expensive. |
| `tourrank_r`, `rounds=2` | `TourRankStrategy` | You want stronger quality than listwise on a moderate call budget. | More calls than RankGPT, much fewer than TourRank-10. |
| `tourrank_r`, `rounds=10` | `TourRankStrategy` | You are doing quality-focused offline reranking, paper-style evaluation, or final reranking where latency is acceptable. | Highest call cost among built-in methods in normal use. |
| Custom strategy | `RerankStrategy` / `AsyncRerankStrategy` | You need deterministic business logic, a proprietary ranking process, or a new research method. | You own the ranking contract and validation behavior. |

### Applying a Strategy

You can configure and inject a custom strategy into the `AzureOpenAIReranker`.

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

Provider-backed and async strategies use the same public contract. See
the [custom strategy extension guide](https://github.com/pko89403/ranksmith/blob/main/docs/wiki/08_custom_strategy_extension.md)
and [custom strategy example](https://github.com/pko89403/ranksmith/blob/main/examples/custom_strategy.py)
for the full extension guide.

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

Runnable examples live in the `examples/` directory.

- [rankgpt_sync.py](https://github.com/pko89403/ranksmith/blob/main/examples/rankgpt_sync.py): synchronous RankGPT integration
- [rankgpt_async.py](https://github.com/pko89403/ranksmith/blob/main/examples/rankgpt_async.py): async RankGPT integration
- [pairwise_prp.py](https://github.com/pko89403/ranksmith/blob/main/examples/pairwise_prp.py): pairwise PRP strategy
- [tourrank.py](https://github.com/pko89403/ranksmith/blob/main/examples/tourrank.py): TourRank-r with a fake provider
- [custom_strategy.py](https://github.com/pko89403/ranksmith/blob/main/examples/custom_strategy.py): custom strategy contracts

## Benchmarking

The reference benchmark below measures reranking only. It uses the fixed native
MTEB `AskUbuntuDupQuestions` test candidates: `361` queries, `20` candidates per
query, shuffled with seed `13`, using Azure OpenAI deployment `gpt-5.4-nano`.

Full command, call accounting, run scope, and artifact links are in
the [MTEB reranking benchmark notes](https://github.com/pko89403/ranksmith/blob/main/docs/benchmarks/mteb_reranking.md).

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

The auxiliary `prp_sliding_k@20:p1` result is documented in the benchmark
details as a reduced-budget call reference, not as the default PRP result.

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
