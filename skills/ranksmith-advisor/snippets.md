# Code snippets

Two sources, by audience:

- **In-repo / GitHub readers:** the full, runnable, CI-tested programs in
  `examples/` are the canonical reference (executed by `tests/test_examples.py`).
  Prefer adapting these.
- **Downstream users (plugin installed elsewhere):** the minimal starters below
  are copy-paste ready. They use only public `ranksmith` exports, which
  `tests/test_advisor_references.py` verifies.

## Scenario → canonical example

| Scenario | Example (tested) |
| --- | --- |
| Sync listwise (RankGPT) | `examples/rankgpt_sync.py` |
| Async listwise (servers) | `examples/rankgpt_async.py` |
| Pairwise PRP | `examples/pairwise_prp.py` |
| Setwise heapsort | `examples/setwise_heapsort.py` |
| TourRank-r | `examples/tourrank.py` |
| AcuRank + first-stage scores | `examples/acurank.py` |
| Custom strategy contracts | `examples/custom_strategy.py` |

GitHub: https://github.com/pko89403/ranksmith/tree/main/examples

## Minimal starters

### Sync quick start
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

### Swap the strategy
```python
from ranksmith import AzureOpenAIReranker, TourRankStrategy

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=TourRankStrategy(rounds=2),
)
```
Substitute `ListwiseStrategy`, `PairwiseStrategy`, `SetwiseStrategy`, or
`AcuRankStrategy`; see `method-guide.md` for parameters.

### Async
```python
from ranksmith import AsyncAzureOpenAIReranker

reranker = AsyncAzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
)
results = await reranker.rerank("query", documents)
```
Pair async rerankers with `Async*` strategies only.

### AcuRank with first-stage scores
```python
from ranksmith import AcuRankStrategy, AzureOpenAIReranker, Document

documents = [
    Document(id="a", text="...", metadata={"score": 12.5}),
    Document(id="b", text="...", metadata={"score": 9.1}),
]
reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=AcuRankStrategy(target_rank=10, window_size=20),
)
```
Either every document has a numeric `metadata["score"]` or none do.

### Custom strategy skeleton
```python
from collections.abc import Sequence

from ranksmith import Document, RerankResult


class MyStrategy:
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: object,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del query, model_client
        ordered = sorted(
            range(len(documents)),
            key=lambda i: len(documents[i].text),
            reverse=True,
        )
        results = [
            RerankResult(
                document=documents[i],
                rank=rank,
                original_index=i,
                metadata={"strategy": "my"},
            )
            for rank, i in enumerate(ordered, start=1)
        ]
        return results if top_k is None else results[:top_k]
```
Model-backed strategies must validate JSON with `parse_ranking_response()` and
classify provider errors as `RerankProviderError`; see `examples/custom_strategy.py`.

### Error handling
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

### Reuse one ModelClient across strategies
```python
from ranksmith import AzureAOAIProvider, AzureOpenAIReranker, ModelClient, PairwiseStrategy

provider = AzureAOAIProvider(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    api_version="2024-08-01-preview",
)
reranker = AzureOpenAIReranker(
    model_client=ModelClient(provider=provider),
    strategy=PairwiseStrategy(passes=3),
)
```
