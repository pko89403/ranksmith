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

## Strategy

The default strategy is listwise reranking with automatic sliding-window
handling for longer candidate lists.

```python
from ranksmith import ListwiseStrategy

strategy = ListwiseStrategy(
    algorithm="sliding_window",
    window_size=20,
    stride=10,
    max_document_chars=4000,
)
```

Version 1 supports `direct` and `sliding_window` algorithms. Pointwise,
pairwise, tournament, bayesian, and confidence-style algorithms are left for
future versions.

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
