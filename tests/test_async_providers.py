import pytest

from ranksmith import (
    AsyncAzureOpenAIReranker,
    AsyncListwiseStrategy,
    Document,
    RerankParseError,
    RerankProviderError,
)


class AsyncFakeProvider:
    def __init__(self, responses: list[str], fail: Exception | None = None) -> None:
        self.responses = responses
        self.fail = fail
        self.calls: list[list[str]] = []

    async def rank(self, query: str, documents: list[Document]) -> str:
        if self.fail is not None:
            raise self.fail
        self.calls.append([document.text for document in documents])
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_async_reranks_string_documents_and_preserves_indexes() -> None:
    provider = AsyncFakeProvider(['{"ranking": [3, 1, 2]}'])
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )

    results = await reranker.rerank("query", ["alpha", "beta", "gamma"])

    assert [result.document.text for result in results] == ["gamma", "alpha", "beta"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.original_index for result in results] == [2, 0, 1]


@pytest.mark.asyncio
async def test_async_reranks_document_objects() -> None:
    provider = AsyncFakeProvider(['{"ranking": [2, 1]}'])
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )
    documents = [
        Document(id="a", text="first", metadata={"source": "one"}),
        Document(id="b", text="second", metadata={"source": "two"}),
    ]

    results = await reranker.rerank("query", documents, top_k=1)

    assert len(results) == 1
    assert results[0].document.id == "b"
    assert results[0].document.metadata == {"source": "two"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"ranking": [1, 1]}',
        '{"ranking": [1]}',
        '{"ranking": [1, 3]}',
        '{"ranking": ["1", "2"]}',
    ],
)
async def test_async_invalid_llm_ranking_fast_fails(response: str) -> None:
    provider = AsyncFakeProvider([response])
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )

    with pytest.raises(RerankParseError):
        await reranker.rerank("query", ["alpha", "beta"])


@pytest.mark.asyncio
async def test_async_provider_errors_are_wrapped() -> None:
    provider = AsyncFakeProvider([], fail=RuntimeError("timeout"))
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
    )

    with pytest.raises(RerankProviderError) as error:
        await reranker.rerank("query", ["alpha"])

    assert "timeout" in str(error.value)


@pytest.mark.asyncio
async def test_async_sliding_window_uses_multiple_provider_calls() -> None:
    provider = AsyncFakeProvider(
        [
            '{"ranking": [2, 1]}',
            '{"ranking": [2, 1]}',
        ]
    )
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=AsyncListwiseStrategy(
            algorithm="sliding_window", window_size=2, stride=1
        ),
    )

    results = await reranker.rerank("query", ["a", "b", "c"])

    assert [result.document.text for result in results] == ["c", "b", "a"]
    assert provider.calls == [["a", "b"], ["b", "c"]]


@pytest.mark.asyncio
async def test_async_rankgpt_sliding_window_bubbles_top_document_up() -> None:
    provider = AsyncFakeProvider(
        [
            '{"ranking": [3, 1, 2]}',
            '{"ranking": [3, 1, 2]}',
        ]
    )
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        provider=provider,
        strategy=AsyncListwiseStrategy(
            algorithm="rankgpt_sliding_window", window_size=3, stride=2
        ),
    )

    results = await reranker.rerank("query", ["a", "b", "c", "d", "e"])

    assert [result.document.text for result in results] == ["e", "a", "b", "c", "d"]
    assert provider.calls == [["c", "d", "e"], ["a", "b", "e"]]
