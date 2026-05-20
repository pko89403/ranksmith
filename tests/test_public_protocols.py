from __future__ import annotations

from collections.abc import Sequence

import pytest

from ranksmith import (
    AsyncAzureOpenAIReranker,
    AzureOpenAIReranker,
    Document,
    ListwiseStrategy,
    ModelClient,
    RerankInputError,
    RerankParseError,
    RerankProviderError,
    RerankResult,
    RerankStrategyError,
)


def test_protocols_are_publicly_importable_from_root() -> None:
    from ranksmith import (  # noqa: PLC0415
        AsyncModelClient,
        AsyncModelProvider,
        AsyncRerankStrategy,
        AsyncTourRankStrategy,
        ModelClient,
        ModelProvider,
        RerankStrategy,
        RerankStrategyError,
        TourRankStageConfig,
        TourRankStrategy,
        parse_ranking_response,
        parse_selection_response,
    )

    assert RerankStrategy is not None
    assert AsyncRerankStrategy is not None
    assert ModelClient is not None
    assert AsyncModelClient is not None
    assert ModelProvider is not None
    assert AsyncModelProvider is not None
    assert TourRankStrategy is not None
    assert AsyncTourRankStrategy is not None
    assert TourRankStageConfig is not None
    assert RerankStrategyError is not None
    assert parse_ranking_response is not None
    assert parse_selection_response is not None


def test_protocols_are_publicly_importable_from_protocols_module() -> None:
    from ranksmith.protocols import (  # noqa: PLC0415
        AsyncModelProvider,
        AsyncRerankStrategy,
        ModelProvider,
        RerankStrategy,
    )

    assert RerankStrategy is not None
    assert AsyncRerankStrategy is not None
    assert ModelProvider is not None
    assert AsyncModelProvider is not None


def test_parse_helper_is_importable_from_parsing_module() -> None:
    from ranksmith.parsing import (  # noqa: PLC0415
        parse_ranking_response,
        parse_selection_response,
    )

    assert parse_ranking_response is not None
    assert parse_selection_response is not None


def test_parse_ranking_response_fast_fails_invalid_ranking() -> None:
    from ranksmith import parse_ranking_response  # noqa: PLC0415

    with pytest.raises(RerankParseError):
        parse_ranking_response('{"ranking": [1, 1]}', expected_count=2)


def test_parse_ranking_response_rejects_negative_expected_count() -> None:
    from ranksmith import parse_ranking_response  # noqa: PLC0415

    with pytest.raises(RerankInputError):
        parse_ranking_response('{"ranking": []}', expected_count=-1)


def test_parse_selection_response_accepts_selected_indexes() -> None:
    from ranksmith import parse_selection_response  # noqa: PLC0415

    assert parse_selection_response(
        '{"selected": [3, 1]}',
        expected_count=3,
        selected_count=2,
    ) == [3, 1]


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "{}",
        '{"selected": [1, 1]}',
        '{"selected": [1]}',
        '{"selected": [1, 4]}',
        '{"selected": ["1", "2"]}',
    ],
)
def test_parse_selection_response_fast_fails_invalid_selection(
    response: str,
) -> None:
    from ranksmith import parse_selection_response  # noqa: PLC0415

    with pytest.raises(RerankParseError):
        parse_selection_response(response, expected_count=3, selected_count=2)


def test_parse_selection_response_rejects_negative_counts() -> None:
    from ranksmith import parse_selection_response  # noqa: PLC0415

    with pytest.raises(RerankInputError):
        parse_selection_response(
            '{"selected": []}',
            expected_count=-1,
            selected_count=0,
        )
    with pytest.raises(RerankInputError):
        parse_selection_response(
            '{"selected": []}',
            expected_count=0,
            selected_count=-1,
        )


class LengthDescendingStrategy:
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
                metadata={"strategy": "custom-length"},
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]
        if top_k is None:
            return results
        return results[:top_k]


class UnusedListwiseProvider:
    def rank(self, query: str, documents: list[Document]) -> str:
        del query, documents
        raise AssertionError("custom strategy should not call provider")


def test_custom_sync_strategy_can_be_injected() -> None:
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=UnusedListwiseProvider(),
        strategy=LengthDescendingStrategy(),
    )

    results = reranker.rerank("query", ["a", "bbbb", "cc"], top_k=2)

    assert [result.document.text for result in results] == ["bbbb", "cc"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.original_index for result in results] == [1, 2]
    assert [dict(result.metadata) for result in results] == [
        {"strategy": "custom-length"},
        {"strategy": "custom-length"},
    ]


class RankResponseProvider:
    def rank(self, query: str, documents: list[Document]) -> str:
        del query, documents
        return '{"ranking": [2, 1]}'


class FailingRankProvider:
    def rank(self, query: str, documents: list[Document]) -> str:
        del query, documents
        raise TimeoutError("provider timeout")


class ProviderBackedStrategy:
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: ModelClient,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        from ranksmith import parse_ranking_response  # noqa: PLC0415

        ranking = parse_ranking_response(
            model_client.rank(query, list(documents)),
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
        if top_k is None:
            return results
        return results[:top_k]


class ProviderBackedStrategyWithProviderErrors(ProviderBackedStrategy):
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: ModelClient,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        try:
            return super().rerank(
                query=query,
                documents=documents,
                model_client=model_client,
                top_k=top_k,
            )
        except TimeoutError as exc:
            raise RerankProviderError(str(exc)) from exc


def test_custom_strategy_can_type_provider_protocol() -> None:
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=RankResponseProvider(),
        strategy=ProviderBackedStrategy(),
    )

    results = reranker.rerank("query", ["first", "second"])

    assert [result.document.text for result in results] == ["second", "first"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.original_index for result in results] == [1, 0]


def test_provider_backed_custom_strategy_can_preserve_provider_error() -> None:
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=FailingRankProvider(),
        strategy=ProviderBackedStrategyWithProviderErrors(),
    )

    with pytest.raises(RerankProviderError) as error:
        reranker.rerank("query", ["first", "second"])

    assert "provider timeout" in str(error.value)


class BrokenCustomStrategy:
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: object,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del query, documents, model_client, top_k
        raise RuntimeError("custom strategy bug")


def test_custom_strategy_unexpected_error_is_not_provider_error() -> None:
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=UnusedListwiseProvider(),
        strategy=BrokenCustomStrategy(),
    )

    with pytest.raises(RerankStrategyError) as error:
        reranker.rerank("query", ["first"])

    assert "custom strategy bug" in str(error.value)


class BrokenListwiseSubclass(ListwiseStrategy):
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: object,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del query, documents, model_client, top_k
        raise RuntimeError("subclass strategy bug")


def test_custom_subclass_unexpected_error_is_not_provider_error() -> None:
    reranker = AzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=UnusedListwiseProvider(),
        strategy=BrokenListwiseSubclass(),
    )

    with pytest.raises(RerankStrategyError) as error:
        reranker.rerank("query", ["first"])

    assert "subclass strategy bug" in str(error.value)


class AsyncLengthDescendingStrategy:
    async def rerank(
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
                metadata={"strategy": "custom-async-length"},
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]
        if top_k is None:
            return results
        return results[:top_k]


class UnusedAsyncListwiseProvider:
    async def rank(self, query: str, documents: list[Document]) -> str:
        del query, documents
        raise AssertionError("custom async strategy should not call provider")


class BrokenAsyncCustomStrategy:
    async def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: object,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del query, documents, model_client, top_k
        raise RuntimeError("custom async strategy bug")


@pytest.mark.asyncio
async def test_custom_async_strategy_can_be_injected() -> None:
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=UnusedAsyncListwiseProvider(),
        strategy=AsyncLengthDescendingStrategy(),
    )

    results = await reranker.rerank("query", ["a", "bbbb", "cc"], top_k=2)

    assert [result.document.text for result in results] == ["bbbb", "cc"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.original_index for result in results] == [1, 2]
    assert [dict(result.metadata) for result in results] == [
        {"strategy": "custom-async-length"},
        {"strategy": "custom-async-length"},
    ]


@pytest.mark.asyncio
async def test_custom_async_strategy_unexpected_error_is_not_provider_error() -> None:
    reranker = AsyncAzureOpenAIReranker(
        api_key="key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-4o-mini",
        model_client=UnusedAsyncListwiseProvider(),
        strategy=BrokenAsyncCustomStrategy(),
    )

    with pytest.raises(RerankStrategyError) as error:
        await reranker.rerank("query", ["first"])

    assert "custom async strategy bug" in str(error.value)
