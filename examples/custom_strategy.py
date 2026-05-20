#!/usr/bin/env python
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

# 저장소 루트에서 바로 실행할 수 있도록 src 경로를 추가합니다.
# 패키지를 설치한 사용자는 아래 두 줄이 필요하지 않습니다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksmith import (  # noqa: E402
    AzureOpenAIReranker,
    Document,
    LLMProvider,
    RerankProviderError,
    RerankResult,
    parse_ranking_response,
)


class LengthStrategy:
    """Provider를 쓰지 않는 deterministic custom strategy 예제."""

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
        return _build_results(
            documents,
            ordered_indexes,
            strategy_name="length",
            top_k=top_k,
        )


class KeywordListwiseProvider:
    """예제용 deterministic provider. 실제 서비스에서는 Azure provider를 사용합니다."""

    def __init__(self, query_terms: set[str]) -> None:
        self.query_terms = query_terms

    def rank(self, query: str, documents: list[Document]) -> str:
        del query
        scored_indexes = sorted(
            range(len(documents)),
            key=lambda index: self._score(documents[index]),
            reverse=True,
        )
        ranking = [index + 1 for index in scored_indexes]
        return '{"ranking": [' + ", ".join(str(item) for item in ranking) + "]}"

    def _score(self, document: Document) -> tuple[int, int]:
        text = document.text.lower()
        exact_matches = sum(2 for term in self.query_terms if term in text)
        disease_evidence = 0
        if "빈혈" in text:
            disease_evidence += 5
        if "괴혈병" in text:
            disease_evidence += 4
        if "아니다" in text:
            disease_evidence -= 5
        length_penalty = -len(text)
        return exact_matches + disease_evidence, length_penalty


class FailingListwiseProvider:
    def rank(self, query: str, documents: list[Document]) -> str:
        del query, documents
        raise TimeoutError("provider timeout")


class ProviderBackedStrategy:
    """Provider JSON을 쓰는 custom strategy 예제."""

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: LLMProvider,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        try:
            raw_response = provider.rank(query, list(documents))
        except TimeoutError as exc:
            raise RerankProviderError(str(exc)) from exc

        ranking = parse_ranking_response(
            raw_response,
            expected_count=len(documents),
        )
        ordered_indexes = [number - 1 for number in ranking]
        return _build_results(
            documents,
            ordered_indexes,
            strategy_name="provider-backed",
            top_k=top_k,
        )


def _build_results(
    documents: Sequence[Document],
    ordered_indexes: Sequence[int],
    *,
    strategy_name: str,
    top_k: int | None,
) -> list[RerankResult]:
    results = [
        RerankResult(
            document=documents[original_index],
            rank=rank,
            original_index=original_index,
            metadata={"strategy": strategy_name},
        )
        for rank, original_index in enumerate(ordered_indexes, start=1)
    ]
    return results if top_k is None else results[:top_k]


def main() -> None:
    query = "비타민 결핍으로 생기는 질병"
    documents = [
        Document(
            id="apple",
            text="사과는 비타민을 포함하지만 결핍성 질병 설명과는 직접 관련이 낮다.",
        ),
        Document(
            id="vitamin_b12",
            text="비타민 B12 결핍은 피로, 신경 증상, 악성 빈혈을 유발할 수 있다.",
        ),
        Document(
            id="sleep",
            text="수면 부족은 면역 저하와 관련되지만 비타민 결핍 질병은 아니다.",
        ),
        Document(
            id="vitamin_c",
            text=(
                "비타민 C 결핍은 괴혈병을 일으키며 잇몸 출혈과 상처 회복 지연을 낳는다."
            ),
        ),
    ]

    print("Custom Strategy example")
    print(f"query={query}")

    length_reranker = AzureOpenAIReranker(
        api_key="example-key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="example-deployment",
        provider=KeywordListwiseProvider({"unused"}),
        strategy=LengthStrategy(),
    )
    for result in length_reranker.rerank(query, documents, top_k=2):
        print(
            f"length_strategy rank={result.rank:02d} id={result.document.id} "
            f"original_index={result.original_index}"
        )

    provider_reranker = AzureOpenAIReranker(
        api_key="example-key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="example-deployment",
        provider=KeywordListwiseProvider({"비타민", "결핍", "질병", "빈혈", "괴혈병"}),
        strategy=ProviderBackedStrategy(),
    )
    for result in provider_reranker.rerank(query, documents, top_k=2):
        print(
            f"provider_strategy rank={result.rank:02d} id={result.document.id} "
            f"original_index={result.original_index}"
        )

    failing_reranker = AzureOpenAIReranker(
        api_key="example-key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="example-deployment",
        provider=FailingListwiseProvider(),
        strategy=ProviderBackedStrategy(),
    )
    try:
        failing_reranker.rerank(query, documents)
    except RerankProviderError as exc:
        print(f"provider_error={exc}")


if __name__ == "__main__":
    main()
