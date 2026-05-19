#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

# 저장소 루트에서 바로 실행할 수 있도록 src 경로를 추가합니다.
# 패키지를 설치한 사용자는 아래 두 줄이 필요하지 않습니다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksmith import AzureOpenAIReranker, Document, PairwiseStrategy  # noqa: E402


class KeywordPairwiseProvider:
    """예제용 deterministic provider. 실제 서비스에서는 Azure provider를 사용합니다."""

    def __init__(self, query_terms: set[str]) -> None:
        self.query_terms = query_terms
        self.compare_calls = 0

    def compare(self, query: str, document_a: Document, document_b: Document) -> str:
        del query
        self.compare_calls += 1
        score_a = self._score(document_a)
        score_b = self._score(document_b)
        winner = "A" if score_a >= score_b else "B"
        return f'{{"winner": "{winner}"}}'

    def _score(self, document: Document) -> tuple[int, int]:
        text = document.text.lower()
        exact_matches = sum(2 for term in self.query_terms if term in text)
        disease_evidence = 0
        if "괴혈병" in text:
            disease_evidence += 5
        if "빈혈" in text:
            disease_evidence += 4
        if "아니다" in text:
            disease_evidence -= 4
        length_penalty = -len(text)
        return exact_matches + disease_evidence, length_penalty


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
    provider = KeywordPairwiseProvider({"비타민", "결핍", "질병", "괴혈병", "빈혈"})
    reranker = AzureOpenAIReranker(
        api_key="example-key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="example-deployment",
        provider=provider,
        strategy=PairwiseStrategy(passes=3),
    )

    results = reranker.rerank(query, documents)

    print("Pairwise PRP-Sliding-K example")
    print(f"query={query}")
    for result in results:
        print(
            f"rank={result.rank:02d} id={result.document.id} "
            f"original_index={result.original_index}"
        )
    print(f"compare_calls={provider.compare_calls}")


if __name__ == "__main__":
    main()
