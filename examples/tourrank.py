#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

# 저장소 루트에서 바로 실행할 수 있도록 src 경로를 추가합니다.
# 패키지를 설치한 사용자는 아래 두 줄이 필요하지 않습니다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksmith import (  # noqa: E402
    AzureOpenAIReranker,
    Document,
    TourRankStageConfig,
    TourRankStrategy,
)


class KeywordSelectionProvider:
    def __init__(self, query_terms: set[str]) -> None:
        self.query_terms = query_terms
        self.calls: list[list[str]] = []

    def select(self, query: str, documents: list[Document], top_m: int) -> str:
        del query
        self.calls.append([document.id or "" for document in documents])
        selected = sorted(
            range(len(documents)),
            key=lambda index: self._score(documents[index]),
            reverse=True,
        )[:top_m]
        return json.dumps({"selected": [index + 1 for index in selected]})

    def _score(self, document: Document) -> tuple[int, int]:
        text = document.text.lower()
        matches = sum(2 for term in self.query_terms if term in text)
        if "괴혈병" in text:
            matches += 5
        if "빈혈" in text:
            matches += 4
        return matches, -len(text)


def main() -> None:
    query = "비타민 결핍으로 생기는 질병"
    documents = [
        Document(
            id="apple",
            text="사과는 비타민을 포함하지만 질병 설명과는 거리가 있다.",
        ),
        Document(
            id="vitamin_b12",
            text="비타민 B12 결핍은 악성 빈혈을 유발할 수 있다.",
        ),
        Document(id="sleep", text="수면 부족은 비타민 결핍 질병은 아니다."),
        Document(id="vitamin_c", text="비타민 C 결핍은 괴혈병을 일으킨다."),
        Document(id="hydration", text="수분 섭취는 탈수 예방에 중요하다."),
    ]
    provider = KeywordSelectionProvider({"비타민", "결핍", "질병", "빈혈", "괴혈병"})
    reranker = AzureOpenAIReranker(
        api_key="example-key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="example-deployment",
        provider=provider,
        strategy=TourRankStrategy(
            rounds=2,
            stage_configs=(
                TourRankStageConfig(group_count=1, group_size=5, selected_count=2),
                TourRankStageConfig(group_count=1, group_size=2, selected_count=1),
            ),
        ),
    )

    print("TourRank-r example")
    print(f"query={query}")
    for result in reranker.rerank(query, documents, top_k=3):
        print(
            f"tourrank rank={result.rank:02d} id={result.document.id} "
            f"score={result.metadata['score']} original_index={result.original_index}"
        )
    print(f"select_calls={len(provider.calls)}")


if __name__ == "__main__":
    main()
