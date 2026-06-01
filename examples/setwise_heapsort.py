#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

# 저장소 루트에서 바로 실행할 수 있도록 src 경로를 추가합니다.
# 패키지를 설치한 사용자는 아래 두 줄이 필요하지 않습니다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksmith import AzureOpenAIReranker, Document, SetwiseStrategy  # noqa: E402


class KeywordSelectionClient:
    """예제용 deterministic model client. 실제 서비스에서는 ModelClient를 사용합니다."""

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
        exact_matches = sum(2 for term in self.query_terms if term in text)
        disease_evidence = 0
        if "괴혈병" in text:
            disease_evidence += 5
        if "빈혈" in text:
            disease_evidence += 4
        if "아니다" in text:
            disease_evidence -= 4
        return exact_matches + disease_evidence, -len(text)


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
        Document(id="hydration", text="수분 섭취는 탈수 예방에 중요하다."),
        Document(id="iron", text="철분 부족은 빈혈과 피로 증상으로 이어질 수 있다."),
    ]
    model_client = KeywordSelectionClient({"비타민", "결핍", "질병", "괴혈병", "빈혈"})
    reranker = AzureOpenAIReranker(
        api_key="example-key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="example-deployment",
        model_client=model_client,
        strategy=SetwiseStrategy(set_size=4),
    )

    print("Setwise Heapsort example")
    print(f"query={query}")
    for result in reranker.rerank(query, documents, top_k=3):
        print(
            f"rank={result.rank:02d} id={result.document.id} "
            f"set_size={result.metadata['set_size']} "
            f"original_index={result.original_index}"
        )
    print(f"select_calls={len(model_client.calls)}")


if __name__ == "__main__":
    main()
