#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

# 저장소 루트에서 바로 실행할 수 있도록 src 경로를 추가합니다.
# 패키지를 설치한 사용자는 아래 두 줄이 필요하지 않습니다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksmith import AcuRankStrategy, AzureOpenAIReranker, Document  # noqa: E402


class KeywordListwiseClient:
    """예제용 deterministic model client. 실제 서비스에서는 ModelClient를 사용합니다."""

    def __init__(self, query_terms: set[str]) -> None:
        self.query_terms = query_terms
        self.calls: list[list[str]] = []

    def rank(self, query: str, documents: list[Document]) -> str:
        del query
        self.calls.append([document.id or "" for document in documents])
        ranking = sorted(
            range(len(documents)),
            key=lambda index: self._score(documents[index]),
            reverse=True,
        )
        return json.dumps({"ranking": [index + 1 for index in ranking]})

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
            metadata={"score": 2.0},
        ),
        Document(
            id="vitamin_b12",
            text="비타민 B12 결핍은 피로, 신경 증상, 악성 빈혈을 유발할 수 있다.",
            metadata={"score": 8.0},
        ),
        Document(
            id="sleep",
            text="수면 부족은 면역 저하와 관련되지만 비타민 결핍 질병은 아니다.",
            metadata={"score": 3.0},
        ),
        Document(
            id="vitamin_c",
            text=(
                "비타민 C 결핍은 괴혈병을 일으키며 잇몸 출혈과 상처 회복 지연을 낳는다."
            ),
            metadata={"score": 7.0},
        ),
        Document(
            id="hydration",
            text="수분 섭취는 탈수 예방에 중요하지만 비타민 결핍 질병과는 다르다.",
            metadata={"score": 1.0},
        ),
    ]
    model_client = KeywordListwiseClient({"비타민", "결핍", "질병", "괴혈병", "빈혈"})
    reranker = AzureOpenAIReranker(
        api_key="example-key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="example-deployment",
        model_client=model_client,
        strategy=AcuRankStrategy(
            target_rank=3,
            window_size=3,
            max_adaptive_reranker_calls=1,
            batch_parallelism=1,
        ),
    )

    print("AcuRank example")
    print(f"query={query}")
    for result in reranker.rerank(query, documents, top_k=3):
        print(
            f"rank={result.rank:02d} id={result.document.id} "
            f"mu={result.metadata['mu']:.3f} "
            f"sigma={result.metadata['sigma']:.3f} "
            f"top_k_probability={result.metadata['top_k_probability']:.3f} "
            f"original_index={result.original_index}"
        )
    print(f"rank_calls={len(model_client.calls)}")


if __name__ == "__main__":
    main()
