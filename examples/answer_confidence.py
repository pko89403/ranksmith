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
    AnswerConfidenceRerankStrategy,
    AzureOpenAIReranker,
    Document,
)
from ranksmith.confidence.types import AnswerConfidenceInput  # noqa: E402

NO_ANSWER = "__NO_ANSWER__"


class KeywordAnswerClient:
    """예제용 deterministic model client. 실제 서비스에서는 ModelClient를 사용합니다.

    문서가 질문에 답할 정보를 담고 있으면 답을, 아니면 NO_ANSWER를 반환합니다.
    """

    def __init__(self) -> None:
        self.answer_calls = 0

    def answer(self, query: str, context: str) -> str:
        del query
        self.answer_calls += 1
        if "괴혈병" in context:
            return json.dumps({"answer": "괴혈병"})
        if "빈혈" in context:
            return json.dumps({"answer": "악성 빈혈"})
        return json.dumps({"answer": NO_ANSWER})


class _Score:
    def __init__(self, value: float) -> None:
        self.score = value


class KeywordAnswerConfidenceEstimator:
    """예제용 deterministic estimator. 실제 서비스에서는
    ``StructuralConfidenceEstimator.from_artifact(...)`` 로 학습된 scorer를 씁니다.

    답변이 있고 문맥에 질병 근거가 뚜렷할수록 confidence가 높습니다.
    """

    task_type = "answer_confidence"

    def score(self, item: AnswerConfidenceInput) -> _Score:
        if item.answer == NO_ANSWER:
            return _Score(0.1)
        if "괴혈병" in item.context:
            return _Score(0.9)
        if "빈혈" in item.context:
            return _Score(0.8)
        return _Score(0.5)


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
    model_client = KeywordAnswerClient()
    estimator = KeywordAnswerConfidenceEstimator()
    reranker = AzureOpenAIReranker(
        api_key="example-key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="example-deployment",
        model_client=model_client,
        strategy=AnswerConfidenceRerankStrategy(estimator=estimator),
    )

    results = reranker.rerank(query, documents)

    print("Answer Confidence rerank example")
    print(f"query={query}")
    for result in results:
        print(
            f"rank={result.rank:02d} id={result.document.id} "
            f"answer_confidence={result.metadata['answer_confidence']:.2f}"
        )
    print(f"answer_calls={model_client.answer_calls}")


if __name__ == "__main__":
    main()
