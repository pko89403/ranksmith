#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

# 저장소 루트에서 바로 실행할 수 있도록 src 경로를 추가합니다.
# 패키지를 설치한 사용자는 아래 두 줄이 필요하지 않습니다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksmith import AzureOpenAIReranker, Document, ListwiseStrategy  # noqa: E402


def main() -> None:
    """
    동기(Synchronous) 방식으로 Azure OpenAI 모델을 활용하여 문서를 랭킹하는 가이드.
    실행 전, 아래 환경 변수가 쉘에 설정되어 있거나 .env 파일에 구성되어 있어야 합니다.
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_LLM_DEPLOYMENT
    """
    load_env_file(ROOT / ".env")

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")

    if not all([api_key, endpoint, deployment]):
        print(
            "오류: 환경 변수 AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_LLM_DEPLOYMENT가 모두 설정되어 있어야 합니다.\n"
            "팁: .env 파일이 있다면 python-dotenv 등을 사용해 로드하거나, "
            "터미널에서 export 명령어로 값을 설정하세요."
        )
        sys.exit(1)

    # 1. Reranker 초기화
    reranker = AzureOpenAIReranker(
        api_key=api_key,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        strategy=ListwiseStrategy(
            algorithm="rankgpt_sliding_window",
            window_size=10,
            stride=5,
        ),
    )

    # 2. 질의(Query)와 검색된 초기 문서들(Documents) 준비
    query = "비타민 결핍과 관련된 질병"
    documents = [
        Document(
            id="doc_1",
            text="사과는 맛있고 비타민이 풍부하지만 질병의 직접적 원인은 아니다.",
        ),
        Document(
            id="doc_2",
            text=(
                "비타민 B12 결핍은 피로, 기억력 저하 및 "
                "심각한 악성 빈혈을 유발할 수 있다."
            ),
        ),
        Document(
            id="doc_3",
            text="수면 부족은 전반적인 면역 체계를 약화시켜 질병에 취약하게 만든다.",
        ),
        Document(
            id="doc_4",
            text=(
                "비타민 C가 심하게 결핍되면 괴혈병이 발생하여 잇몸 출혈 등이 나타난다."
            ),
        ),
    ]

    print(f"질의(Query): {query}")
    print("-" * 50)
    print("랭킹 평가 중... (Azure OpenAI API 호출)")

    # 3. 랭킹 실행
    results = reranker.rerank(query, documents)

    # 4. 결과 출력
    print("\n최종 랭킹 결과:")
    for result in results:
        print(
            f"순위 {result.rank:02d} | ID: {result.document.id} | "
            f"원본 인덱스: {result.original_index}"
        )
        print(f"-> 내용: {result.document.text}\n")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if separator == "":
            raise SystemExit(f"Invalid .env line without '=': {line}")
        os.environ.setdefault(key.strip(), clean_env_value(value))


def clean_env_value(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith(("'", '"')):
        quote = stripped[0]
        end = stripped.find(quote, 1)
        if end == -1:
            raise SystemExit("Invalid .env quoted value.")
        return stripped[1:end]
    return stripped.split("#", maxsplit=1)[0].strip()


if __name__ == "__main__":
    main()
