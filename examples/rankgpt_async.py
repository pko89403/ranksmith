#!/usr/bin/env python
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# 저장소 루트에서 바로 실행할 수 있도록 src 경로를 추가합니다.
# 패키지를 설치한 사용자는 아래 두 줄이 필요하지 않습니다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openai import AsyncAzureOpenAI  # noqa: E402

from ranksmith import (  # noqa: E402
    AsyncAzureOpenAIReranker,
    AsyncListwiseStrategy,
    Document,
)


async def async_main() -> None:
    """
    비동기(Asynchronous) 방식으로 Azure OpenAI 모델을 활용하여 문서를 랭킹하는 가이드.
    대규모 I/O(다중 문서 병렬 처리)나 FastAPI 등 비동기 웹 프레임워크 환경에 적합합니다.

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
            "AZURE_OPENAI_LLM_DEPLOYMENT가 모두 설정되어 있어야 합니다."
        )
        sys.exit(1)

    # 1. 비동기 클라이언트 직접 생성 (선택 사항: 타임아웃, Max Retries 등 설정 가능)
    client = AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-10-21",  # 필요한 API 버전에 맞게 수정
    )

    # 2. 비동기 Reranker 초기화
    reranker = AsyncAzureOpenAIReranker(
        client=client,
        azure_deployment=deployment,
        strategy=AsyncListwiseStrategy(
            algorithm="rankgpt_sliding_window",
            window_size=10,
            stride=5,
        ),
    )

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
    print("랭킹 평가 중... (Async Azure OpenAI API 호출)")

    # 3. 비동기 랭킹 실행 (await 사용)
    results = await reranker.rerank(query, documents)

    # 4. 결과 출력
    print("\n최종 랭킹 결과:")
    for result in results:
        print(
            f"순위 {result.rank:02d} | ID: {result.document.id} | "
            f"원본 인덱스: {result.original_index}"
        )
        print(f"-> 내용: {result.document.text}\n")


def main() -> None:
    # asyncio 이벤트 루프 실행
    asyncio.run(async_main())


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
