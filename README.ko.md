# ranksmith

<p align="center">
  <img src="https://raw.githubusercontent.com/pko89403/ranksmith/main/assets/ranksmith-icon.png" alt="ranksmith icon" width="160">
</p>

후보 문서를 더 나은 순서로 벼리는 LLM reranking 패키지입니다.

[English README](README.md)

`ranksmith`는 LLM 기반 reranking을 위한 작은 Python 패키지입니다. v1은
Azure OpenAI 기반 zero-shot listwise reranking에 집중합니다.

## 설치

```bash
pip install ranksmith
```

## 빠른 시작

```python
from ranksmith import AzureOpenAIReranker, Document

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
)

results = reranker.rerank(
    query="What is listwise reranking?",
    documents=[
        Document(id="a", text="Listwise reranking compares candidates together."),
        Document(id="b", text="Vector search retrieves candidate documents."),
    ],
    top_k=2,
)

for result in results:
    print(result.rank, result.original_index, result.document.id)
```

`rank`는 사람이 읽기 쉬운 1-based 값입니다. `original_index`는 입력 리스트로
다시 매핑하기 쉽도록 0-based 값입니다.

## 전략

기본 전략은 listwise reranking이며, 후보 문서가 많으면 sliding window로
처리합니다.

```python
from ranksmith import ListwiseStrategy

strategy = ListwiseStrategy(
    algorithm="sliding_window",
    window_size=20,
    stride=10,
    max_document_chars=4000,
)
```

v1은 `direct`, `sliding_window` 알고리즘을 지원합니다. Pointwise, pairwise,
tournament, bayesian, confidence 계열 알고리즘은 이후 버전에서 확장합니다.

## 비동기 지원 (Async Support)

대규모 트래픽이나 비동기 웹 프레임워크(FastAPI 등)를 위해 비동기 처리를 완벽하게 지원합니다.

```python
from ranksmith import AsyncAzureOpenAIReranker

reranker = AsyncAzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
)

results = await reranker.rerank("query", documents)
```

## 실전 가이드 (Examples)

실제 프로덕션 환경에 **RankGPT** 알고리즘을 바로 연동할 수 있는 완성된 형태의 예제 코드를 제공합니다. 환경 변수(`.env`) 세팅 방법과 함께 `examples/` 폴더에서 확인하세요.

- [`examples/rankgpt_sync.py`](examples/rankgpt_sync.py): 기본적인 동기 방식의 RankGPT 연동 가이드
- [`examples/rankgpt_async.py`](examples/rankgpt_async.py): 다중 문서 병렬 처리 및 고성능 비동기 방식의 RankGPT 연동 가이드

## 결과 모델

```python
result.document        # Document
result.rank            # 1-based rank
result.original_index  # 0-based input index
result.metadata        # 전략별 metadata
```

## 에러 처리

`ranksmith`는 fast fail 정책을 따릅니다. 긴 문서를 조용히 자르거나,
잘못된 순위를 자동 보정하거나, 검증되지 않은 LLM 출력을 반환하지 않습니다.

```python
from ranksmith import DocumentTooLongError, RerankParseError, RerankProviderError

try:
    results = reranker.rerank("query", documents)
except DocumentTooLongError:
    ...
except RerankParseError:
    ...
except RerankProviderError:
    ...
```
