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

## 지원하는 전략 및 알고리즘 (Strategy & Algorithm)

`ranksmith`는 평가 방식(Strategy)과 그 방식을 풀어내는 세부 구현체(Algorithm)를 분리하여 제공합니다. v1에서는 **Listwise Reranking (RankGPT 기반)** 전략을 공식 지원합니다.

### 1. ListwiseStrategy (RankGPT)
프롬프트에 여러 문서를 한 번에 넣고 LLM에게 전체 순위를 매기도록 요청하는 전략입니다.

- **`direct` 알고리즘**
  - 모든 문서를 한 번의 LLM API 호출로 처리합니다.
  - 후보 문서의 개수가 적고 LLM의 컨텍스트 윈도우 내에 모두 들어갈 때 적합합니다.
- **`sliding_window` 알고리즘 (기본값)**
  - 후보 문서가 너무 많아 한 번에 처리할 수 없을 때, 문서를 윈도우 크기(`window_size`)만큼 나누어 겹치면서(`stride`) 반복적으로 순위를 매깁니다.
  - 많은 수의 문서를 처리하거나, 프롬프트 길이가 길어져 LLM의 순위 판단 능력이 떨어지는 현상(Lost in the middle)을 방지할 때 필수적입니다.
- **`rankgpt_sliding_window` 알고리즘**
  - RankGPT 방식의 뒤에서 앞으로(back-to-first) 이동하는 sliding window와 bubble-up 동작을 구현합니다.
  - RankGPT의 윈도우 순회 방식을 쓰면서도 ranksmith의 엄격한 JSON 출력 검증을 유지하고 싶을 때 적합합니다.

### 전략 적용 방법

사용자 정의 전략을 `AzureOpenAIReranker`에 주입(Inject)하여 사용할 수 있습니다.

```python
from ranksmith import AzureOpenAIReranker, ListwiseStrategy

# 1. 원하는 전략과 알고리즘 구성
strategy = ListwiseStrategy(
    algorithm="rankgpt_sliding_window",
    window_size=20,             # 한 번에 평가할 문서 수
    stride=10,                  # 다음 윈도우로 넘어갈 때 겹칠 문서 수
    max_document_chars=4000,    # 문서당 최대 허용 글자 수
)

# 2. Reranker에 주입
reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=strategy, # <-- 전략 주입
)

results = reranker.rerank("query", documents)
```

> **참고**: `strategy`를 명시하지 않으면 기본적으로 `ListwiseStrategy(algorithm="sliding_window")`가 자동으로 적용됩니다. v1은 `direct`, `sliding_window`, `rankgpt_sliding_window`를 지원합니다. 추후 버전에서 Pointwise, Pairwise, Tournament 등의 전략도 확장될 예정입니다.

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
