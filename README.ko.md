# ranksmith

<p align="center">
  <img src="https://raw.githubusercontent.com/pko89403/ranksmith/main/assets/ranksmith-icon.png" alt="ranksmith icon" width="160">
</p>

후보 문서를 더 나은 순서로 벼리는 LLM reranking 패키지입니다.

[English README](https://github.com/pko89403/ranksmith/blob/main/README.md)

`ranksmith`는 LLM 기반 reranking을 위한 작은 Python 패키지입니다. v1은
Azure OpenAI 기반 zero-shot reranking에 집중합니다.

주요 특징:

- listwise RankGPT, pairwise PRP, tournament 방식 TourRank-r built-in Strategy
- 커스텀 reranking 메소드를 위한 public Strategy contract
- 엄격한 JSON parsing과 fast-fail 오류 정책
- sync/async Azure OpenAI reranker
- 근거 artifact가 커밋된 재현 가능한 benchmark 요약

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

`ranksmith`는 평가 방식(Strategy)과 실행 절차(Algorithm)를 분리합니다.

### 추천 사용 시나리오

| Method | Strategy | 추천 상황 | Trade-off |
| --- | --- | --- | --- |
| `rankgpt_sliding_window` | `ListwiseStrategy` | production 또는 evaluation에서 기본 LLM reranker가 필요할 때 | 호출 수가 적지만, 한 번에 전체 순위를 출력해야 하므로 output format에 민감할 수 있음 |
| `prp_sliding_k` | `PairwiseStrategy` | pairwise preference 비교가 필요하거나 PRP 방식 재현이 필요할 때 | LLM 호출 수가 많고, 기본 `passes=10`은 비용이 큼 |
| `tourrank_r`, `rounds=2` | `TourRankStrategy` | 중간 수준 호출 예산에서 listwise보다 강한 품질을 원할 때 | RankGPT보다 호출 수가 많지만 TourRank-10보다 훨씬 가벼움 |
| `tourrank_r`, `rounds=10` | `TourRankStrategy` | 품질 중심 offline reranking, 논문식 평가, 최종 reranking처럼 latency를 감수할 수 있을 때 | 일반 사용 기준 built-in 중 호출 비용이 가장 큼 |
| Custom strategy | `RerankStrategy` / `AsyncRerankStrategy` | deterministic business logic, proprietary ranking, 새 research method가 필요할 때 | ranking contract와 validation을 직접 책임져야 함 |

### 전략 적용 방법

사용자 정의 전략을 `AzureOpenAIReranker`에 주입(Inject)하여 사용할 수 있습니다.

```python
from ranksmith import AzureOpenAIReranker, ListwiseStrategy

strategy = ListwiseStrategy(
    algorithm="rankgpt_sliding_window",
    window_size=20,
    stride=10,
    max_document_chars=4000,
)

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=strategy,
)

results = reranker.rerank("query", documents)
```

TourRank-r도 같은 주입 지점을 사용합니다.

```python
from ranksmith import AzureOpenAIReranker, TourRankStrategy

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=TourRankStrategy(rounds=2, group_parallelism=1),
)
```

품질 중심 실행에서는 TourRank-10을 명시합니다.

```python
reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=TourRankStrategy(rounds=10),
)
```

> **참고**: `strategy`를 명시하지 않으면 기본적으로 `ListwiseStrategy(algorithm="rankgpt_sliding_window")`가 자동으로 적용됩니다. Pairwise PRP와 TourRank-r은 listwise보다 LLM 호출 수가 많으므로 live benchmark 전 호출 수를 확인해야 합니다.

## 커스텀 Strategy

커스텀 reranking 메소드는 `ListwiseStrategy.algorithm`에 새 문자열 값을 추가하는
방식보다, 새 Strategy 클래스로 구현하는 방식을 권장합니다. Strategy는 정규화된
`Document` 목록, provider, 선택적 `top_k`를 받아 `RerankResult` 목록을 반환합니다.

```python
from collections.abc import Sequence

from ranksmith import (
    AzureOpenAIReranker,
    Document,
    RerankResult,
)


class LengthStrategy:
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
        results = [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={"strategy": "length"},
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]
        return results if top_k is None else results[:top_k]


reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=LengthStrategy(),
)
```

provider-backed Strategy와 async Strategy도 같은 public contract를 따릅니다.
자세한 확장 가이드는
[커스텀 Strategy 확장 가이드](https://github.com/pko89403/ranksmith/blob/main/docs/wiki/08_custom_strategy_extension.md)와
[custom strategy 예제](https://github.com/pko89403/ranksmith/blob/main/examples/custom_strategy.py)를
참고하세요.

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

실행 가능한 예제는 `examples/` 폴더에 있습니다.

- [rankgpt_sync.py](https://github.com/pko89403/ranksmith/blob/main/examples/rankgpt_sync.py): 동기 RankGPT 연동
- [rankgpt_async.py](https://github.com/pko89403/ranksmith/blob/main/examples/rankgpt_async.py): 비동기 RankGPT 연동
- [pairwise_prp.py](https://github.com/pko89403/ranksmith/blob/main/examples/pairwise_prp.py): pairwise PRP Strategy
- [tourrank.py](https://github.com/pko89403/ranksmith/blob/main/examples/tourrank.py): fake provider 기반 TourRank-r
- [custom_strategy.py](https://github.com/pko89403/ranksmith/blob/main/examples/custom_strategy.py): custom Strategy 계약

## 벤치마크

아래 참고 측정은 reranking만 측정합니다. MTEB `AskUbuntuDupQuestions` test의
고정 후보를 사용했으며, query `361`개, query당 후보 `20`개, seed `13` shuffle,
Azure OpenAI deployment `gpt-5.4-nano` 조건입니다.

전체 명령, 호출 수 산식, 측정 범위, artifact 링크는
[MTEB reranking benchmark 문서](https://github.com/pko89403/ranksmith/blob/main/docs/benchmarks/mteb_reranking.ko.md)에
분리했습니다.

| Method | NDCG@10 | MRR@10 | MAP | Recall@10 | p50 latency | Invalid rate | LLM calls/query | Total calls | Queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | 0.3926 | 0.4594 | 0.3711 | 0.4993 | 0.0 ms | 0.000 | 0.0 | 0 | 361 |
| `rankgpt_sliding_window@20` | 0.6908 | 0.7470 | 0.6355 | 0.7671 | 1820.5 ms | 0.008 | 1.0 | 374 | 361 |
| `tourrank_r@20:r2` | 0.7023 | 0.7642 | 0.6421 | 0.7785 | 8297.1 ms | 0.000 | 8.0 | 2,888 | 361 |
| `tourrank_r@20:r10` | 0.7135 | 0.7734 | 0.6597 | 0.7836 | 39026.4 ms | 0.006 | 39.9 | 14,409 | 361 |

이 run에서는 `tourrank_r@20:r10`이 가장 높은 점수를 냈습니다.
`tourrank_r@20:r2`는 더 적은 호출과 낮은 latency로 근접한 결과를 냈습니다.
기본 `passes=10`의 `prp_sliding_k@20`은 이 full-query benchmark에서 실행하지
않았습니다. query당 `380`회, 전체 `137,180`회 호출이 필요하므로 이 설정의
품질/latency metric은 여기서 보고하지 않습니다.

보조 측정인 `prp_sliding_k@20:p1`은 기본 PRP 결과가 아니라 reduced-budget 호출
참고값으로 상세 문서에만 기록합니다.

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
from ranksmith import (
    DocumentTooLongError,
    RerankParseError,
    RerankProviderError,
    RerankStrategyError,
)

try:
    results = reranker.rerank("query", documents)
except DocumentTooLongError:
    ...
except RerankParseError:
    ...
except RerankProviderError:
    ...
except RerankStrategyError:
    ...
```
