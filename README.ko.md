# ranksmith

<p align="center">
  <img src="https://raw.githubusercontent.com/pko89403/ranksmith/main/assets/ranksmith-icon.png" alt="ranksmith icon" width="160">
</p>

후보 문서를 더 나은 순서로 벼리는 LLM reranking 패키지입니다.

[English README](README.md)

`ranksmith`는 LLM 기반 reranking을 위한 작은 Python 패키지입니다. v1은
Azure OpenAI 기반 zero-shot reranking에 집중합니다.

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
from ranksmith import AzureOpenAIReranker, ListwiseStrategy, PairwiseStrategy

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

Pairwise PRP도 같은 방식으로 주입합니다.

```python
strategy = PairwiseStrategy(
    algorithm="prp_sliding_k",
    passes=10,
    max_document_chars=4000,
)

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=strategy,
)
```

TourRank-r도 같은 방식으로 주입합니다.

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

provider를 직접 쓰는 Strategy도 public provider protocol로 타입을 잡을 수 있습니다.

```python
from collections.abc import Sequence

from ranksmith import (
    Document,
    LLMProvider,
    RerankResult,
    parse_ranking_response,
)


class ProviderBackedStrategy:
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        provider: LLMProvider,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        ranking = parse_ranking_response(
            provider.rank(query, list(documents)),
            expected_count=len(documents),
        )
        ordered_indexes = [number - 1 for number in ranking]
        results = [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={"strategy": "provider-backed"},
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]
        return results if top_k is None else results[:top_k]
```

비동기 Strategy는 같은 계약을 `async def rerank(...)`로 구현하고
`AsyncRerankStrategy`로 타입을 지정하면 됩니다. 커스텀 Strategy에서 예상하지
못한 예외가 발생하면 `AzureOpenAIReranker`는 이를 `RerankStrategyError`로
감쌉니다. 오류 분류가 중요하면 `RerankError` 하위 예외를 직접 발생시키세요.

실행 가능한 오프라인 예제는 [`examples/custom_strategy.py`](examples/custom_strategy.py)를
참고하세요. deterministic Strategy, provider-backed Strategy, 엄격한 ranking
파싱, provider 오류 분류를 함께 보여줍니다.

PRP wall time을 줄이려면 async strategy를 사용하세요. PRP-Sliding-K 방식은
유지됩니다. 인접 pair는 여전히 아래에서 위로 순차 처리하고, 같은 pair의
A/B와 B/A 호출만 동시에 보냅니다.

```python
from ranksmith import AsyncAzureOpenAIReranker, AsyncPairwiseStrategy

reranker = AsyncAzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=AsyncPairwiseStrategy(
        passes=10,
        pair_order_parallelism=2,
    ),
)
```

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

## 벤치마크

`ranksmith`의 비교 스크립트는 reranking 단계의 live LLM 호출 수를 실행 전에
추정해 출력합니다. 호출 수는 query 수, 선택한 algorithm, `window_size`,
`stride`, `passes`, query별 candidate 수에 따라 달라집니다.

- `rankgpt_sliding_window`: RankGPT back-to-front window마다 LLM 1회 호출
- `prp_sliding_k`: query마다 `2 * passes * max(document_count - 1, 0)` pairwise LLM 호출
- `tourrank_r`: query마다 `tourrank_rounds * sum(stage.group_count)` selection
  LLM 호출. 후보가 정확히 100개이면 논문 top-100 stage를 쓰고, 그 외 후보
  수에는 명시적인 single-group halving stage를 생성해 실행합니다. 논문
  top-100 stage 기준 TourRank-2는 query당 26회, TourRank-10은 query당
  130회 호출합니다.

비교 스크립트는 first-stage candidate, embedding, community를 생성하지
않습니다. candidate TSV를 embedding retrieval이나 community-building
pipeline으로 만들었다면, 그 비용은 별도로 기록해야 합니다.

일반적인 전체 pipeline 비용은 다음처럼 나눠서 봅니다.

1. Candidate generation: corpus/query vector 생성을 위한 embedding 호출과,
   community 생성/요약에 쓰인 LLM 호출
2. Reranking: `ranksmith`가 선택된 reranking algorithm 실행을 위해 호출한 LLM
   호출

community retrieval까지 포함한 실험 요약에는 `embedding calls=<n>`,
`community LLM calls=<n>`, `reranking LLM calls=<n>`처럼 구분해 기록하는 것을
권장합니다.

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

## MTEB Reranking 참고 측정

이 벤치마크는 재랭킹만 측정합니다. first-stage retrieval은 포함하지 않고,
MTEB가 제공하는 고정 후보 집합을 그대로 사용합니다.

```bash
UV_NATIVE_TLS=true uv run python scripts/evaluate_mteb_reranking.py \
  --tasks AskUbuntuDupQuestions \
  --methods \
    original \
    rankgpt_sliding_window@20 \
    prp_sliding_k@20:p1 \
    tourrank_r@20:r2 \
    tourrank_r@20:r10 \
  --output-dir benchmark-results/mteb-reranking/askubuntu-full-tourrank-prp-20260520 \
  --max-document-chars 4000 \
  --shuffle-candidates --shuffle-seed 13 \
  --rankgpt-window-size 20 --rankgpt-step 10 \
  --concurrency 24 \
  --retry-invalid-outputs 1 \
  --input-token-price-per-1m 2.50 \
  --output-token-price-per-1m 10.00 \
  --allow-live
```

측정 범위:

- Dataset: `AskUbuntuDupQuestions`, `test` split
- Queries: `361`
- Candidates: MTEB가 제공하는 `top_ranked` 후보, query당 `20`개
- Candidate order: seed `13`으로 shuffled
- Model: Azure OpenAI deployment `gpt-5.4-nano`
- Validation: strict JSON, invalid output은 zero-scored
- Resume policy: failed row는 `--resume --retry-failed-results`로 재시도
- Artifact: `benchmark-results/mteb-reranking/askubuntu-full-tourrank-prp-20260520`

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

보조 측정인 `prp_sliding_k@20:p1`은 `tourrank_r@20:r10`과 비슷한 호출 예산을
보기 위해 같은 361개 query에서만 실행했습니다: NDCG@10 `0.5360`, MRR@10
`0.7261`, MAP `0.4983`, Recall@10 `0.5773`, p50 latency `19919.1 ms`,
invalid rate `0.000`, query당 `38.0`회, 전체 `13,718`회 호출.
