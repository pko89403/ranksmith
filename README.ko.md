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

`ranksmith`는 평가 방식(Strategy)과 그 방식을 풀어내는 세부 구현체(Algorithm)를 분리하여 제공합니다. v1에서는 listwise reranking과 pairwise PRP reranking을 지원합니다.

### 1. ListwiseStrategy (RankGPT)
프롬프트에 여러 문서를 한 번에 넣고 LLM에게 전체 순위를 매기도록 요청하는 전략입니다.

- **`rankgpt_sliding_window` 알고리즘 (기본값)**
  - RankGPT 방식의 뒤에서 앞으로(back-to-first) 이동하는 sliding window와 bubble-up 동작을 구현합니다.
  - RankGPT의 윈도우 순회 방식을 쓰면서도 ranksmith의 엄격한 JSON 출력 검증을 유지하고 싶을 때 적합합니다.

### 2. PairwiseStrategy (PRP)
두 문서씩 비교하는 Pairwise Ranking Prompting 전략입니다.

- **`prp_sliding_k` 알고리즘**
  - 현재 순위의 아래쪽부터 인접 문서 쌍을 비교합니다.
  - 위치 편향을 줄이기 위해 같은 쌍을 A/B, B/A 순서로 두 번 호출합니다.
  - 두 유효 비교가 충돌하면 동률로 보고 현재 순서를 유지합니다.
  - 기본 `passes=10`이며, reference 논문의 PRP-Sliding-10 설정과 맞춥니다.
  - query당 예상 provider 호출 수는 `2 * passes * max(document_count - 1, 0)`입니다.
  - `AsyncPairwiseStrategy`는 `pair_order_parallelism=2`로 같은 pair의 A/B, B/A 호출만 병렬 실행할 수 있으며, PRP 순회 방식과 호출 수는 바꾸지 않습니다.

### 3. TourRankStrategy (TourRank-r)
후보 문서를 토너먼트 참가자처럼 보고, 각 stage의 group 안에서 provider가
top-`m` 문서를 선택합니다. 선택된 문서는 다음 stage로 진출하고 점수를 얻으며,
최종 순위는 누적 점수 내림차순으로 정렬합니다.

- **`tourrank_r` 알고리즘**
  - 기본값은 비용과 품질 균형을 위한 `rounds=2`입니다.
  - 품질 중심 평가, 논문식 재현, 최종 offline reranking처럼 추가 LLM 호출을
    감수할 수 있는 경우에는 `rounds=10`을 우선 권장합니다.
  - 기본 stage는 정확히 100개 후보 문서를 가정합니다:
    `100 -> 50 -> 20 -> 10 -> 5 -> 2`.
  - 다른 후보 수에서는 `stage_configs`를 명시해야 하며, ranksmith는 자동 보정이나 조용한 truncation을 하지 않습니다.
  - `TourRankStrategy`는 sync 호출 안정성을 위해 기본 `group_parallelism=1`입니다. 같은 stage의 group을 병렬 실행하려면 값을 높입니다. 병렬 실행 중 한 group이 실패해도 이미 시작된 group 호출은 끝까지 진행될 수 있습니다.
  - `AsyncTourRankStrategy`는 기본적으로 group을 병렬 실행합니다. `group_parallelism`을 지정하면 동시 provider 호출 수를 제한합니다.

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

아래 결과는 참고 기준점이며 보편적 leaderboard가 아닙니다.
점수는 dataset, model, candidate 수, latency 예산, invalid output 비율에 따라 달라지며,
이 벤치마크는 native MTEB 후보 집합 위의 재랭킹만 측정합니다 (first-stage retrieval 미포함).

```bash
uv run python scripts/evaluate_mteb_reranking.py \
  --tasks AskUbuntuDupQuestions SciDocsRR StackOverflowDupQuestions \
  --methods \
    original \
    rankgpt_sliding_window@20 \
    prp_sliding_k@20 \
    tourrank_r@20:r2 \
    tourrank_r@20:r10 \
  --output-dir benchmark-results/mteb-reranking/example \
  --max-queries 50 \
  --max-document-chars 4000 \
  --shuffle-candidates --shuffle-seed 13 \
  --rankgpt-window-size 20 --rankgpt-step 10 \
  --prp-passes 10 \
  --concurrency 4 \
  --input-token-price-per-1m 2.50 \
  --output-token-price-per-1m 10.00 \
  --allow-live
```

TourRank method는 `tourrank_r@N:rR` 형식을 사용합니다. `N`은 rerank할
native MTEB 후보 수이고, `R`은 tournament round 수입니다. `:rR`을 생략하면
runner가 `:r2`로 정규화합니다. 권장하는 간결한 비교는
`tourrank_r@20:r2`와 `tourrank_r@20:r10`이며, candidate scope를 고정한 채
TourRank round 효과만 비교합니다.

MTEB runner는 PRP와 TourRank method에 `AsyncAzureOpenAIReranker`를
사용합니다. `--concurrency`는 독립적인 query-method 실행을 병렬화하며,
각 Strategy 내부의 순회 방식과 호출 수는 바꾸지 않습니다.

### TourRank-r Live Smoke Snapshot

아래 비교는 TourRank-r 실행 경로를 확인하기 위해 실제 live Azure로 수행한
smoke 결과입니다. `AskUbuntuDupQuestions` query 1개만 사용했으므로 품질
결론이 아니라 integration과 호출 수 확인 자료로 해석해야 합니다.

```bash
uv run python scripts/evaluate_mteb_reranking.py \
  --tasks AskUbuntuDupQuestions \
  --methods \
    original \
    rankgpt_sliding_window@20 \
    prp_sliding_k@20 \
    tourrank_r@20:r2 \
    tourrank_r@20:r10 \
  --output-dir benchmark-results/mteb-reranking/tourrank-smoke-20260520-121112 \
  --max-queries 1 \
  --max-document-chars 4000 \
  --shuffle-candidates --shuffle-seed 13 \
  --rankgpt-window-size 20 --rankgpt-step 10 \
  --prp-passes 10 \
  --concurrency 2 \
  --input-token-price-per-1m 2.50 \
  --output-token-price-per-1m 10.00 \
  --allow-live
```

Scope:

- Task: `AskUbuntuDupQuestions`
- Split: `test`
- Queries: `1`
- Candidate order: shuffled with seed `13`
- Max document length: `4000` characters
- Validation: strict JSON validation, invalid outputs score `0`
- Artifact: `benchmark-results/mteb-reranking/tourrank-smoke-20260520-121112`

| Method | NDCG@10 | MRR@10 | MAP | Recall@10 | p50 latency | Invalid rate | LLM calls/query | Queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | 0.7126 | 1.0000 | 0.7784 | 0.5000 | 0.0 ms | 0.000 | 0 | 1 |
| `rankgpt_sliding_window@20` | 0.9337 | 1.0000 | 0.9248 | 0.7500 | 3645.6 ms | 0.000 | 1 | 1 |
| `prp_sliding_k@20` | 0.7569 | 1.0000 | 0.8209 | 0.5833 | 198438.8 ms | 0.000 | 380 | 1 |
| `tourrank_r@20:r2` | 0.8630 | 1.0000 | 0.8941 | 0.6667 | 9285.5 ms | 0.000 | 8 | 1 |
| `tourrank_r@20:r10` | 0.8580 | 1.0000 | 0.8738 | 0.6667 | 41412.7 ms | 0.000 | 40 | 1 |

이 smoke run에서 두 TourRank-r variant는 모두 strict JSON validation을
통과했습니다. `tourrank_r@20:r10`은 `tourrank_r@20:r2`보다 selection 호출을
5배 사용했으며, 이는 설정한 round 수와 일치합니다.

### PRP vs RankGPT Snapshot

아래 비교는 같은 `AskUbuntuDupQuestions` 설정으로 실행했으며 결과 artifact는
`benchmark-results/mteb-reranking/n30-prp-vs-rankgpt-rerun`에 저장되어 있습니다.
이 결과는 native MTEB candidate set 기준입니다. 해당 task는 query당 후보를
20개만 제공하므로, 표준적인 top-100 RankGPT 설정은 아닙니다.

| Method | NDCG@10 | MRR@10 | MAP | Recall@10 | p50 latency | p95 latency | Invalid rate | LLM calls/query | Total LLM calls | Mean cost/query | Queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | 0.4431 | 0.5668 | 0.3895 | 0.5871 | 0.0 ms | 0.0 ms | 0.000 | 0 | 0 | - | 30 |
| `rankgpt_sliding_window@20` | 0.6830 | 0.6834 | 0.6400 | 0.7706 | 1842.6 ms | 2542.6 ms | 0.033 | 1 | 30 | $0.001530 | 30 |
| `prp_sliding_k@20` | 0.6714 | 0.7837 | 0.6132 | 0.7451 | 213583.6 ms | 230670.9 ms | 0.000 | 380 | 11,400 | $0.172772 | 30 |

RankGPT listwise가 NDCG@10, MAP, Recall@10, latency, cost에서 앞섰습니다.
PRP는 MRR@10에서 앞섰지만, `passes=10`과 후보 20개 기준 query마다 약 380회의
pairwise LLM 호출이 필요했습니다. Strict validation 정책에 따라 RankGPT의
invalid LLM output 1건은 zero-score로 집계했습니다.

일반적인 top-100 RankGPT 설정에서 `window_size=20`, `step=10`을 쓰면
`rankgpt_sliding_window@100`은 query당 listwise LLM 호출 9회가 필요합니다.
같은 후보 100개를 `prp_sliding_k@100`으로 비교하면
`2 * 10 * (100 - 1) = 1,980`회의 pairwise LLM 호출이 필요합니다.
