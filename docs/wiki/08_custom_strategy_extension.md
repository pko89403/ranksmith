# 커스텀 Strategy 확장 가이드

## 목적
`ranksmith`의 공식 확장 지점은 Strategy다.

새 reranking 방법을 추가할 때는 기존 `ListwiseStrategy.algorithm` 문자열을 늘리기보다,
새 Strategy 클래스를 만든다. 이렇게 하면 비교 단위, model client 계약, 오류 정책을
섞지 않고 유지할 수 있다.

## 언제 새 Strategy를 만드는가
- 기존 listwise / pairwise 알고리즘과 순위 생성 절차가 다를 때.
- model client를 쓰지 않는 deterministic reranking을 만들 때.
- 자체 scoring, rule, hybrid ranking, tournament, confidence aggregation을 실험할 때.
- model client 응답 계약이 기존 `ranking` 또는 `winner`와 다를 때.

## Sync Strategy 계약
```python
from collections.abc import Sequence

from ranksmith import Document, RerankResult


class MyStrategy:
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: object,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        ...
```

규칙:
- `documents`는 이미 `Document`로 정규화되어 들어온다.
- 반환값은 `list[RerankResult]`다.
- `rank`는 1-based다.
- `original_index`는 입력 `documents` 기준 0-based다.
- `top_k`가 있으면 최종 결과에서 자른다.

## ModelClient를 쓰지 않는 Strategy
model client가 필요 없으면 `model_client: object`로 받고 사용하지 않는다.

```python
class LengthStrategy:
    def rerank(self, *, query, documents, model_client, top_k=None):
        del query, model_client
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
```

## ModelClient를 쓰는 Strategy
model client JSON ranking은 반드시 `parse_ranking_response()`로 검증한다.
selection 기반 model client JSON은 반드시 `parse_selection_response()`로 검증한다.

```python
from ranksmith import ModelClient, RerankProviderError, parse_ranking_response


class ModelClientBackedStrategy:
    def rerank(self, *, query, documents, model_client: ModelClient, top_k=None):
        try:
            raw_response = model_client.rank(query, list(documents))
        except TimeoutError as exc:
            raise RerankProviderError(str(exc)) from exc

        ranking = parse_ranking_response(
            raw_response,
            expected_count=len(documents),
        )
        ordered_indexes = [number - 1 for number in ranking]
        ...
```

중요:
- `parse_ranking_response()`는 invalid JSON, 누락, 중복, 범위 밖 ranking을 `RerankParseError`로 실패시킨다.
- custom Strategy 내부 model client 실패는 Strategy가 직접 `RerankProviderError`로 분류해야 한다.
- `AzureOpenAIReranker`는 custom Strategy 내부에서 발생한 임의 예외를 `RerankStrategyError`로 감싼다.

## Async Strategy 계약
비동기 Strategy는 같은 계약을 `async def rerank(...)`로 구현한다.

```python
from collections.abc import Sequence

from ranksmith import Document, RerankResult


class MyAsyncStrategy:
    async def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: object,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        ...
```

model client를 await해야 한다면 `AsyncModelClient`를 사용한다.

## 하지 않는다
- 잘못된 ranking을 조용히 보정하지 않는다.
- 긴 문서를 조용히 자르지 않는다.
- `rank`를 0-based로 반환하지 않는다.
- `original_index`를 재정렬 후 index로 바꾸지 않는다.
- 기존 `ListwiseStrategy.algorithm`에 임의 문자열을 추가해 새 알고리즘처럼 쓰지 않는다.
- 외부 reference가 부족한 algorithm 세부사항을 추론해서 구현하지 않는다.

## 테스트 체크리스트
커스텀 Strategy를 추가하거나 예제로 제공할 때 확인한다.

- public import가 되는가.
- model client 없이 동작하는 Strategy는 model client를 호출하지 않는가.
- model-backed Strategy는 `parse_ranking_response()`를 쓰는가.
- selection-based Strategy는 `parse_selection_response()`를 쓰는가.
- model client 실패를 `RerankProviderError`로 분류하는가.
- Strategy 내부 버그가 `RerankStrategyError`로 분류되는가.
- sync와 async 경로가 같은 계약을 지키는가.
- `./scripts/verify.sh`가 통과하는가.

## 실행 가능한 예제
- `examples/custom_strategy.py`
