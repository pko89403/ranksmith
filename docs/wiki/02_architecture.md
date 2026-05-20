# 아키텍처

## 계층
- Provider: LLM을 어떻게 호출할지 담당한다.
- Strategy: reranking에 쓰는 비교 단위다.
- Algorithm: 최종 순위를 만드는 절차다.

## 현재 구조
```text
AzureOpenAIReranker
= Azure OpenAI provider + Strategy + input normalization
```

## Provider
v1은 Azure OpenAI만 구현한다.

Listwise provider call은 1-based ranking permutation을 담은 JSON 문자열을 반환한다.

Pairwise provider call은 `"A"` 또는 `"B"` winner를 담은 JSON 문자열을 반환한다.

Selection provider call은 group 안에서 선택된 문서의 1-based index list를 담은 JSON 문자열을 반환한다.

## Strategy
v1 공개 strategy:
- `ListwiseStrategy`
- `AsyncListwiseStrategy`
- `PairwiseStrategy`
- `AsyncPairwiseStrategy`
- `TourRankStrategy`
- `AsyncTourRankStrategy`

공식 확장 지점:
- 새 reranking 방법은 새 Strategy 클래스로 추가한다.
- Strategy protocol과 provider protocol은 `ranksmith.protocols` 및 root import에서 공개한다.
- provider JSON ranking을 직접 다루는 custom Strategy는 `parse_ranking_response()`를 사용해 검증한다.
- selection 기반 Strategy는 `parse_selection_response()`를 사용해 selected index를 검증한다.
- 자세한 확장 규칙은 `docs/wiki/08_custom_strategy_extension.md`를 따른다.

향후 strategy 후보:
- `PointwiseStrategy`

## Algorithm
v1 지원 algorithm:
- `rankgpt_sliding_window`
- `prp_sliding_k`
- `tourrank_r`

향후 algorithm 후보:
- `bayesian`
- `confidence`

## LLM 응답 계약
Listwise JSON permutation:

```json
{"ranking": [3, 1, 2]}
```

Pairwise JSON winner:

```json
{"winner": "A"}
```

Selection JSON:

```json
{"selected": [3, 1]}
```

잘못된 JSON, 누락 값, 중복 값, 범위 밖 값, 정수가 아닌 값, 잘못된 winner 값은 `RerankParseError`로 실패한다.
