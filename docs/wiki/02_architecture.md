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

## Strategy
v1 공개 strategy:
- `ListwiseStrategy`
- `AsyncListwiseStrategy`
- `PairwiseStrategy`
- `AsyncPairwiseStrategy`

향후 strategy 후보:
- `PointwiseStrategy`

## Algorithm
v1 지원 algorithm:
- `rankgpt_sliding_window`
- `prp_sliding_k`

향후 algorithm 후보:
- `tournament`
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

잘못된 JSON, 누락 값, 중복 값, 범위 밖 값, 정수가 아닌 값, 잘못된 winner 값은 `RerankParseError`로 실패한다.
