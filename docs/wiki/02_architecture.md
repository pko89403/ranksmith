# 아키텍처

## 계층
- Provider: LLM을 어떻게 호출할지 담당한다.
- Strategy: reranking에 쓰는 비교 단위다.
- Algorithm: 최종 순위를 만드는 절차다.

## 현재 구조
```text
AzureOpenAIReranker
= Azure OpenAI provider + ListwiseStrategy + input normalization
```

## Provider
v1은 Azure OpenAI만 구현한다.

Provider는 1-based ranking permutation을 담은 JSON 문자열을 반환한다.

## Strategy
v1은 `ListwiseStrategy`만 공개한다.

향후 strategy 후보:
- `PointwiseStrategy`
- `PairwiseStrategy`

## Algorithm
v1 지원 algorithm:
- `direct`
- `sliding_window`

향후 algorithm 후보:
- `tournament`
- `bayesian`
- `confidence`

## LLM 응답 계약
JSON permutation만 유효하다:

```json
{"ranking": [3, 1, 2]}
```

잘못된 JSON, 누락 값, 중복 값, 범위 밖 값, 정수가 아닌 값은 `RerankParseError`로 실패한다.
