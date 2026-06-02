# 아키텍처

## 계층
- ModelProvider: vendor별 JSON completion 호출을 담당한다.
- ModelClient: ranksmith 도메인의 `rank`, `compare`, `select` 계약과 prompt 생성을 담당한다.
- Strategy: reranking에 쓰는 비교 단위다.
- Algorithm: 최종 순위를 만드는 절차다.

## 현재 구조
```text
AzureOpenAIReranker
= AzureAOAIProvider + ModelClient + Strategy + input normalization
```

## 파일 구조
```text
src/ranksmith/
  azure.py                 # public reranker entry
  model.py                 # ranksmith domain model client
  parsing.py               # LLM response contract parser
  strategies/
    __init__.py            # public strategy exports
    _common.py             # shared validation/capability guards
    _listwise.py
    _pairwise.py
    _setwise.py
    _tourrank.py
    _acurank.py
  providers/
    __init__.py            # public provider exports
    _azure.py              # Azure OpenAI implementation
    _stubs.py              # unimplemented provider stubs
  _providers.py            # backward-compatible re-export layer
```

외부 사용자는 root import 또는 `ranksmith.strategies`, `ranksmith.providers`의 public export를 사용한다.
`strategies/_*.py`, `providers/_*.py`는 내부 구현 모듈로 취급한다.

## ModelProvider
실제 SDK 호출은 Azure OpenAI만 구현한다.
OpenAI, Anthropic, Gemini provider는 향후 구현을 위한 public stub이며 호출 시 fast fail 한다.

Provider는 `ModelRequest`를 받아 `ModelResponse`를 반환한다.
Provider는 ranking 도메인 prompt의 의미를 알지 않는다.

## ModelClient

Listwise model client call은 1-based ranking permutation을 담은 JSON 문자열을 반환한다.

Pairwise model client call은 `"A"` 또는 `"B"` winner를 담은 JSON 문자열을 반환한다.

Selection model client call은 group 안에서 선택된 문서의 1-based index list를 담은 JSON 문자열을 반환한다.

## Strategy
v1 공개 strategy:
- `ListwiseStrategy`
- `AsyncListwiseStrategy`
- `PairwiseStrategy`
- `AsyncPairwiseStrategy`
- `SetwiseStrategy`
- `AsyncSetwiseStrategy`
- `TourRankStrategy`
- `AsyncTourRankStrategy`
- `AcuRankStrategy`
- `AsyncAcuRankStrategy`

공식 확장 지점:
- 새 reranking 방법은 새 Strategy 클래스로 추가한다.
- Strategy protocol과 model client/provider contract는 `ranksmith.protocols` 및 root import에서 공개한다.
- model client JSON ranking을 직접 다루는 custom Strategy는 `parse_ranking_response()`를 사용해 검증한다.
- selection 기반 Strategy는 `parse_selection_response()`를 사용해 selected index를 검증한다.
- 자세한 확장 규칙은 `docs/wiki/08_custom_strategy_extension.md`를 따른다.

향후 strategy 후보:
- `PointwiseStrategy`

## Algorithm
v1 지원 algorithm:
- `rankgpt_sliding_window`
- `prp_sliding_k`
- `setwise_heapsort`
- `tourrank_r`
- `acurank`

향후 algorithm 후보:
- `confidence`

## Confidence
`ranksmith.confidence`는 reranking Strategy나 Algorithm이 아니라, closed model output confidence를 계산하는 utility layer다.

현재 범위:
- frozen HuggingFace encoder 기반 token-level trajectory 생성
- `structural-v1` 70차원 feature extraction
- 학습된 scorer artifact 기반 single-item sync confidence inference
- root import가 아닌 `ranksmith.confidence` submodule export

제외:
- training pipeline
- semantic feature fusion
- batch/async inference
- reranking Strategy

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
`true`, `false`는 JSON bool이며 정수로 인정하지 않는다.
