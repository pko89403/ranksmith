# 아키텍처

## 계층
- ModelProvider: vendor별 JSON completion 호출을 담당한다.
- ModelClient: ranksmith 도메인의 `rank`, `compare`, `select`, `answer` 계약과 prompt 생성을 담당한다.
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
    common.py              # shared validation/capability guards
    listwise.py
    pairwise.py
    setwise.py
    tourrank.py
    acurank.py
    confidence.py          # answer-confidence rerank (experimental)
    confidence_gain.py
    cbdr.py
  providers/
    __init__.py            # public provider exports
    azure.py               # Azure OpenAI implementation
  integrations/
    __init__.py            # public runtime helper exports
    azure_answer_generator.py
    answer_generator.py
    lmstudio_provider.py
    validation.py
```

외부 사용자는 root import 또는 `ranksmith.strategies`, `ranksmith.providers`, `ranksmith.integrations`의 public export를 사용한다.
strategies/, providers/, integrations/ 하위 개별 모듈은 내부 구현으로 취급한다.

## ModelProvider
실제 SDK 호출은 Azure OpenAI만 구현한다.
다른 vendor는 사용자 정의 `ModelProvider` 구현으로 연결한다.

Provider는 `ModelRequest`를 받아 `ModelResponse`를 반환한다.
Provider는 ranking 도메인 prompt의 의미를 알지 않는다.

## Integrations
`ranksmith.integrations`는 closed model runtime helper layer다.
Strategy나 Algorithm을 추가하지 않고, 기존 Strategy가 필요로 하는 외부 hook을 공식 조립 경로로 제공한다.

현재 범위:
- `AzureAnswerGenerator`: Azure OpenAI JSON answer generation helper
- `ProviderAnswerGenerator`: `ModelProvider` 기반 sync JSON answer generation helper
- `LMStudioModelProvider`: LM Studio OpenAI-compatible runtime helper
- confidence generation과 같은 no-answer sentinel prompt contract
- root import가 아닌 `ranksmith.integrations` submodule export

제외:
- async answer generation
- non-Azure hosted provider implementation
- scorer training

## ModelClient

Listwise model client call은 1-based ranking permutation을 담은 JSON 문자열을 반환한다.

Pairwise model client call은 `"A"` 또는 `"B"` winner를 담은 JSON 문자열을 반환한다.

Selection model client call은 group 안에서 선택된 문서의 1-based index list를 담은 JSON 문자열을 반환한다.

Answer model client call은 주어진 context만으로 질문에 답한 `"answer"` 문자열을 담은 JSON 문자열을 반환한다(문맥에 답이 없으면 `"__NO_ANSWER__"`).

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
- `AnswerConfidenceRerankStrategy` (실험적 — 학습된 estimator artifact 필요)
- `AsyncAnswerConfidenceRerankStrategy` (실험적)
- `ConfidenceGainStrategy`
- `CBDRStrategy`

공식 확장 지점:
- 새 reranking 방법은 새 Strategy 클래스로 추가한다.
- Strategy protocol과 model client/provider contract는 `ranksmith.protocols` 및 root import에서 공개한다.
- model client JSON ranking을 직접 다루는 custom Strategy는 `parse_ranking_response()`를 사용해 검증한다.
- selection 기반 Strategy는 `parse_selection_response()`를 사용해 selected index를 검증한다.
- answer 기반 Strategy는 `parse_answer_response()`를 사용해 answer 문자열을 검증한다.
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
- `confidence_gain`
- `cbdr`

benchmark opt-in algorithm:
- `answer_confidence` (`scripts/compare_reranking.py`, 학습된 artifact 필요)

향후 algorithm 후보:
- `Pointwise`

## Confidence
`ranksmith.confidence`(estimator)는 reranking Strategy가 아니라 closed model output confidence를 계산하는 utility layer다. 이 estimator를 소비하는 실험적 reranker가 `AnswerConfidenceRerankStrategy`(`ranksmith.strategies`)이며, 스펙에 기록된 소규모 자체 평가에서 Listwise에 진다(`docs/specs/spec_confidence_aware_reranking.md` — 증거 artifact는 리포에 커밋되어 있지 않다).

현재 범위:
- frozen HuggingFace encoder 기반 token-level trajectory 생성
- `structural-v1` 70차원 feature extraction
- 학습된 scorer artifact 기반 single-item sync confidence inference
- bounded batch sync confidence inference
- `answer_confidence`
- `judgment_confidence`
- `query_answerability_confidence`
- `query_context_answerability_confidence`
- root import가 아닌 `ranksmith.confidence` submodule export

`score_batch(..., max_workers>1)`은 같은 encoder/scorer instance를 worker thread들이 공유하므로, concurrent call에 안전한 backend에서만 사용한다. 기본값은 안정성을 위해 `max_workers=1`이다.

제외:
- semantic feature fusion
- async inference
- reranking Strategy

`ConfidenceGainStrategy`는 confidence utility layer 자체가 아니라, `ranksmith.confidence`의 query-only 및 query+context answerability scorer를 소비하는 별도 sync Strategy다.
`Conf(Q+C)-Conf(Q)`를 계산해 confidence gain 내림차순으로 문서를 정렬한다.

`CBDRStrategy`는 `Conf(Q)`가 `skip_threshold` 이상이면 context reranking을 skip하고 original order를 보존한다.
`Conf(Q)`가 threshold보다 낮으면 `Conf(Q+C)-Conf(Q)` confidence gain으로 문서를 정렬한다.
true pre-retrieval skip, retriever integration, async CBDR은 구현하지 않는다.

`ranksmith.confidence_training`은 Phase 1 compatible scorer artifact를 만들기 위한 별도 training utility layer다.

현재 범위:
- task별 canonical JSONL validation/loading
- deterministic train/valid/test split
- frozen HuggingFace encoder와 `structural-v1` feature extraction 재사용
- LightGBM binary classifier training
- validation split 기반 sigmoid calibration
- Phase 1 `ScorerMetadata` compatible artifact export
- CBDR용 answerability task CLI wrapper
- source/group balance dataset report helper

제외:
- 외부 benchmark/source adapter
- label 생성
- reranking Strategy 또는 Algorithm

`ranksmith.confidence_generation`은 closed model output을 생성해 confidence training canonical JSONL로 저장하는 utility layer다.

현재 범위:
- answer-oriented raw JSONL -> `answer_confidence` canonical JSONL
- relevance-oriented raw JSONL -> `judgment_confidence` canonical JSONL
- query-only answerability raw JSONL -> `query_answerability_confidence` canonical JSONL
- query+context answerability raw JSONL -> `query_context_answerability_confidence` canonical JSONL
- sync closed model call
- resume 가능한 JSONL output
- CBDR용 answerability task CLI wrapper

제외 (package 범위 — SQuAD 학습 데이터 빌더와 turnkey CLI는 package 밖 `scripts/build_answer_confidence_training_data.py`, `scripts/train_answer_confidence.py`로 제공):
- async generation
- dataset adapter
- runtime reranking Strategy 또는 Algorithm

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

Answer JSON:

```json
{"answer": "vitamin C deficiency"}
```

잘못된 JSON, 누락 값, 중복 값, 범위 밖 값, 정수가 아닌 값, 잘못된 winner 값은 `RerankParseError`로 실패한다.
`true`, `false`는 JSON bool이며 정수로 인정하지 않는다.
