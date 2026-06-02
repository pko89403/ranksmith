# Spec: Structural Confidence Module

## 1. 개요 (Overview)
- **작업 목적**: closed model의 hidden state, attention, logits에 접근하지 못하는 상황에서, closed model 출력의 confidence를 추정하는 공식 utility module을 추가한다.
- **Reference**:
  - `docs/wiki/references/Trust in One Round- Confidence Estimation for Large Language Models via Structural Signals.pdf`
- **상태**: `[x] Draft` | `[ ] In Progress` | `[ ] Completed`

이 기능은 reranking `Strategy`가 아니다.
새 reranking `Algorithm`도 아니다.

`ranksmith.confidence`는 closed model 출력에 대한 confidence score를 계산하는 독립 모듈이다.
향후 Pointwise, validator, confidence-aware reranker가 이 score를 사용할 수 있다.

## 2. 요구 사항 및 제약 (Requirements & Constraints)

### 범위
이번 스펙은 **Phase 1: inference core**만 다룬다.

포함한다:
- HuggingFace `AutoTokenizer` / `AutoModel` 기반 frozen encoder 로딩.
- final hidden states 기반 structural feature extraction.
- scorer protocol 주입.
- LightGBM/joblib artifact loader.
- `answer_confidence`와 `judgment_confidence` input type 분리.
- scorer metadata 검증.
- single-item sync inference.

포함하지 않는다:
- LightGBM 학습 파이프라인.
- dataset 생성.
- label 생성.
- feature cache.
- calibration/evaluation pipeline.
- reranking Strategy 추가.
- benchmark 수치 문서화.
- semantic feature extraction.
- batch inference.
- async API.

학습 파이프라인은 **Phase 2 별도 스펙**에서 다룬다.

### 범위 고정 (Scope Freeze)
Phase 1 범위는 아래 항목으로 고정한다.

구현한다:
- `ranksmith.confidence` submodule.
- `AnswerConfidenceInput`.
- `JudgmentConfidenceInput`.
- `StructuralConfidenceEstimator.score()` 단건 sync inference.
- HuggingFace frozen encoder wrapper.
- `structural-v1` 70차원 structural feature extraction.
- scorer protocol.
- joblib wrapper loader.
- LightGBM Booster + metadata JSON loader.
- scorer/encoder/template/feature metadata compatibility validation.
- confidence-specific error.
- optional dependency lazy import.
- HuggingFace token 비노출 처리.
- CPU-only inference.
- minimal README / README.ko usage 문서.

구현하지 않는다:
- training pipeline.
- dataset / label 생성.
- feature cache.
- calibration / evaluation pipeline.
- semantic feature 또는 Struct+Sent fusion.
- batch inference.
- async API.
- reranking Strategy.
- benchmark 수치 또는 성능 claim.
- non-CPU device support.
- root import export.
- artifact save/export helper.

Phase 1 구현 중 위 제외 항목이 필요해 보이면 구현하지 않고 새 스펙으로 분리한다.
이 스펙의 완료 조건은 “학습된 scorer artifact가 주어졌을 때 single input confidence score를 계산할 수 있음”이다.

### 입력 (Inputs)
공통 설정:
- `encoder_name: str`
  - 기본값: `"bert-base-uncased"`
  - 실제 로딩은 HuggingFace `AutoTokenizer.from_pretrained()`와 `AutoModel.from_pretrained()`로 수행한다.
- `encoder_revision: str | None`
  - 기본값: `None`
  - scorer artifact와 tokenizer/model revision을 맞추기 위한 metadata 필드다.
- `tokenizer_name: str | None`
  - 기본값: `None`
  - `None`이면 `encoder_name`과 같은 값을 사용한다.
- `tokenizer_revision: str | None`
  - 기본값: `None`
  - tokenizer revision을 scorer artifact와 맞추기 위한 metadata 필드다.
- `hf_token: str | None`
  - 기본값: `None`
  - HuggingFace private/gated model 접근용 token이다.
  - `None`이면 HuggingFace 기본 환경변수/캐시 동작에 맡긴다.
- `local_files_only: bool`
  - 기본값: `False`
  - `True`이면 HuggingFace Hub 네트워크 접근 없이 로컬 캐시/경로만 사용한다.
- `cache_dir: str | None`
  - 기본값: `None`
  - HuggingFace tokenizer/model cache directory로 전달한다.
- `device: str`
  - 기본값: `"cpu"`
  - Phase 1은 `"cpu"`만 공식 지원한다.
  - `"cuda"`, `"mps"` 등은 Phase 1에서 fast fail 한다.
- `max_length: int`
  - 기본값: `256`
- `task_type: Literal["answer_confidence", "judgment_confidence"]`
- `scorer`
  - protocol 주입 또는 LightGBM/joblib artifact loader 결과.

`AnswerConfidenceInput`:
- `context: str`
- `answer: str`

`JudgmentConfidenceInput`:
- `query: str`
- `document: str`
- `judgment: str`

내부 type alias:
- `StructuralConfidenceInput = AnswerConfidenceInput | JudgmentConfidenceInput`

### 출력 (Outputs)
`StructuralConfidenceEstimator.score(...)`는 confidence result를 반환한다.

필수 필드:
- `score: float`
  - `0.0 <= score <= 1.0`
- `task_type: str`
- `feature_schema_version: str`
- `metadata: dict[str, object]`

metadata 예:
- `encoder_name`
- `encoder_revision`
- `tokenizer_name`
- `tokenizer_revision`
- `max_length`
- `feature_dim`
- `feature_dtype`
- `granularity`
- `input_template_version`
- `scorer_type`

### 제약 사항 (Constraints)
- encoder를 학습하지 않는다. Encoder는 frozen으로 사용한다.
- scorer는 특정 encoder, tokenizer, task type, input template, feature schema에 맞춰 학습된 artifact여야 한다.
- scorer metadata가 estimator 설정과 다르면 fast fail 한다.
- token truncation은 기본적으로 허용하지 않는다.
- 명시적으로 `allow_truncation=True`인 경우에만 tokenizer truncation을 허용한다.
- `max_length`는 `34` 이상이어야 한다.
  - `structural-v1`은 non-trivial low frequency `k = 1..16`을 고정 사용하므로, 안정적인 FFT schema를 위해 최소 길이를 둔다.
- required text field는 `strip()` 후 빈 문자열이면 허용하지 않는다.
- core dependency에 `torch`, `transformers`, `lightgbm`, `joblib`, `numpy`, `scipy`를 추가하지 않는다.
- confidence 관련 dependency는 optional extra로 분리한다.
- `confidence` optional extra는 inference에 필요한 dependency만 포함한다.
  - `torch`
  - `transformers`
  - `numpy`
  - `scipy`
  - `joblib`
  - `lightgbm`
- `import ranksmith.confidence`는 core install에서도 성공해야 한다.
- `torch`, `transformers`, `lightgbm`, `joblib`, `numpy`, `scipy`는 실제 encoder/scorer/feature 기능을 호출할 때만 lazy import한다.
- HuggingFace token은 metadata, result, error message, serialized artifact에 저장하지 않는다.
- HuggingFace token은 `from_pretrained(..., token=hf_token)` 호출에만 전달한다.
- token이 필요한 모델에서 인증이 실패하면 token 값을 노출하지 않는 `ConfidenceDependencyError` 또는 encoder load error로 실패한다.
- 구현은 외부 reference code를 복사하지 않는다.
- Phase 1은 structural-only confidence만 지원한다.
  - Trust 원문의 semantic feature 또는 Struct+Sent fusion은 구현하지 않는다.
  - semantic feature를 추가하려면 별도 feature schema와 scorer artifact schema가 필요하다.
- Phase 1은 `score()` 단건 sync API만 지원한다.
  - `score_batch()`와 async API는 Phase 2 이후 별도 스펙에서 다룬다.

## 3. 상세 설계 (Architecture & Design)

### 모듈 구조
```text
src/ranksmith/confidence/
  __init__.py
  _types.py          # input/result/metadata dataclass
  _errors.py         # confidence-specific errors
  _dependencies.py   # optional dependency lazy import helpers
  _templates.py      # task-specific input template formatting
  _features.py       # structural feature extraction
  _encoder.py        # HuggingFace frozen encoder wrapper
  _scorer.py         # scorer protocol + LightGBM/joblib loader
  _structural.py     # StructuralConfidenceEstimator
```

### Public API
```python
from ranksmith.confidence import (
    AnswerConfidenceInput,
    JudgmentConfidenceInput,
    StructuralConfidenceEstimator,
    StructuralConfidenceResult,
    StructuralConfidenceScorer,
    load_lightgbm_scorer,
)
```

### 사용 예
```python
from ranksmith.confidence import (
    AnswerConfidenceInput,
    StructuralConfidenceEstimator,
    load_lightgbm_scorer,
)

scorer = load_lightgbm_scorer("structural-confidence.joblib")

estimator = StructuralConfidenceEstimator.from_pretrained(
    encoder_name="bert-base-uncased",
    encoder_revision=None,
    hf_token=None,
    cache_dir=None,
    device="cpu",
    scorer=scorer,
    task_type="answer_confidence",
    max_length=256,
)

result = estimator.score(
    AnswerConfidenceInput(
        context="The provided passage...",
        answer="The model's answer...",
    )
)
```

`judgment_confidence`:
```python
result = estimator.score(
    JudgmentConfidenceInput(
        query="who played karen in married to the mob?",
        document="Nancy Travis played Karen...",
        judgment="direct evidence",
    )
)
```

### 동작 메커니즘
1. 사용자가 `StructuralConfidenceEstimator`를 생성한다.
2. estimator는 frozen encoder와 scorer를 가진다.
3. scorer metadata와 estimator 설정을 검증한다.
4. 입력 task type에 맞게 text sequence를 구성한다.
5. tokenizer로 tokenization한다.
6. `allow_truncation=False`이면 truncation 없이 tokenization한 뒤 길이를 검사한다.
7. token 길이가 `max_length`를 넘으면 encoder forward 전에 실패한다.
8. `allow_truncation=True`이면 명시적으로 tokenizer truncation을 적용한다.
9. frozen encoder forward pass로 final hidden states를 얻는다.
10. hidden-state trajectory에서 structural features를 계산한다.
11. scorer가 feature vector를 받아 confidence score를 예측한다.
12. score와 metadata를 `StructuralConfidenceResult`로 반환한다.

### Input template
`input_template_version = "structural-template-v1"`

`answer_confidence`:
```text
Context:
{context}

Answer:
{answer}
```

`judgment_confidence`:
```text
Query:
{query}

Document:
{document}

Judgment:
{judgment}
```

입력 template은 feature trajectory에 직접 영향을 준다.
학습과 추론 artifact는 같은 `input_template_version`을 사용해야 한다.
`AnswerConfidenceInput`은 `answer_confidence` estimator에서만 허용한다.
`JudgmentConfidenceInput`은 `judgment_confidence` estimator에서만 허용한다.

### Feature schema
`feature_schema_version = "structural-v1"`

`structural-v1`은 70차원 `float64` vector를 만든 뒤 scorer 입력 직전에 1차원 `list[float]`로 변환한다.
feature 계산에는 padding token을 제외한 모든 tokenizer output token을 사용한다.
special token은 tokenizer가 만든 실제 sequence의 일부로 보고 포함한다.
encoder output tensor는 `detach()` 후 CPU로 이동하고 `float64` numpy array로 변환한 뒤 feature 계산에 사용한다.

70차원 feature는 encoder hidden size와 다른 개념이다.
예를 들어 `bert-base-uncased`의 final hidden states는 `T x 768` 형태지만, `structural-v1`은 이 token-level trajectory의 구조를 70개 scalar descriptor로 요약한다.
즉 `768`은 token hidden vector 차원이고, `70`은 scorer가 입력으로 받는 engineered structural feature 차원이다.

feature 순서는 고정한다.

1. `spectral_stability` 48차원
   - `frequency_domain_smoothness` 32차원
     - hidden trajectory `H ∈ R^(T x D)`를 token axis 기준으로 zero-pad하여 `max_length`까지 맞춘다.
     - real FFT를 token axis에 적용하고, FFT 값은 padded length로 나누어 scale을 고정한다.
     - non-trivial low frequency `k = 1..16`을 사용한다.
     - 각 `k`마다 hidden dimension별 power `abs(fft[k]) ** 2`의 `mean`, `max`를 이 순서로 기록한다.
   - `graph_spectral_diffusion` 16차원
     - non-padding token hidden vector를 L2 normalize한다.
     - cosine similarity matrix를 만들고 음수 similarity는 `0`으로 clamp한다.
     - self-loop는 `0`으로 둔다.
     - normalized Laplacian `L = I - D^(-1/2) W D^(-1/2)`를 계산한다.
     - degree가 `0`인 node의 inverse sqrt degree는 `0.0`으로 처리한다.
     - 가장 작은 eigenvalue 16개를 오름차순으로 기록한다.
     - 작은 수치 오차로 생긴 eigenvalue의 음수값은 `abs(value) <= 1e-12`일 때 `0.0`으로 clamp한다.
     - `abs(value) > 1e-12`인 음수 eigenvalue, NaN, Inf는 실패한다.
     - token 수가 부족해 16개를 만들 수 없으면 뒤를 `0.0`으로 padding한다.
2. `local_variation` 6차원
   - 연속 token displacement `Δ_t = ||h_t - h_(t-1)||_2`를 사용한다.
   - 순서:
     1. total path length: `sum(Δ_t)`
     2. mean displacement: `mean(Δ_t)`
     3. displacement variance: `var(Δ_t)`
     4. start-end distance: `||h_T - h_1||_2`
     5. embedding-wise variance: hidden dimension별 variance의 평균
     6. centroid norm: `||mean(H, axis=0)||_2`
3. `shape_coherence` 16차원
   - 모든 token pair distance `||h_i - h_j||_2`를 계산한다.
   - max distance가 `0`보다 크면 pair distance를 max distance로 나누어 `[0, 1]`에 맞춘다.
   - `[0, 1]` 구간을 16개 equal-width bin으로 나누고 normalized histogram을 기록한다.
   - pair가 없으면 16개 모두 `0.0`으로 둔다.

수치 안정성 규칙:
- hidden states, intermediate matrix, final feature vector에 NaN 또는 Inf가 있으면 실패한다.
- non-padding token이 `0`개면 실패한다.
- non-padding token이 `1`개면 local variation displacement와 pair distance는 empty로 보고 정의된 zero fallback을 사용한다.
- feature 계산 결과는 반드시 길이 `70`이고 모든 값이 finite여야 한다.

위 세부 계산 규칙은 Trust 논문의 descriptor family를 ranksmith에서 재현 가능하게 고정한 `structural-v1` schema다.
논문이 명시하지 않은 clamp, self-loop, FFT scaling, histogram normalization 같은 세부 선택은 scorer artifact compatibility를 위해 ranksmith schema에 포함한다.
따라서 이 값을 바꾸려면 `feature_schema_version`을 새로 올려야 한다.

기본 granularity는 Trust 논문의 기본 설정을 따라 `two_scale`로 둔다.
- global descriptor: 전체 trajectory에서 70차원 descriptor를 계산한다.
- local descriptor: window size `5`, stride `2`의 overlapping window마다 70차원 descriptor를 계산하고 평균한다.
- token 수가 5보다 작으면 전체 trajectory를 하나의 local window로 사용한다.
- 최종 descriptor는 `(global_descriptor + local_descriptor) / 2`다.

metadata에는 다음 값을 반드시 기록한다.
- `feature_schema_version = "structural-v1"`
- `feature_dim = 70`
- `feature_dtype = "float64"`
- `granularity = "two_scale"`
- `local_window_size = 5`
- `local_stride = 2`
- `max_length`

### 의사 알고리즘 (Pseudo-algorithm)
```text
input = AnswerConfidenceInput(...) or JudgmentConfidenceInput(...)

if task_type == "answer_confidence":
    require AnswerConfidenceInput
    sequence = format_answer_sequence(context, answer)

if task_type == "judgment_confidence":
    require JudgmentConfidenceInput
    sequence = format_judgment_sequence(query, document, judgment)

tokens = tokenize(sequence, max_length, allow_truncation)
hidden_states = frozen_encoder(tokens).last_hidden_state
features = extract_structural_features(hidden_states, attention_mask)
score = scorer.predict_confidence(features)

return StructuralConfidenceResult(score, metadata)
```

### 의사 코드 (Pseudo-code)
```python
class StructuralConfidenceEstimator:
    @classmethod
    def from_pretrained(
        cls,
        *,
        encoder_name: str = "bert-base-uncased",
        encoder_revision: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        hf_token: str | None = None,
        local_files_only: bool = False,
        cache_dir: str | None = None,
        device: str = "cpu",
        scorer: StructuralConfidenceScorer,
        task_type: str,
        max_length: int = 256,
        allow_truncation: bool = False,
    ) -> "StructuralConfidenceEstimator":
        encoder = FrozenAutoEncoder.from_pretrained(
            encoder_name=encoder_name,
            encoder_revision=encoder_revision,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            hf_token=hf_token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            device=device,
            max_length=max_length,
            allow_truncation=allow_truncation,
        )
        validate_scorer_metadata(
            scorer.metadata,
            encoder_name=encoder_name,
            encoder_revision=encoder_revision,
            tokenizer_name=tokenizer_name or encoder_name,
            tokenizer_revision=tokenizer_revision,
            task_type=task_type,
            max_length=max_length,
            input_template_version="structural-template-v1",
            feature_schema_version="structural-v1",
            feature_dim=70,
        )
        return cls(encoder=encoder, scorer=scorer, task_type=task_type)

    def score(self, item: StructuralConfidenceInput) -> StructuralConfidenceResult:
        text = format_input(item, task_type=self.task_type)
        hidden_states, mask = self.encoder.encode(text)
        features = extract_structural_features(hidden_states, mask)
        score = self.scorer.predict_confidence(features)
        validate_score(score)
        return StructuralConfidenceResult(
            score=score,
            task_type=self.task_type,
            feature_schema_version="structural-v1",
            metadata={
                "encoder_name": self.encoder.encoder_name,
                "encoder_revision": self.encoder.encoder_revision,
                "tokenizer_name": self.encoder.tokenizer_name,
                "tokenizer_revision": self.encoder.tokenizer_revision,
                "max_length": self.encoder.max_length,
                "feature_dim": len(features),
                "feature_dtype": "float64",
                "granularity": "two_scale",
                "input_template_version": "structural-template-v1",
                "scorer_type": self.scorer.metadata.scorer_type,
                "artifact_schema_version": self.scorer.metadata.artifact_schema_version,
            },
        )
```

### Scorer protocol and artifact contract
`StructuralConfidenceScorer`는 estimator가 의존하는 최소 계약이다.

```python
class StructuralConfidenceScorer(Protocol):
    metadata: ScorerMetadata

    def predict_confidence(self, features: Sequence[float]) -> float:
        """Return calibrated confidence probability for one feature vector."""
        raise NotImplementedError
```

`ScorerMetadata` 필수 필드:
```python
@dataclass(frozen=True)
class ScorerMetadata:
    artifact_schema_version: str
    scorer_type: str
    task_type: Literal["answer_confidence", "judgment_confidence"]
    encoder_name: str
    encoder_revision: str | None
    tokenizer_name: str
    tokenizer_revision: str | None
    input_template_version: str
    feature_schema_version: str
    feature_dim: int
    feature_dtype: str
    max_length: int
    granularity: str
    local_window_size: int
    local_stride: int
    score_output: Literal["probability"]
    positive_class_index: int = 1
```

`artifact_schema_version = "structural-artifact-v1"`

artifact schema 규칙:
- metadata는 JSON-serializable dict여야 한다.
- joblib wrapper의 `"metadata"`도 JSON-serializable dict에서 `ScorerMetadata`로 변환 가능해야 한다.
- artifact에는 HuggingFace token, local filesystem credential, temporary path를 저장하지 않는다.
- `artifact_schema_version`이 estimator가 지원하는 값과 다르면 실패한다.

`load_lightgbm_scorer()`는 두 artifact 형태를 지원한다.

1. joblib wrapper
   - joblib 파일은 dict 형태여야 한다.
   - 필수 key:
     - `"model"`: `predict_proba()` 또는 probability `predict()`를 제공하는 객체
     - `"metadata"`: `ScorerMetadata`로 변환 가능한 dict
2. LightGBM Booster file
   - model file과 별도 metadata JSON을 함께 받는다.
   - `load_lightgbm_scorer(model_path, metadata_path=...)` 형태를 사용한다.

scoring 규칙:
- `predict_proba([features])`가 있으면 `positive_class_index` 열을 score로 사용한다.
- `predict_proba([features])` 결과는 2차원이어야 하며 첫 번째 차원은 `1`이어야 한다.
- `positive_class_index`는 output column 범위 안에 있어야 한다.
- `predict([features])`만 있으면 반환값은 이미 probability여야 한다.
- `predict([features])` 결과는 scalar 또는 길이 1 array여야 한다.
- raw margin, class label, decision score는 Phase 1에서 지원하지 않는다.
- 최종 score가 `0.0 <= score <= 1.0` 범위를 벗어나면 실패한다.
- 최종 score가 NaN 또는 Inf이면 실패한다.
- metadata의 필수 필드는 엄격하게 검증한다.
- metadata의 unknown field는 forward compatibility를 위해 보존하되 검증에는 사용하지 않는다.

### API scope
- Public input type은 `AnswerConfidenceInput`과 `JudgmentConfidenceInput`으로 분리한다.
- `StructuralConfidenceInput`이라는 union alias를 내부 type hint로 둘 수 있지만 public 예제에는 사용하지 않는다.
- `StructuralConfidenceEstimator.score()`는 단건 sync inference만 수행한다.
- `score_batch()`는 Phase 1에 추가하지 않는다.
- async estimator는 Phase 1에 추가하지 않는다.
- confidence 전용 error는 `ranksmith.confidence` submodule에서 public export한다.
- root import에는 confidence class/error를 추가하지 않는다.

### 통합 지점 (Integration Points)
- `pyproject.toml`
  - optional extra 추가:
    - `confidence`: inference dependency
    - `confidence-train`: Phase 2 후보로 문서만 언급하고 이번에는 추가하지 않는다.
- `src/ranksmith/confidence/`
  - 새 공식 confidence module.
- `src/ranksmith/__init__.py`
  - root export 여부는 최소화한다.
  - 1차 구현에서는 `ranksmith.confidence` submodule import를 기본으로 한다.
- `docs/wiki/02_architecture.md`
  - Confidence utility layer를 추가한다.
- `docs/wiki/04_references_index.md`
  - Trust reference 처리 상태를 요약 완료로 갱신한다.
- `README.md` / `README.ko.md`
  - 공식 module이므로 최소 섹션을 추가한다.
  - benchmark 수치나 성능 claim은 쓰지 않는다.
  - optional extra 설치와 minimal usage만 설명한다.

## 4. 재사용 및 모듈화 (Reusability & Modularization)

### 공통 컴포넌트 식별 (Shared Components)
- `AnswerConfidenceInput`
  - `answer_confidence` 전용 입력이다.
- `JudgmentConfidenceInput`
  - `judgment_confidence` 전용 입력이다.
- `StructuralConfidenceScorer`
  - LightGBM 외 scorer도 주입할 수 있게 protocol로 정의한다.
- `ScorerMetadata`
  - artifact compatibility 검증에 사용한다.
- feature extractor
  - 향후 training pipeline에서도 그대로 재사용한다.

### 추상화 방안 (Abstraction Plan)
- Encoder wrapper는 HuggingFace 객체를 직접 public API에 노출하지 않는다.
- HuggingFace token 처리는 `_encoder.py` 내부에서만 수행한다.
  - `hf_token`은 `AutoTokenizer.from_pretrained(..., token=hf_token)`와 `AutoModel.from_pretrained(..., token=hf_token)`에만 전달한다.
  - `hf_token`은 dataclass repr, result metadata, scorer metadata, error message에 포함하지 않는다.
  - `local_files_only=True`는 tokenizer/model 양쪽에 동일하게 전달한다.
  - `cache_dir`는 tokenizer/model 양쪽에 동일하게 전달한다.
  - Phase 1 `device`는 `"cpu"`만 허용한다.
- scorer loader는 LightGBM/joblib에 한정하되, estimator는 scorer protocol만 의존한다.
- feature schema version을 명시해 향후 feature 변경 시 artifact mismatch를 감지한다.
- input template formatting은 `_templates.py`로 분리한다.
  - estimator는 task type에 맞는 formatter만 호출한다.
  - formatter는 required field 검증과 template 적용을 함께 담당한다.
- optional dependency import는 `_dependencies.py`에 모은다.
  - core install에서 module import가 깨지지 않도록 feature 호출 시점에만 dependency를 확인한다.
- scorer metadata 검증은 `_scorer.py` 또는 `_structural.py`의 별도 helper로 분리한다.
  - estimator 생성 시 metadata mismatch를 모두 검증한다.
  - scoring 직후 feature length와 score range를 다시 검증한다.
- estimator는 runtime mutable state를 갖지 않는 것을 원칙으로 한다.
  - encoder와 scorer 객체 참조 외에 요청별 mutable cache를 두지 않는다.
  - `score()`는 입력을 변형하지 않는다.
  - thread safety를 깨는 내부 mutation은 금지한다.

## 5. 에러 핸들링 (Error Handling)

새 error 후보:
- `ConfidenceError`
- `ConfidenceDependencyError`
- `ConfidenceInputError`
- `ConfidenceArtifactError`

에러 정책:
- optional dependency가 없으면 `ConfidenceDependencyError`.
- HuggingFace 인증 실패, gated model 접근 실패, model load 실패는 token 값을 숨긴 에러로 실패한다.
- `device != "cpu"`이면 `ConfidenceInputError`.
- task type과 입력 필드가 맞지 않으면 `ConfidenceInputError`.
- `answer_confidence` estimator에 `JudgmentConfidenceInput`이 들어오면 `ConfidenceInputError`.
- `judgment_confidence` estimator에 `AnswerConfidenceInput`이 들어오면 `ConfidenceInputError`.
- required text field가 `strip()` 후 비어 있으면 `ConfidenceInputError`.
- `max_length < 34`이면 `ConfidenceInputError`.
- non-padding token이 `0`개면 `ConfidenceInputError`.
- scorer metadata가 없거나 불완전하면 `ConfidenceArtifactError`.
- scorer metadata가 JSON-serializable dict가 아니면 `ConfidenceArtifactError`.
- scorer metadata의 `artifact_schema_version`이 지원 값과 다르면 `ConfidenceArtifactError`.
- scorer metadata와 estimator 설정이 다르면 `ConfidenceArtifactError`.
- scorer metadata의 `feature_dim`이 실제 feature 길이와 다르면 `ConfidenceArtifactError`.
- scorer metadata의 `input_template_version`이 estimator와 다르면 `ConfidenceArtifactError`.
- scorer metadata의 `tokenizer_name` 또는 `tokenizer_revision`이 estimator와 다르면 `ConfidenceArtifactError`.
- scorer metadata의 `score_output`이 `"probability"`가 아니면 `ConfidenceArtifactError`.
- 입력이 `max_length`를 넘고 `allow_truncation=False`면 `ConfidenceInputError`.
- hidden states, intermediate feature 계산, final feature vector에 NaN 또는 Inf가 있으면 `ConfidenceArtifactError`.
- Laplacian eigenvalue 계산에서 허용 오차보다 큰 음수, NaN, Inf가 나오면 `ConfidenceArtifactError`.
- `predict_proba()` output shape이 잘못됐거나 positive class index가 범위 밖이면 `ConfidenceArtifactError`.
- `predict()` output이 scalar/length-1 probability가 아니면 `ConfidenceArtifactError`.
- scorer가 `0.0 <= score <= 1.0` 범위 밖 값을 반환하면 `ConfidenceArtifactError`.

`RerankInputError`와 직접 섞지 않는다.
이 모듈은 reranking Strategy가 아니므로 confidence-specific error를 둔다.

## 6. 테스트 계획 (Test Plan)

### 성공 케이스 (Happy Paths)
- fake encoder hidden states로 `extract_structural_features()`가 고정 길이 vector를 반환한다.
- `structural-v1` feature vector는 정확히 70차원이다.
- `structural-v1` feature 순서는 spectral 48차원, local 6차원, shape 16차원이다.
- `two_scale` descriptor는 global descriptor와 local-window 평균 descriptor의 평균이다.
- encoder output은 CPU `float64` numpy 기반 feature 계산으로 변환된다.
- graph degree가 0인 node가 있어도 finite feature를 만든다.
- non-padding token이 1개일 때 zero fallback으로 finite 70차원 feature를 만든다.
- fake scorer protocol로 `StructuralConfidenceEstimator.score()`가 score를 반환한다.
- `answer_confidence` estimator는 `AnswerConfidenceInput`을 정상 처리한다.
- `judgment_confidence` estimator는 `JudgmentConfidenceInput`을 정상 처리한다.
- scorer metadata가 estimator 설정과 일치하면 estimator 생성이 성공한다.
- `import ranksmith.confidence`는 optional dependency가 없어도 성공한다.
- `hf_token`은 result metadata에 포함되지 않는다.
- HuggingFace load error message에는 token 문자열이 포함되지 않는다.
- `cache_dir`는 tokenizer/model load 양쪽에 전달된다.
- `device="cpu"`는 정상 처리된다.
- scorer metadata의 unknown field는 보존되지만 검증에는 쓰이지 않는다.

### 엣지/실패 케이스 (Edge & Failure Cases)
- `answer_confidence`에서 `answer`가 없으면 실패한다.
- `judgment_confidence`에서 `document` 또는 `judgment`가 없으면 실패한다.
- `answer_confidence` estimator에 `JudgmentConfidenceInput`을 넣으면 실패한다.
- `judgment_confidence` estimator에 `AnswerConfidenceInput`을 넣으면 실패한다.
- `device != "cpu"`이면 실패한다.
- `score_batch()` API는 존재하지 않는다.
- metadata의 `encoder_name`이 다르면 실패한다.
- metadata의 `encoder_revision`이 다르면 실패한다.
- metadata의 `tokenizer_name`이 다르면 실패한다.
- metadata의 `tokenizer_revision`이 다르면 실패한다.
- metadata의 `task_type`이 다르면 실패한다.
- metadata의 `input_template_version`이 다르면 실패한다.
- metadata의 `feature_schema_version`이 다르면 실패한다.
- metadata의 `feature_dim`이 실제 feature 길이와 다르면 실패한다.
- metadata의 `max_length`가 다르면 실패한다.
- metadata의 `granularity`가 다르면 실패한다.
- metadata의 `local_window_size`가 다르면 실패한다.
- metadata의 `local_stride`가 다르면 실패한다.
- metadata의 `artifact_schema_version`이 다르면 실패한다.
- metadata가 JSON-serializable dict가 아니면 실패한다.
- `max_length < 34`이면 실패한다.
- required text field가 whitespace-only이면 실패한다.
- hidden states 또는 feature vector에 NaN/Inf가 있으면 실패한다.
- Laplacian eigenvalue에 허용 범위 밖 음수/NaN/Inf가 있으면 실패한다.
- `predict_proba()` output shape이 잘못되면 실패한다.
- `positive_class_index`가 범위 밖이면 실패한다.
- `predict()` output이 scalar/length-1 probability가 아니면 실패한다.
- `allow_truncation=False`에서 token 길이가 초과되면 실패한다.
- `allow_truncation=False`에서는 encoder forward 전에 token 길이 초과가 감지된다.
- optional dependency가 없을 때 loader를 호출하면 명확한 에러가 난다.
- `local_files_only=True`는 tokenizer/model load 양쪽에 전달된다.
- `hf_token`은 tokenizer/model load 양쪽에 전달되지만 저장되거나 노출되지 않는다.
- scorer artifact가 raw margin 또는 class label을 반환하면 실패한다.
- scorer가 범위 밖 score를 반환하면 실패한다.

### 공통 Reranking Smoke/Benchmark
- 이번 기능은 reranking algorithm이 아니므로 `tests/fixtures/reranking_smoke_fixture.jsonl` 기반 metric test는 추가하지 않는다.
- `scripts/compare_reranking.py`에는 추가하지 않는다.
- 실제 HuggingFace 모델 로딩 test는 opt-in으로만 실행한다.
  - opt-in env var: `RANKSMITH_RUN_HF_TESTS=1`
  - 기본 test run에서는 fake encoder/mock loader만 사용한다.

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] Trust 논문의 적용 범위와 ranksmith 내 위치 확인
- [x] Phase 1 scope freeze 반영
- [ ] 사용자 스펙 검토 및 최종 승인

### Phase 2: 로직 구현 (Implementation)
- [ ] `pyproject.toml`: confidence optional dependency 추가
- [ ] `src/ranksmith/confidence/__init__.py`: public confidence module export 추가
- [ ] `src/ranksmith/confidence/_types.py`: input/result/metadata dataclass 구현
  - [ ] `AnswerConfidenceInput` 구현
  - [ ] `JudgmentConfidenceInput` 구현
  - [ ] confidence 전용 error submodule export 결정 반영
- [ ] `src/ranksmith/confidence/_errors.py`: confidence-specific error 구현
- [ ] `src/ranksmith/confidence/_dependencies.py`: optional dependency lazy import helper 구현
- [ ] `src/ranksmith/confidence/_templates.py`: `answer_confidence` / `judgment_confidence` input template 구현
- [ ] `src/ranksmith/confidence/_features.py`: structural feature extraction 구현
  - [ ] spectral stability 48차원 구현
  - [ ] local variation 6차원 구현
  - [ ] shape coherence 16차원 구현
  - [ ] `two_scale` descriptor aggregation 구현
  - [ ] feature length/dtype validation 구현
  - [ ] NaN/Inf validation 구현
  - [ ] graph degree zero fallback 구현
  - [ ] eigenvalue tolerance validation 구현
  - [ ] single-token zero fallback 구현
- [x] `src/ranksmith/confidence/_encoder.py`: frozen HuggingFace AutoModel wrapper 구현
  - [x] tokenizer/model lazy import 구현
  - [x] `hf_token` 전달 및 비노출 정책 구현
  - [x] `local_files_only` 전달 구현
  - [x] `cache_dir` 전달 구현
  - [x] `device="cpu"` validation 구현
  - [x] encoder eval mode 및 gradient 비활성화 구현
  - [x] output tensor detach/CPU/float64 변환 구현
  - [x] `allow_truncation=False` token length preflight 구현
  - [x] attention mask 기반 padding 제외 처리 구현
- [ ] `src/ranksmith/confidence/_scorer.py`: scorer protocol과 LightGBM/joblib loader 구현
  - [ ] `ScorerMetadata` parsing/validation 구현
  - [ ] `artifact_schema_version` validation 구현
  - [ ] JSON-serializable metadata validation 구현
  - [ ] joblib wrapper artifact loader 구현
  - [ ] LightGBM Booster + metadata JSON loader 구현
  - [ ] `predict_proba` positive class score 추출 구현
  - [ ] `predict_proba` output shape validation 구현
  - [ ] probability-only `predict` path 구현
  - [ ] `predict` output shape validation 구현
  - [ ] unknown metadata field 보존 구현
- [ ] `src/ranksmith/confidence/_structural.py`: `StructuralConfidenceEstimator` 구현
  - [ ] estimator 설정과 scorer metadata mismatch 검증 구현
  - [ ] confidence result metadata 생성 구현
  - [ ] request별 mutable state 없음 확인

### Phase 3: 검증 (Verification)
- [ ] `tests/test_confidence_features.py`: feature extraction unit test 추가
- [ ] `tests/test_confidence_estimator.py`: estimator input/result/error test 추가
- [ ] `tests/test_confidence_scorer.py`: scorer protocol/metadata/loader test 추가
- [ ] `tests/test_confidence_api_scope.py`: input type 분리, no batch, no async, root export 제외 test 추가
- [ ] `tests/test_confidence_templates.py`: input template formatting 및 required field 검증 test 추가
- [ ] `tests/test_confidence_dependencies.py`: core install import 및 lazy dependency error test 추가
- [x] `tests/test_confidence_hf_token.py`: HuggingFace token 전달/비노출/local-only test 추가
- [ ] `tests/test_confidence_numeric_stability.py`: NaN/Inf/eigenvalue/degree-zero/single-token 안정성 test 추가
- [x] `tests/test_confidence_hf_options.py`: `cache_dir`, `device`, HF opt-in env 정책 test 추가
- [ ] optional dependency 관련 test 조건 분리
- [ ] `./scripts/verify.sh` 실행

### Phase 4: 완료 및 정리
- [ ] `docs/wiki/02_architecture.md`: Confidence utility layer 반영
- [ ] `docs/wiki/references/structural_confidence.md`: Trust reference 요약 작성
- [ ] `docs/wiki/04_references_index.md`: Trust reference 상태 갱신
- [ ] `README.md` / `README.ko.md`: optional extra와 minimal usage 반영
- [ ] 본 문서 최상단의 **상태**를 `Completed`로 변경

## 8. Phase 2 예고: Training Pipeline

다음 스펙에서 다룰 항목:
- training dataset schema
- `answer_confidence` label 생성 방식
- `judgment_confidence` label 생성 방식
- feature extraction batch runner
- feature cache artifact
- LightGBM CPU training
- calibration
- evaluation metric
- scorer artifact 저장 포맷

Phase 2는 이번 구현에 포함하지 않는다.
