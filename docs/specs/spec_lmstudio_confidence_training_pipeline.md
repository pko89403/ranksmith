# Spec: LM Studio Confidence Training Pipeline

## 1. 개요 (Overview)
- **작업 목적**: MacBook에서 LM Studio로 로컬 LLM을 배포해 answerability confidence 학습 데이터를 생성하고, CBDR이 사용할 `Conf(Q)` / `Conf(Q+C)` scorer artifact를 재현 가능하게 만든다.
- **Reference**:
  - `docs/wiki/references/structural_confidence.md`
  - `docs/wiki/references/parametric_post_retrieval_confidence.md`
  - `docs/specs/spec_confidence_generation_pipeline.md`
  - `docs/specs/spec_confidence_training_pipeline.md`
  - `docs/specs/spec_confidence_gain_reranking.md`
  - `docs/specs/spec_cbdr_strategy.md`
  - `docs/specs/spec_cbdr_user_facing_integration.md`
  - LM Studio model page: `https://lmstudio.ai/models/google/gemma-4-12b`
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`

이번 작업은 새 reranking algorithm이 아니다.
이미 구현된 `confidence_generation`, `confidence_training`, `CBDRStrategy`를 로컬 LM Studio 환경에서 실제로 운용하기 위한 integration / CLI / run artifact layer다.

## 2. 요구 사항 및 제약 (Requirements & Constraints)

### 2.1 목표
- LM Studio OpenAI-compatible server를 `ModelProvider`로 사용할 수 있어야 한다.
- query-only answerability dataset과 query+context answerability dataset을 생성할 수 있어야 한다.
- 두 dataset에서 각각 scorer artifact를 학습할 수 있어야 한다.
- 학습 결과는 `CBDRStrategy.from_artifacts(...)`에 바로 연결 가능해야 한다.
- 단일 도메인 과적합을 피하기 위해 source / group 기반 리포트를 남겨야 한다.

### 2.2 입력 (Inputs)
- LM Studio runtime:
  - `base_url`: 기본값 `http://localhost:1234/v1`
  - `model`: 명시 인자 또는 환경 변수
  - `api_key`: LM Studio 호환용 optional 값
  - `timeout`
  - 권장 context length: `16384`
  - 권장 max output tokens: `128`
- raw generation dataset:
  - 권장 생성 경로: `scripts/build_qa_confidence_raw_dataset.py`
    - TriviaQA-style QA examples를 입력으로 사용한다.
    - positive context는 answer alias를 포함하는 evidence로 제한한다.
    - negative context는 질문 기준 TF-IDF retrieved context 중 현재 answer alias를 포함하지 않는 context로 생성한다.
    - synthetic fixed seed QA는 실제 학습 데이터로 사용하지 않는다.
  - `query_answerability_confidence`
    - `id`
    - `query`
    - `gold_answer`
    - optional `source`, `group_id`, `metadata`
  - `query_context_answerability_confidence`
    - `id`
    - `query`
    - `context`
    - `gold_answer`
    - optional `source`, `group_id`, `metadata`
- training config:
  - `task_type`
  - `dataset_path`
  - `output_dir`
  - `export_path`
  - HuggingFace encoder options
  - split options

### 2.3 출력 (Outputs)
권장 run directory:

```text
runs/confidence/<run_id>/
  raw/
    query_answerability.jsonl
    query_context_answerability.jsonl
  canonical/
    query_answerability_confidence.jsonl
    query_context_answerability_confidence.jsonl
  training/
    query_answerability/
      model.joblib
      metadata.json
      report.json
      report.md
    query_context_answerability/
      model.joblib
      metadata.json
      report.json
      report.md
  artifacts/
    query_answerability.joblib
    query_context_answerability.joblib
  reports/
    generation_summary.json
    dataset_balance.json
    generalization_report.json
```

### 2.4 제약 사항 (Constraints)
- Fast fail을 유지한다.
- JSON 응답을 조용히 보정하지 않는다.
- hidden state, logits, attention, logprobs에 의존하지 않는다.
- LM Studio 모델은 scorer가 아니라 label 생성을 위한 answer generator다.
- confidence scorer 학습은 기존 `FrozenAutoEncoder + structural-v1 features + LightGBM + sigmoid calibration`을 사용한다.
- public root import는 늘리지 않는다.
- `ranksmith.integrations` submodule export는 허용한다.
- scorer training 결과를 README benchmark 수치로 쓰려면 실제 benchmark summary artifact가 필요하다.

## 3. 상세 설계 (Architecture & Design)

### 3.1 전체 흐름
```text
TriviaQA/NQ-style QA evidence examples
  -> build_qa_confidence_raw_dataset.py
  -> raw QA/context examples
  -> LMStudioModelProvider
  -> confidence_generation canonical JSONL
  -> confidence_training scorer artifacts
  -> CBDRStrategy.from_artifacts(...)
  -> compare_reranking.py --algorithm cbdr
```

### 3.2 LM Studio Provider
새 provider는 `ModelProvider` protocol만 만족한다.

통합 지점:

```text
src/ranksmith/integrations/_lmstudio_provider.py
src/ranksmith/integrations/__init__.py
tests/test_lmstudio_provider.py
```

Public submodule API:

```python
from ranksmith.integrations import LMStudioModelProvider

provider = LMStudioModelProvider(
    base_url="http://localhost:1234/v1",
    model="google/gemma-4-12b",
    api_key="lm-studio",
    max_tokens=128,
    timeout=60,
)
```

기본 정책:
- `base_url`: `LMSTUDIO_BASE_URL` 또는 `http://localhost:1234/v1`
- `model`: 명시 인자 또는 `LMSTUDIO_MODEL`
- `api_key`: `LMSTUDIO_API_KEY` 또는 `"lm-studio"`
- `temperature`: `ModelRequest.temperature` 그대로 사용. generation pipeline에서는 이미 `0`을 전달한다.
- `response_format`: ranksmith의 `json_object` 요청을 LM Studio의 `json_schema` 요청으로 변환한다.
- `max_tokens`: 기본값 `128`, 명시 인자로 조정 가능.

LM Studio 0.4.16 기준 실제 확인 결과:

```text
response_format={"type":"json_object"} -> 실패
response_format={"type":"json_schema", ...} -> 성공
```

따라서 `LMStudioModelProvider`는 `ModelRequest.response_format == "json_object"`일 때 아래 schema를 전달한다.

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "ranksmith_json_response",
    "schema": {
      "type": "object"
    }
  }
}
```

answerability task의 최종 출력 형태 `{"answer":"..."}`는 기존 prompt와 strict parser가 검증한다.

LM Studio / Gemma 4 12B 운영 권장값:

```text
model: google/gemma-4-12b
context length: 16384
max output tokens: 128
temperature: 0
thinking: disabled when supported by LM Studio runtime
response_format: json_schema
output contract: {"answer":"..."}
```

`thinking`은 OpenAI-compatible 표준 필드가 아니므로 provider public config에 넣지 않는다.
LM Studio 런타임 UI 또는 모델 설정에서 끄는 운영 조건으로 문서화한다.

`query_context_answerability_confidence` 생성에서는 기본 `max_context_chars`를 `8000`으로 권장한다.
이는 LM Studio context length `16384`에서 system prompt, JSON schema, query 여유를 확보하기 위한 값이다.
단, ranksmith 원칙상 숨은 truncation은 하지 않는다.
`max_context_chars` 초과 입력은 기존 generation loader 정책대로 명시적으로 실패하거나, 사용자가 CLI 옵션으로 한도를 조정해야 한다.

### 3.3 QA Raw Dataset Builder
LM Studio 호출 전에 원천 QA dataset을 ranksmith generation raw schema로 변환한다.

통합 지점:

```text
scripts/build_qa_confidence_raw_dataset.py
```

초기 공식 지원:
- `--source triviaqa`
- HuggingFace `mandarjoshi/trivia_qa`, 기본 config `rc`
- 기본 split `train[:20000]`
- 기본 출력:
  - `query_answerability_raw.jsonl`
  - `query_context_answerability_raw.jsonl`
  - `dataset_manifest.json`

정책:
- positive query-context row는 정답 alias가 들어 있는 evidence context만 사용한다.
- negative query-context row는 질문과 유사한 retrieved context를 쓰되, 현재 정답 alias가 포함되면 제외한다.
- context가 `--max-context-chars`를 넘으면 조용히 자르지 않고 제외한다.
- 요청한 row 수를 만들 수 없으면 fast-fail한다.

예시:

```bash
uv run --with datasets python scripts/build_qa_confidence_raw_dataset.py \
  --source triviaqa \
  --dataset-name mandarjoshi/trivia_qa \
  --dataset-config rc \
  --split 'train[:20000]' \
  --output-dir runs/confidence/local/raw \
  --max-source-items 20000 \
  --max-query-items 5000 \
  --max-query-context-items 5000 \
  --max-context-chars 8000
```

### 3.4 Dataset Generation CLI
기존 `ranksmith.confidence_generation`은 library API만 제공한다.
로컬 실험을 재현 가능하게 하려면 얇은 CLI가 필요하다.

통합 지점:

```text
scripts/generate_confidence_dataset.py
tests/test_generate_confidence_dataset_script.py
```

지원 task:
- `query_answerability_confidence`
- `query_context_answerability_confidence`

초기 범위에서는 `answer_confidence`, `judgment_confidence` CLI는 추가하지 않는다.
CBDR에 필요한 두 task만 먼저 고정한다.

예시:

```bash
uv run python scripts/generate_confidence_dataset.py \
  --task query_answerability_confidence \
  --provider lmstudio \
  --lmstudio-base-url http://localhost:1234/v1 \
  --lmstudio-model google/gemma-4-12b \
  --input runs/confidence/local/raw/query_answerability.jsonl \
  --output runs/confidence/local/canonical/query_answerability_confidence.jsonl \
  --resume
```

```bash
uv run python scripts/generate_confidence_dataset.py \
  --task query_context_answerability_confidence \
  --provider lmstudio \
  --lmstudio-base-url http://localhost:1234/v1 \
  --lmstudio-model google/gemma-4-12b \
  --input runs/confidence/local/raw/query_context_answerability.jsonl \
  --output runs/confidence/local/canonical/query_context_answerability_confidence.jsonl \
  --max-context-chars 8000 \
  --resume
```

### 3.5 Training CLI
기존 `ranksmith.confidence_training`도 library API만 제공한다.
학습 재현성을 위해 CLI를 추가한다.

통합 지점:

```text
scripts/train_confidence_scorer.py
tests/test_train_confidence_scorer_script.py
```

예시:

```bash
uv run python scripts/train_confidence_scorer.py \
  --task query_answerability_confidence \
  --dataset runs/confidence/local/canonical/query_answerability_confidence.jsonl \
  --output-dir runs/confidence/local/training/query_answerability \
  --export-path runs/confidence/local/artifacts/query_answerability.joblib \
  --encoder-name bert-base-uncased \
  --max-length 256
```

```bash
uv run python scripts/train_confidence_scorer.py \
  --task query_context_answerability_confidence \
  --dataset runs/confidence/local/canonical/query_context_answerability_confidence.jsonl \
  --output-dir runs/confidence/local/training/query_context_answerability \
  --export-path runs/confidence/local/artifacts/query_context_answerability.joblib \
  --encoder-name bert-base-uncased \
  --max-length 256
```

### 3.6 Dataset Balance / Generalization Report
범용성을 주장하려면 단순 train/test metric만으로 부족하다.
source별, group별, held-out source별 리포트가 필요하다.

초기 구현은 새 학습 알고리즘을 만들지 않고 report utility만 추가한다.

통합 지점:

```text
src/ranksmith/confidence_training/_dataset_report.py
scripts/report_confidence_dataset.py
tests/test_confidence_training_dataset_report.py
```

리포트 항목:
- total sample count
- positive / negative count
- positive rate
- source별 count / positive rate
- group_id count
- duplicate id 여부
- missing source 비율
- missing group_id 비율

held-out generalization metric은 학습 pipeline 내부의 split 방식만으로는 충분하지 않다.
이번 범위에서는 다음을 명시한다.

```text
generalization claim = source-balanced dataset report + source-heldout manual run evidence
```

즉, CLI는 source별 분포를 보여주고, benchmark claim은 별도 실행 artifact가 있을 때만 문서화한다.

### 3.7 CBDR 연결
기존 `compare_reranking.py --algorithm cbdr`를 그대로 사용한다.
이번 작업은 CBDR algorithm을 변경하지 않는다.

예시:

```bash
uv run python scripts/compare_reranking.py \
  --dataset fixture \
  --algorithm cbdr \
  --cbdr-base-artifact runs/confidence/local/artifacts/query_answerability.joblib \
  --cbdr-context-artifact runs/confidence/local/artifacts/query_context_answerability.joblib \
  --allow-live
```

## 4. 재사용 및 모듈화 (Reusability & Modularization)

### 4.1 재사용 컴포넌트
- `ModelProvider`
- `ModelRequest`
- `ModelResponse`
- `confidence_generation` prompt / parser / labeling
- `confidence_training` dataset loader / feature extraction / trainer / artifact export
- `CBDRStrategy.from_artifacts(...)`

### 4.2 새 모듈 책임
- `LMStudioModelProvider`
  - LM Studio OpenAI-compatible API 호출만 담당한다.
  - answerability prompt 의미를 알지 않는다.
  - ranksmith의 `json_object` 요청을 LM Studio의 `json_schema` 요청으로 변환한다.
- `generate_confidence_dataset.py`
  - provider 조립과 generation config 생성을 담당한다.
  - label 계산은 기존 generation pipeline에 위임한다.
- `train_confidence_scorer.py`
  - CLI arg를 `ConfidenceTrainingConfig`로 변환한다.
  - 학습 본체는 기존 training pipeline에 위임한다.
- `dataset_report`
  - dataset 품질/분포 리포트만 담당한다.
  - scorer 학습이나 ranking correction을 하지 않는다.
- `build_qa_confidence_raw_dataset.py`
  - QA 원천 dataset을 ranksmith raw generation schema로 변환한다.
  - retrieved negative context 생성까지만 담당한다.
  - LM Studio 호출, canonical label 생성, scorer 학습은 하지 않는다.

## 5. 에러 핸들링 (Error Handling)

### 5.1 LM Studio Provider
- LM Studio 서버 연결 실패: `RerankProviderError`
- HTTP non-2xx: `RerankProviderError`
- 응답 JSON 파싱 실패: `RerankProviderError`
- `choices[0].message.content` 누락: `RerankProviderError`
- 빈 content: `RerankProviderError`
- `model` 누락: `RerankInputError`

### 5.2 Generation CLI
- 지원하지 않는 task: CLI parser error
- provider가 `lmstudio`가 아님: CLI parser error
- input schema 오류: 기존 `ConfidenceGenerationInputError`
- output 존재 + `--overwrite/--resume` 없음: 기존 fast fail
- malformed model output: 기존 `ConfidenceGenerationParseError`

### 5.2.1 QA Raw Dataset Builder
- HuggingFace `datasets` 미설치: 명시적 실행 방법과 함께 fast-fail
- usable evidence 부족: 요청 row 수를 채울 수 없으면 fast-fail
- context 길이 초과: 숨은 truncation 없이 제외
- retrieved negative 부족: 요청 row 수를 채울 수 없으면 fast-fail

### 5.3 Training CLI
- unsupported task: 기존 `ConfidenceTrainingConfigError`
- label이 한쪽 클래스만 존재: 기존 training error
- HF model load 실패: 기존 training error 또는 원 예외를 wrapped error로 유지
- export path 쓰기 실패: fast fail

### 5.4 Dataset Report
- canonical JSONL schema 오류: 기존 dataset loader error
- source/group 누락은 실패가 아니라 warning count로 기록한다.
- duplicate id는 실패로 처리한다.

## 6. 테스트 계획 (Test Plan)

### 6.1 Unit Tests
- `LMStudioModelProvider`
  - 정상 OpenAI-compatible response를 `ModelResponse`로 변환한다.
  - `LMSTUDIO_MODEL` env fallback을 사용한다.
  - `json_object` 요청을 `json_schema`로 변환한다.
  - `max_tokens` 기본값 `128`을 전달한다.
  - model 누락 시 `RerankInputError`.
  - HTTP error / malformed response / empty content는 `RerankProviderError`.
- `generate_confidence_dataset.py`
  - query-only task가 올바른 generation function을 호출한다.
  - query+context task가 올바른 generation function을 호출한다.
  - LM Studio provider 옵션이 전달된다.
  - `--resume`, `--overwrite`, `--max-items`, `--max-context-chars`가 config에 반영된다.
- `train_confidence_scorer.py`
  - CLI args가 `ConfidenceTrainingConfig`로 변환된다.
  - HF options가 전달된다.
  - task mismatch / ratio 오류는 fast fail.
- dataset report
  - source별 count와 positive rate를 계산한다.
  - duplicate id를 실패 처리한다.
  - missing source/group count를 기록한다.
- QA raw dataset builder
  - TriviaQA-style local JSONL fixture에서 query raw와 query-context raw를 생성한다.
  - query-context row는 positive/negative 균형을 유지한다.
  - negative row metadata에 `negative_retrieved_tfidf`를 기록한다.

### 6.2 Smoke Tests
- fake LM Studio provider로 query-only canonical JSONL 생성.
- fake LM Studio provider로 query+context canonical JSONL 생성.
- small synthetic canonical dataset으로 training CLI가 artifact를 생성하는지 확인.
- 생성된 artifact가 `CBDRStrategy.from_artifacts(...)`에서 task type 검증을 통과하는지 확인.

### 6.3 Live Verification
LM Studio 서버가 떠 있을 때만 수동 실행한다.
CI 기본 경로에는 포함하지 않는다.

```bash
uv run python scripts/generate_confidence_dataset.py \
  --task query_answerability_confidence \
  --provider lmstudio \
  --lmstudio-base-url http://localhost:1234/v1 \
  --lmstudio-model google/gemma-4-12b \
  --input tests/fixtures/confidence_query_answerability_raw.jsonl \
  --output /tmp/ranksmith-lmstudio-query-answerability.jsonl \
  --overwrite \
  --max-items 3
```

검증 기준:
- 응답이 strict JSON이다.
- `answer`가 비어 있지 않다.
- canonical row의 `label`이 0 또는 1이다.
- positive/negative count가 리포트된다.
- LM Studio usage에서 reasoning tokens가 `0`이거나 낮게 유지된다.

### 6.4 최종 검증
```bash
uv run pytest tests/test_lmstudio_provider.py tests/test_generate_confidence_dataset_script.py tests/test_train_confidence_scorer_script.py tests/test_confidence_training_dataset_report.py -q
./scripts/verify.sh
```

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] LM Studio / Gemma 4 12B 운영 조건 확인
- [x] 스펙 문서(본 문서) 사용자 검토 및 확정

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/integrations/_lmstudio_provider.py`: LM Studio OpenAI-compatible `ModelProvider` 구현
- [x] `src/ranksmith/integrations/__init__.py`: `LMStudioModelProvider` submodule export 추가
- [x] `scripts/generate_confidence_dataset.py`: CBDR용 answerability generation CLI 추가
- [x] `scripts/train_confidence_scorer.py`: confidence training CLI 추가
- [x] `src/ranksmith/confidence_training/_dataset_report.py`: dataset balance/generalization helper 추가
- [x] `scripts/report_confidence_dataset.py`: dataset report CLI 추가
- [x] `scripts/build_qa_confidence_raw_dataset.py`: TriviaQA 기반 raw dataset builder 추가
- [x] `scripts/run_lmstudio_confidence_pipeline.py`: raw 생성 → LM Studio generation → training → optional benchmark 통합 runner 추가

### Phase 3: 검증 (Verification)
- [x] `tests/test_lmstudio_provider.py`: provider 정상/실패 케이스 테스트 추가
- [x] `tests/test_generate_confidence_dataset_script.py`: generation CLI 테스트 추가
- [x] `tests/test_train_confidence_scorer_script.py`: training CLI 테스트 추가
- [x] `tests/test_confidence_training_dataset_report.py`: dataset report 테스트 추가
- [x] fake provider 기반 canonical generation smoke 실행
- [x] synthetic artifact 기반 CBDR loading smoke 실행
- [x] LM Studio live smoke는 수동 opt-in으로 실행
- [x] TriviaQA-style local fixture 기반 raw builder smoke 실행
- [x] `./scripts/verify.sh` 스크립트를 통한 린트/타입/전체 테스트 통과 확인

### Phase 4: 완료 및 정리
- [x] `README.md`: LM Studio confidence pipeline 최소 예시 추가
- [x] `README.ko.md`: 동일 구조의 한국어 예시 추가
- [x] `docs/wiki/02_architecture.md`: LM Studio integration과 confidence pipeline CLI 위치 반영
- [x] 본 문서 최상단의 **상태**를 `Completed`로 변경
