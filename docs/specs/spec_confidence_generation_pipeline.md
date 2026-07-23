# Spec: Confidence Generation Pipeline

## 1. 개요 (Overview)
- **작업 목적**: closed model을 호출해 confidence training에 필요한 canonical JSONL을 생성한다.
- **Reference**:
  - `docs/wiki/references/structural_confidence.md`
  - `docs/specs/spec_structural_confidence.md`
  - `docs/specs/spec_confidence_training_pipeline.md`
  - `docs/specs/spec_confidence_runtime_readiness.md`
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`

기존 `ranksmith.confidence_training`은 이미 생성된 canonical JSONL만 학습한다.
이번 기능은 그 앞 단계인 **closed model generation**을 별도 utility layer로 추가한다.

지원하는 generation 갈래는 두 개다.

1. **Answer-oriented generation**
   - `query + context + gold_answer`
   - closed model answer 생성
   - normalized exact match로 `answer_confidence` canonical JSONL 생성

2. **Relevance-oriented generation**
   - `query + document + relevance_label`
   - closed model binary relevance judgment 생성
   - qrel-derived truth와 비교해 `judgment_confidence` canonical JSONL 생성

## 2. 요구 사항 및 제약 (Requirements & Constraints)

### 범위
포함한다:
- `ranksmith.confidence_generation` submodule 추가.
- raw JSONL 입력 validation/loading.
- closed model 호출 wrapper.
- model JSON output parsing.
- answer-oriented label 생성.
- relevance-oriented label 생성.
- confidence training canonical JSONL export.
- checkpoint/resume 지원.
- provider가 반환한 `RerankUsage` callback 전달.
- sync API만 지원.

포함하지 않는다:
- async generation.
- CLI.
- qrel/BEIR/MTEB adapter.
- SQuAD/QA dataset adapter.
- semantic answer evaluator.
- multi-sample/self-consistency.
- runtime reranking Strategy/Algorithm 연결.
- scorer 학습 실행.
- benchmark 수치 README 반영.
- root package export.

### Public API

`ranksmith.confidence_generation` submodule에서만 export한다.

```python
from ranksmith.confidence_generation import (
    AnswerGenerationConfig,
    RelevanceGenerationConfig,
    ConfidenceGenerationResult,
    ConfidenceGenerationError,
    ConfidenceGenerationInputError,
    ConfidenceGenerationParseError,
    generate_answer_confidence_dataset,
    generate_judgment_confidence_dataset,
)
```

Root import에는 추가하지 않는다.

### 입력: Answer-oriented raw JSONL

각 line은 하나의 answer generation sample이다.

필수 필드:
- `id: str`
- `query: str`
- `context: str`
- `gold_answer: str | list[str]`

선택 필드:
- `source: str`
- `group_id: str`
- `metadata: dict[str, object]`

그 외 field는 허용하지 않는다.
unexpected field는 `ConfidenceGenerationInputError`로 실패한다.
`context` 길이가 `max_context_chars`를 넘으면 실패한다.

예시:

```json
{"id":"a1","query":"Who played Karen?","context":"...","gold_answer":["Nancy Travis"]}
```

### 입력: Relevance-oriented raw JSONL

각 line은 하나의 relevance judgment generation sample이다.

필수 필드:
- `id: str`
- `query: str`
- `document: str`
- `relevance_label: int | float | bool`

선택 필드:
- `source: str`
- `group_id: str`
- `metadata: dict[str, object]`

그 외 field는 허용하지 않는다.
unexpected field는 `ConfidenceGenerationInputError`로 실패한다.
`document` 길이가 `max_document_chars`를 넘으면 실패한다.

예시:

```json
{"id":"j1","query":"Who played Karen?","document":"...","relevance_label":1}
```

### Closed model output

#### Answer-oriented

허용 JSON shape:

```json
{"answer":"Nancy Travis"}
```

규칙:
- `answer`는 non-empty string이어야 한다.
- 빈 응답, 잘못된 JSON, 누락 필드, 추가 타입은 실패한다.
- `answer` 외 unexpected output field는 실패한다.
- answer text를 조용히 보정하지 않는다.

#### Relevance-oriented

허용 JSON shape:

```json
{"judgment":"relevant"}
```

허용 값:
- `"relevant"`
- `"not_relevant"`

규칙:
- 위 두 값 외에는 실패한다.
- `judgment` 외 unexpected output field는 실패한다.
- rationale, score, confidence 같은 추가 의미 필드는 이번 범위에서 사용하지 않는다.

### 출력: `answer_confidence` canonical JSONL

기존 `confidence_training` schema와 호환되어야 한다.

```json
{
  "id": "a1",
  "context": "...",
  "answer": "Nancy Travis",
  "gold_answer": ["Nancy Travis"],
  "label": 1,
  "source": "optional",
  "group_id": "optional",
  "metadata": {
    "input_metadata": {},
    "generation": {
      "generation_task": "answer_oriented",
      "query": "Who played Karen?",
      "match_policy": "normalized_exact",
      "no_answer_value": "__NO_ANSWER__",
      "raw_model_output": "{\"answer\":\"Nancy Travis\"}"
    }
  }
}
```

`label` 의미:
- `1`: generated answer가 `gold_answer`와 normalized exact match.
- `0`: generated answer가 어떤 `gold_answer`와도 match하지 않음.

### 출력: `judgment_confidence` canonical JSONL

기존 `confidence_training` schema와 호환되어야 한다.

```json
{
  "id": "j1",
  "query": "Who played Karen?",
  "document": "...",
  "judgment": "relevant",
  "relevance_label": 1,
  "label": 1,
  "source": "optional",
  "group_id": "optional",
  "metadata": {
    "input_metadata": {},
    "generation": {
      "generation_task": "relevance_oriented",
      "parsed_judgment": "relevant",
      "truth_judgment": "relevant",
      "truth_positive_threshold": 0.0,
      "truth_positive_operator": "gt",
      "raw_model_output": "{\"judgment\":\"relevant\"}"
    }
  }
}
```

`label` 의미:
- `1`: closed model judgment가 qrel-derived truth와 일치.
- `0`: closed model judgment가 qrel-derived truth와 불일치.

### Label 정책

#### Answer match

이번 범위는 `normalized_exact`만 지원한다.

Normalization:
- strip leading/trailing whitespace
- lowercase
- collapse internal whitespace

No-answer policy:
- sentinel은 `ranksmith.model.NO_ANSWER_VALUE`(`"__NO_ANSWER__"`) 고정 상수다.
  (초안의 config 필드 `no_answer_value`는 answer prompt를 리랭커와 공유 상수로
  통일하면서 제거됐다 — 학습/추론 프롬프트 일관성 요구사항.)
- closed model answer가 sentinel과 정확히 같으면 항상 `label=0`이다.
- `gold_answer`가 우연히 sentinel과 같아도 match로 인정하지 않는다.

제외:
- punctuation removal
- alias table
- semantic equivalence
- LLM-as-judge answer evaluation

#### Relevance truth conversion

기본값:

```python
truth_positive_threshold = 0.0
truth_positive_operator = "gt"  # "gt" | "gte"
```

규칙:
- `gt`: `relevance_label > threshold`이면 relevant.
- `gte`: `relevance_label >= threshold`이면 relevant.
- `bool`은 threshold와 무관하게 `True -> relevant`, `False -> not_relevant`.
- `relevance_label`이 numeric/bool이 아니면 실패한다.

### Output file 정책

- `overwrite=True`, `resume=True`는 동시에 사용할 수 없다.
- output file이 이미 있고 `overwrite=False`, `resume=False`이면 실패한다.
- `overwrite=True`이면 기존 output file을 새로 쓴다.
- `resume=True`이면 기존 output file을 읽어 completed id를 수집한 뒤 append한다.
- `resume=True`인데 output file이 없으면 새 파일을 만든다.
- parent directory가 없으면 생성한다.

### Metadata / source 병합 정책

Raw input `metadata`는 output `metadata.input_metadata` 아래에 nested로 보존한다.
Generation metadata는 output `metadata.generation` 아래에 nested로 기록한다.

예시:

```json
{
  "metadata": {
    "input_metadata": {"dataset": "sample"},
    "generation": {
      "generation_task": "relevance_oriented",
      "parsed_judgment": "relevant",
      "truth_judgment": "relevant"
    }
  }
}
```

규칙:
- raw input metadata는 JSON object여야 한다.
- raw input metadata key는 string이어야 한다.
- raw input metadata value는 JSON-serializable이어야 한다.
- raw input metadata에 `input_metadata`나 `generation` key가 있어도 그대로 `input_metadata` 아래에 들어가므로 충돌하지 않는다.
- `include_raw_model_output=False`이면 `metadata.generation.raw_model_output` key를 쓰지 않는다.
- config `source`는 row `source`가 없을 때만 사용한다.
- row `source`와 config `source`가 모두 있고 값이 달라도 row `source`를 우선한다.
- `group_id`는 row 값만 사용한다. config-level `group_id`는 제공하지 않는다.

### Config

#### `AnswerGenerationConfig`

필드:
- `input_path: str | Path`
- `output_path: str | Path`
- `provider: ModelProvider`
- `overwrite: bool = False`
- `resume: bool = False`
- `max_items: int | None = None`
- `max_context_chars: int = 4000`
- `include_raw_model_output: bool = True`
- `on_usage: Callable[[RerankUsage], None] | None = None`
- `source: str | None = None`

(sentinel은 config가 아니라 `ranksmith.model.NO_ANSWER_VALUE` 공유 상수를 쓴다.
초안에 있던 `no_answer_value` 필드는 제거됐다.)

#### `RelevanceGenerationConfig`

필드:
- `input_path: str | Path`
- `output_path: str | Path`
- `provider: ModelProvider`
- `truth_positive_threshold: float = 0.0`
- `truth_positive_operator: Literal["gt", "gte"] = "gt"`
- `overwrite: bool = False`
- `resume: bool = False`
- `max_items: int | None = None`
- `max_document_chars: int = 4000`
- `include_raw_model_output: bool = True`
- `on_usage: Callable[[RerankUsage], None] | None = None`
- `source: str | None = None`

### Result

`ConfidenceGenerationResult`

필드:
- `output_path: Path`
- `input_count: int`
- `generated_count: int`
- `skipped_count: int`
- `positive_count: int`
- `negative_count: int`

`ConfidenceGenerationResult`는 usage를 합산하지 않는다.
provider가 `ModelResponse.usage`를 반환하면 `on_usage` callback으로 각 call의 usage를 그대로 전달한다.

Count 기준:
- `input_count`: input JSONL에서 validation을 통과한 전체 sample 수.
- `generated_count`: 이번 실행에서 새로 생성해 output에 append/write한 row 수.
- `skipped_count`: `resume=True` 때문에 기존 output id와 중복되어 건너뛴 row 수.
- `positive_count`: 이번 실행에서 새로 생성한 row 중 `label=1` 개수.
- `negative_count`: 이번 실행에서 새로 생성한 row 중 `label=0` 개수.

`max_items`는 **이번 실행에서 생성할 신규 row 수**를 제한한다.
예를 들어 기존 output에 10개가 있고 `max_items=5`, `resume=True`이면 신규 row를 최대 5개 생성한다.

## 3. 상세 설계 (Architecture & Design)

### 모듈 구조

```text
src/ranksmith/confidence_generation/
  __init__.py
  _types.py       # config/result/raw sample dataclass
  _errors.py      # generation-specific errors
  _io.py          # JSONL read/write, resume id loading
  _prompts.py     # answer/relevance prompt builders
  _parsing.py     # strict JSON parser
  _labeling.py    # answer match, relevance truth conversion
  _pipeline.py    # public generation functions
```

이 모듈은 `confidence_training`과 분리한다.
`confidence_training`은 학습만 담당하고, closed model 호출 책임을 갖지 않는다.

### Data Flow

```text
raw JSONL
  -> validate raw sample
  -> skip completed id if resume=True
  -> build closed model prompt
  -> provider.complete(ModelRequest)
  -> parse strict JSON
  -> derive binary label
  -> append canonical JSONL row
  -> return generation summary
```

### Prompt 계약

#### Answer prompt

System:

```text
You answer questions using only the provided context. Return only JSON with an "answer" string.
```

User:

```text
Question:
{query}

Context:
{context}

Return JSON exactly like this shape:
{"answer":"..."}

Use only the context. If the context does not contain the answer, return {"answer":"__NO_ANSWER__"}.
```

#### Relevance prompt

System:

```text
You judge document relevance. Return only JSON with a "judgment" value of "relevant" or "not_relevant".
```

User:

```text
Query:
{query}

Document:
{document}

Return JSON exactly like this shape:
{"judgment":"relevant"}

Use "relevant" if the document contains information useful for answering the query.
Use "not_relevant" otherwise.
```

### 의사 알고리즘

#### Answer-oriented

```text
for sample in input:
  if resume and sample.id in output_ids:
    skipped += 1
    continue

  response = call_model(answer_prompt(sample))
  answer = parse_answer(response.content)
  label = 1 if normalized_exact(answer, sample.gold_answer) else 0
  row = answer_confidence_canonical(sample, answer, label, response.content)
  append(row)
```

#### Relevance-oriented

```text
for sample in input:
  if resume and sample.id in output_ids:
    skipped += 1
    continue

  response = call_model(relevance_prompt(sample))
  judgment = parse_judgment(response.content)
  truth = truth_from_relevance_label(sample.relevance_label, threshold, operator)
  label = 1 if judgment == truth else 0
  row = judgment_confidence_canonical(sample, judgment, truth, label, response.content)
  append(row)
```

### 의사 코드

```python
def generate_judgment_confidence_dataset(config):
    samples = load_relevance_generation_samples(config.input_path)
    completed_ids = load_completed_ids(config.output_path) if config.resume else set()
    writer = open_jsonl_writer(config.output_path, overwrite=config.overwrite, append=config.resume)
    generated_count = 0

    for sample in samples:
        if sample.id in completed_ids:
            skipped += 1
            continue
        if config.max_items is not None and generated_count >= config.max_items:
            break

        response = config.provider.complete(
            ModelRequest(
                messages=[
                    ModelMessage(role="system", content=RELEVANCE_SYSTEM),
                    ModelMessage(role="user", content=build_relevance_prompt(sample)),
                ],
                response_format="json_object",
                temperature=0,
            )
        )
        judgment = parse_relevance_judgment(response.content)
        truth = relevance_truth(
            sample.relevance_label,
            config.truth_positive_threshold,
            config.truth_positive_operator,
        )
        label = int(judgment == truth)
        writer.write(canonical_judgment_row(sample, judgment, truth, label, response.content))
        generated_count += 1
```

### 통합 지점

추가:
- `src/ranksmith/confidence_generation/__init__.py`
- `src/ranksmith/confidence_generation/_types.py`
- `src/ranksmith/confidence_generation/_errors.py`
- `src/ranksmith/confidence_generation/_io.py`
- `src/ranksmith/confidence_generation/_prompts.py`
- `src/ranksmith/confidence_generation/_parsing.py`
- `src/ranksmith/confidence_generation/_labeling.py`
- `src/ranksmith/confidence_generation/_pipeline.py`
- `tests/test_confidence_generation_*.py`

수정:
- `docs/wiki/02_architecture.md`: confidence generation layer 추가.
- `README.md` / `README.ko.md`: 구현된 뒤 짧은 submodule 소개만 추가한다. 자세한 사용 예시는 README에 길게 넣지 않는다.

수정하지 않음:
- `src/ranksmith/model.py`: `ModelClient.judge()`를 이번 범위에 추가하지 않는다.
- `src/ranksmith/__init__.py`: root export를 추가하지 않는다.
- `src/ranksmith/confidence_training/*`: training 책임을 유지한다.
- `src/ranksmith/strategies/*`: reranking Strategy를 추가하지 않는다.

## 4. 재사용 및 모듈화 (Reusability & Modularization)

### Shared Components

공유 대상:
- JSONL reader/writer.
- provider call wrapper.
- strict JSON object parser.
- usage emission.
- canonical metadata builder.
- resume id loader.

task별 분리:
- raw input schema.
- prompt builder.
- output parser.
- label derivation.
- canonical row builder.

### Abstraction Plan

공통 helper:
- `_read_jsonl_objects(path)`
- `_validate_non_empty_text(value, field_name)`
- `_call_provider(provider, system, user)`
- `_emit_usage(usage, callback)`
- `_parse_json_object(content)`
- `_open_output_writer(path, overwrite, resume)`
- `_load_completed_ids(path)`

task별 helper:
- `parse_answer_output(content)`
- `parse_relevance_output(content)`
- `normalized_exact_match(answer, gold_answer)`
- `relevance_truth(value, threshold, operator)`

## 5. 에러 핸들링 (Error Handling)

Exception hierarchy:

```python
class ConfidenceGenerationError(Exception): ...
class ConfidenceGenerationInputError(ConfidenceGenerationError): ...
class ConfidenceGenerationParseError(ConfidenceGenerationError): ...
```

Provider 호출 실패:
- 기존 `RerankProviderError`는 그대로 전파한다.
- 예상 밖 예외는 `RerankProviderError`로 감싼다.
- provider 계층 실패는 `ConfidenceGenerationError`로 감싸지 않는다.
- input/parse/output 정책 실패만 `ConfidenceGeneration*Error`를 사용한다.

입력 실패:
- 파일 읽기 실패.
- JSONL line이 JSON object가 아님.
- 필수 필드 누락.
- unexpected field.
- string field가 비어 있음.
- duplicate id.
- `relevance_label`이 numeric/bool이 아님.
- `truth_positive_operator`가 `"gt"` 또는 `"gte"`가 아님.
- `max_items < 1`.
- `max_context_chars < 1`.
- `max_document_chars < 1`.
- metadata가 JSON object가 아님.
- metadata key가 string이 아님.
- metadata value가 JSON-serializable이 아님.
- `overwrite=True`, `resume=True`가 동시에 지정됨.
- output file이 있는데 `overwrite=False`, `resume=False`.

출력 parsing 실패:
- provider content가 빈 문자열.
- valid JSON이 아님.
- JSON object가 아님.
- 필수 output field 누락.
- answer가 non-empty string이 아님.
- judgment가 `"relevant"` / `"not_relevant"`가 아님.
- model output에 unexpected field가 있음.

Resume 실패:
- 기존 output JSONL이 canonical row가 아님.
- 기존 output에 duplicate id가 있음.
- 기존 output task가 현재 generation task와 맞지 않음.

원칙:
- 조용히 row를 건너뛰지 않는다.
- parse 실패를 negative label로 바꾸지 않는다.
- output 값을 자동 보정하지 않는다.
- 실패 전 이미 append된 row는 유지할 수 있다. 재실행은 `resume=True`로 이어간다.

## 6. 테스트 계획 (Test Plan)

### 성공 케이스

- answer raw JSONL을 읽고 `answer_confidence` canonical JSONL을 생성한다.
- answer normalized exact match가 `label=1`을 만든다.
- answer mismatch가 `label=0`을 만든다.
- answer가 `NO_ANSWER_VALUE` sentinel과 같으면 gold answer와 무관하게 `label=0`을 만든다.
- gold answer list 중 하나와 match하면 `label=1`이다.
- relevance raw JSONL을 읽고 `judgment_confidence` canonical JSONL을 생성한다.
- `relevance_label > 0` 기본 truth가 relevant를 만든다.
- `truth_positive_operator="gte"`가 threshold 포함 비교를 수행한다.
- bool relevance label이 `True/False` truth로 변환된다.
- `resume=True`가 기존 output id를 건너뛰고 나머지만 생성한다.
- `max_items`가 resume 후 신규 생성 row 수를 제한한다.
- provider usage가 `on_usage` callback으로 전달된다.
- raw metadata가 `metadata.input_metadata` 아래에 보존된다.
- generation metadata가 `metadata.generation` 아래에 기록된다.
- `include_raw_model_output=False`이면 raw model output을 metadata에 쓰지 않는다.
- row `source`가 있으면 config `source`보다 우선한다.
- row `source`가 없으면 config `source`를 사용한다.

### 실패 케이스

- input file missing.
- invalid JSONL.
- missing required field.
- unexpected field.
- empty query/context/document.
- duplicate input id.
- output exists with no overwrite/resume.
- `overwrite=True`와 `resume=True` 동시 지정.
- invalid `truth_positive_operator`.
- invalid `max_items`.
- invalid `max_context_chars`.
- invalid `max_document_chars`.
- invalid metadata.
- provider empty response.
- invalid provider JSON.
- answer field missing.
- answer field empty.
- judgment field missing.
- unsupported judgment value.
- model output unexpected field.
- resume output duplicate id.
- resume output task mismatch.

### 검증 명령

개발 완료 전 실행:

```bash
uv run pytest tests/test_confidence_generation_*.py -q
uv run ruff check src/ranksmith/confidence_generation tests/test_confidence_generation_*.py
uv run mypy src/ranksmith/confidence_generation tests/test_confidence_generation_*.py
./scripts/verify.sh
```

환경에 따라 PyPI TLS 문제가 재현되면 다음도 함께 기록한다.

```bash
UV_NATIVE_TLS=true ./scripts/verify.sh
```

### Reranking Smoke/Benchmark

이번 기능은 reranking Strategy/Algorithm을 추가하지 않는다.
따라서 `tests/fixtures/reranking_smoke_fixture.jsonl`와 live benchmark는 변경하지 않는다.

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 기존 confidence runtime/training 문서 확인
- [x] answer-oriented / relevance-oriented generation 범위 확정
- [x] raw JSONL core + external adapter 후속 분리 확정
- [x] binary relevance judgment output 확정
- [x] relevance truth threshold configurable 정책 확정
- [x] 사용자 spec review 및 최종 승인

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/confidence_generation/_errors.py`: generation error hierarchy 구현
- [x] `src/ranksmith/confidence_generation/_types.py`: config/result/raw sample dataclass 구현
- [x] `src/ranksmith/confidence_generation/_io.py`: JSONL load/write/resume helper 구현
- [x] `src/ranksmith/confidence_generation/_prompts.py`: answer/relevance prompt builder 구현
- [x] `src/ranksmith/confidence_generation/_parsing.py`: strict output parser 구현
- [x] `src/ranksmith/confidence_generation/_labeling.py`: answer match/relevance truth helper 구현
- [x] `src/ranksmith/confidence_generation/_pipeline.py`: public generation functions 구현
- [x] `src/ranksmith/confidence_generation/__init__.py`: submodule public export 구현

### Phase 3: 검증 (Verification)
- [x] `tests/test_confidence_generation_io.py`: raw/canonical JSONL IO 테스트 추가
- [x] `tests/test_confidence_generation_parsing.py`: model output parser 테스트 추가
- [x] `tests/test_confidence_generation_labeling.py`: answer/relevance label 테스트 추가
- [x] `tests/test_confidence_generation_pipeline.py`: provider fake 기반 end-to-end 테스트 추가
- [x] `tests/test_confidence_generation_api.py`: submodule export/root non-export 테스트 추가
- [x] `./scripts/verify.sh` 통과 확인

### Phase 4: 완료 및 정리
- [x] `docs/wiki/02_architecture.md`: confidence generation layer 추가
- [x] `README.md` / `README.ko.md`: 구현된 public submodule 설명 동기화
- [x] 본 문서 상태를 `Completed`로 변경
