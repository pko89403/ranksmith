# Spec: Native MTEB Reranking Evaluation CLI

## 1. 개요 (Overview)
- **작업 목적**: MTEB native Reranking task에서 ranksmith reranking method의 품질, 비용, 지연 시간, token 사용량, invalid output 비율을 함께 측정하는 CLI를 만든다. 결과는 사용자가 method를 고를 때 참고할 실험 자료로 제공하며, 보편적인 leaderboard나 절대적인 승자 판정으로 사용하지 않는다.
- **Reference**
  - MTEB Task API: `mteb.get_tasks(task_types=["Reranking"], languages=["eng"])`
  - MTEB Model API: native Reranking task는 query-candidate set을 재정렬하는 평가 단위다.
  - 기존 ranksmith 계약: `{"ranking": [3, 1, 2]}` 형태의 1-based integer permutation.
- **상태**: `[ ] Draft` | `[ ] In Progress` | `[x] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **작업 경로 (Project Paths)**
  - Project root: `/Users/skiiwoo/Documents/New project 2`
  - Code:
    - `/Users/skiiwoo/Documents/New project 2/scripts/evaluate_mteb_reranking.py`
    - `/Users/skiiwoo/Documents/New project 2/src/ranksmith/_mteb_eval.py`
    - `/Users/skiiwoo/Documents/New project 2/src/ranksmith/_metrics.py`
  - Tests:
    - `/Users/skiiwoo/Documents/New project 2/tests/test_mteb_eval.py`
  - Docs:
    - `/Users/skiiwoo/Documents/New project 2/docs/specs/spec_mteb_reranking_evaluation.md`
    - `/Users/skiiwoo/Documents/New project 2/README.md` 또는 `/Users/skiiwoo/Documents/New project 2/README.ko.md`
- **입력 (Inputs)**
  - MTEB native Reranking task.
  - 평가할 method 목록: `original`, `direct@20`, `rankgpt_sliding_window@20`, `rankgpt_sliding_window@50`, `rankgpt_sliding_window@100`.
  - Azure OpenAI 환경 변수와 `.env` 파일.
  - 실행 제한 옵션: `--allow-live`, `--max-queries`, `--max-document-chars`, `--tasks`, `--output-dir`.
- **출력 (Outputs)**
  - query-method 단위 JSONL 로그.
  - task-level summary JSON.
  - overall summary JSON.
  - 사람이 읽을 수 있는 Markdown reference table.
  - 기본 저장 위치:
    - `/Users/skiiwoo/Documents/New project 2/benchmark-results/mteb-reranking/<experiment_id>/query_results.jsonl`
    - `/Users/skiiwoo/Documents/New project 2/benchmark-results/mteb-reranking/<experiment_id>/task_summary.json`
    - `/Users/skiiwoo/Documents/New project 2/benchmark-results/mteb-reranking/<experiment_id>/overall_summary.json`
    - `/Users/skiiwoo/Documents/New project 2/benchmark-results/mteb-reranking/<experiment_id>/result_tables.md`
    - `/Users/skiiwoo/Documents/New project 2/benchmark-results/mteb-reranking/<experiment_id>/metadata.json`
- **제약 사항 (Constraints)**
  - Retrieval task, `prediction_folder`, `convert_to_reranking()`은 사용하지 않는다.
  - core public API는 변경하지 않는다.
  - MTEB 호환 처리는 CLI 내부 adapter로만 둔다.
  - 비용이 발생하는 LLM 실행은 `--allow-live` 없이는 거부한다.
  - invalid LLM output은 main metric에서 repair하지 않는다.
  - main metric의 invalid query-method는 `zero_score`로 처리한다.
  - repair metric은 diagnostic 결과에만 기록한다.
  - latency와 LLM call 수는 MVP에서 반드시 정량화한다.
  - token/cost는 provider response에서 usage를 얻을 수 있으면 정량화한다.
  - provider usage를 얻을 수 없으면 token/cost 필드는 `null`로 기록하고 `usage_unavailable_reason`을 남긴다.
  - 가격표는 자동 추론하지 않고 CLI 입력 또는 config 파일로만 받는다.
  - unknown MTEB raw schema는 추론하지 않고 fast fail한다.
  - candidate text가 `--max-document-chars`를 초과하면 fast fail한다.
  - 숨은 truncation은 하지 않는다.
  - MVP에서는 `--allow-truncate`를 제공하지 않는다.
  - 공식 method 이름은 `rankgpt_sliding_window@N`으로 통일한다. `sliding@N`은 CLI alias로만 허용한다.
  - `benchmark-results/`는 local artifact로 취급하고 git에 commit하지 않는다.

## 3. 상세 설계 (Architecture & Design)

### 3.1 접근 방식
MVP는 `scripts/evaluate_mteb_reranking.py` 하나의 CLI를 추가한다.

이 CLI는 MTEB native Reranking task를 로드하고, task별 raw sample을 내부 schema로 정규화한 뒤, 동일한 candidate set에 대해 여러 reranking method를 실행한다.

core `ranksmith` API는 그대로 둔다.

```text
MTEB native Reranking task
-> MTEB sample adapter
-> RerankingSample
-> original/direct/sliding method runner
-> strict metric + diagnostic repair metric
-> JSONL / JSON / Markdown report
```

### 3.2 내부 데이터 모델
```python
@dataclass(frozen=True)
class MtebRerankingCandidate:
    doc_id: str
    text: str
    label: float


@dataclass(frozen=True)
class MtebRerankingSample:
    task_name: str
    split: str
    query_id: str
    query: str
    candidates: tuple[MtebRerankingCandidate, ...]
```

```python
@dataclass(frozen=True)
class MethodRunResult:
    method_name: str
    valid: bool
    doc_ids: tuple[str, ...]
    repaired_doc_ids: tuple[str, ...]
    failure_type: str | None
    raw_outputs: tuple[WindowRawOutput, ...]
    llm_calls: int
    latency_ms: float
```

```python
@dataclass(frozen=True)
class WindowRawOutput:
    window_start: int
    window_end: int
    raw_text: str
    parsed_ranking: tuple[int, ...] | None
    valid: bool
    failure_type: str | None
    latency_ms: float
```

### 3.3 MTEB task adapter
MVP는 task schema를 명확히 확인한 task만 지원한다.

기본 task subset:
- `AskUbuntuDupQuestions`
- `SciDocsRR`
- `StackOverflowDupQuestions`

지원하지 않는 raw schema를 만나면 다음 메시지로 실패한다.

```text
Unsupported native MTEB reranking schema for task=<task_name>.
Add an explicit adapter before running this task.
```

Adapter 책임:
1. `task.load_data()` 실행.
2. `test` split을 읽는다.
3. raw row에서 query, candidate text, label을 추출한다.
4. stable `query_id`, `doc_id`가 없으면 task/split/index 기반 id를 만든다.
5. label은 MTEB 값 그대로 보존한다.
6. candidate가 2개 미만이면 실패한다.
7. candidate text가 `max_document_chars`를 초과하면 실패한다.

CLI는 schema 확인용 기능을 제공한다.

```bash
uv run python scripts/evaluate_mteb_reranking.py --list-tasks
uv run python scripts/evaluate_mteb_reranking.py --inspect-task-schema AskUbuntuDupQuestions
```

`--list-tasks` 출력:
- MTEB에서 발견한 English native Reranking task name.
- CLI에서 adapter가 구현된 task 여부.
- 기본 subset 포함 여부.

`--inspect-task-schema` 출력:
- task name.
- split 목록.
- 첫 번째 row의 top-level keys.
- query field 후보.
- candidate field 후보.
- label field 후보.
- adapter support 여부.

MVP task adapter contract는 코드와 문서에 함께 고정한다.

```text
AskUbuntuDupQuestions: explicit adapter required
SciDocsRR: explicit adapter required
StackOverflowDupQuestions: explicit adapter required
```

실제 MTEB version에서 task schema가 다르면 구현자는 adapter를 추론해서 맞추지 않는다. schema inspect 결과를 보고 spec 또는 adapter contract를 갱신한 뒤 진행한다.

### 3.4 Method 정의

#### original
MTEB candidate order를 그대로 사용한다.

```python
def run_original(sample):
    return [candidate.doc_id for candidate in sample.candidates]
```

LLM metric:
- `llm_calls = 0`
- `llm_latency_ms = 0`

#### direct@k
상위 `k`개 candidate만 단일 LLM call로 재정렬하고, 나머지는 original order로 붙인다.

```python
def run_direct(query, candidates, top_k):
    k = min(top_k, len(candidates))
    head = candidates[:k]
    tail = candidates[k:]
    ranking = call_ranksmith_window(query, head)
    reranked_head = apply_integer_permutation(head, ranking)
    return reranked_head + tail
```

기본값:
- `direct@20`

#### rankgpt_sliding_window@rank_end
상위 `rank_end` candidate 안에서 back-to-front overlapping window reranking을 수행한다. `rank_end` 이후 candidate는 original order로 유지한다.

```python
def run_rankgpt_sliding_window(
    query,
    candidates,
    *,
    rank_start=0,
    rank_end=100,
    window_size=20,
    step=10,
):
    ranking = list(candidates)
    rank_end = min(rank_end, len(ranking))
    end = rank_end

    while True:
        start = max(rank_start, end - window_size)
        window = ranking[start:end]
        integer_ranking = call_ranksmith_window(query, window)
        reranked_window = apply_integer_permutation(window, integer_ranking)
        ranking[start:end] = reranked_window

        if start == rank_start:
            break

        end -= step

    return ranking
```

기본값:
- `rank_start = 0`
- `rank_end = 100`
- `window_size = 20`
- `step = 10`
- `direction = back-to-front`

### 3.5 LLM 호출, latency, token, cost 기록
CLI는 public API를 확장하지 않는다.

대신 evaluation-only Azure provider와 evaluation-only parser를 둔다.

이 parser는 core의 1-based integer permutation 계약을 그대로 따르되, query-level logging을 위해 failure type을 세분화해서 반환한다. core `_parse_ranking`의 동작과 어긋나면 안 되므로 parity test를 추가한다.

```python
class InstrumentedAzureOpenAIProvider:
    def __init__(
        self,
        *,
        api_key,
        azure_endpoint,
        azure_deployment,
        api_version,
        timeout,
        price_config=None,
    ):
        self.azure_deployment = azure_deployment
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout,
        )
        self.price_config = price_config
        self.calls = []

    def rank(self, query, documents):
        started = monotonic()
        response = self.client.chat.completions.create(
            model=self.azure_deployment,
            messages=build_reranking_messages(query, documents),
            response_format={"type": "json_object"},
            temperature=0,
        )
        latency_ms = (monotonic() - started) * 1000
        raw_text = response.choices[0].message.content
        usage = response.usage
        self.calls.append(
            {
                "raw_text": raw_text,
                "latency_ms": latency_ms,
                "document_count": len(documents),
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
                "estimated_cost": estimate_cost(usage, self.price_config),
                "usage_unavailable_reason": None if usage else "provider_usage_missing",
            }
        )
        return raw_text
```

이 provider는 evaluation CLI 전용이다. core `AzureOpenAIProvider`는 content만 반환하므로 usage/cost 기록에 쓰지 않는다.

Window 단위 호출은 `AzureOpenAIReranker.rerank()`를 직접 거치지 않고 evaluation-only Azure provider를 호출한다. 이유는 raw response, latency, failure type, token usage, estimated cost를 query-level log에 남겨야 하기 때문이다. 이 경로는 public API가 아니라 evaluation CLI 내부 구현이다.

Cost 추정은 다음 입력이 있을 때만 수행한다.

```bash
--input-token-price-per-1m 2.50 \
--output-token-price-per-1m 10.00
```

계산식:

```python
estimated_cost = (
    input_tokens / 1_000_000 * input_token_price_per_1m
    + output_tokens / 1_000_000 * output_token_price_per_1m
)
```

가격 입력이 없거나 usage가 없으면 `estimated_cost = null`로 기록한다.

Latency 집계:
- `mean_latency_ms_per_query`
- `p50_latency_ms_per_query`
- `p95_latency_ms_per_query`

Cost 집계:
- `mean_input_tokens_per_query`
- `mean_output_tokens_per_query`
- `mean_total_tokens_per_query`
- `mean_estimated_cost_per_query`
- `total_estimated_cost`

`original`은 LLM을 호출하지 않으므로 다음 값으로 기록한다.
- `llm_calls = 0`
- `llm_latency_ms = 0`
- `input_tokens = 0`
- `output_tokens = 0`
- `total_tokens = 0`
- `estimated_cost = 0`
- `usage_unavailable_reason = null`

### 3.6 Validation policy
Window size가 `n`일 때 valid ranking 조건:
- JSON object여야 한다.
- `"ranking"` field가 있어야 한다.
- `ranking`은 `list[int]`여야 한다.
- 길이는 `n`이어야 한다.
- 값 집합은 `{1, 2, ..., n}`이어야 한다.

failure type:
- `json_parse_failure`
- `missing_ranking`
- `non_integer_rank`
- `length_mismatch`
- `duplicate_rank`
- `missing_rank`
- `out_of_range_rank`

Main metric:
- strict validation만 사용한다.
- invalid query-method는 metric을 직접 0으로 override한다.
- repair 결과는 main result에 섞지 않는다.
- multi-window method에서 window 하나라도 invalid이면 해당 query-method 전체를 `strict_valid=false`로 처리한다.
- 이미 성공한 window raw output은 `query_results.jsonl`에 남긴다.
- strict invalid query-method의 main metric은 모두 0으로 처리한다.

Diagnostic repair:
```python
def repair_integer_ranking(predicted_ranking, n):
    valid_set = set(range(1, n + 1))
    seen = set()
    repaired = []

    for rank in predicted_ranking:
        if isinstance(rank, int) and rank in valid_set and rank not in seen:
            repaired.append(rank)
            seen.add(rank)

    for rank in range(1, n + 1):
        if rank not in seen:
            repaired.append(rank)

    return repaired
```

### 3.7 Metric policy
Primary:
- `ndcg@10`

Secondary:
- `mrr@10`
- `map`
- `recall@10`

Relevance 처리:
- `ndcg@10`: graded label 값을 그대로 사용한다.
- `mrr@10`, `map`, `recall@10`: `label > 0`을 relevant로 처리한다.

Aggregation:
- per-query metric을 먼저 계산한다.
- task metric은 query macro average로 계산한다.
- overall metric은 task macro average로 계산한다.
- optional weighted overall은 query 수를 weight로 계산한다.

Invalid output:
- per-query metric을 직접 0으로 override한다.
- evaluator score tie-breaking에 의존하지 않는다.

### 3.8 CLI 초안
```bash
uv run python scripts/evaluate_mteb_reranking.py \
  --tasks AskUbuntuDupQuestions SciDocsRR StackOverflowDupQuestions \
  --methods original direct@20 rankgpt_sliding_window@20 rankgpt_sliding_window@50 rankgpt_sliding_window@100 \
  --split test \
  --output-dir benchmark-results/mteb-reranking/2026-05-18-native-mteb-reranking \
  --max-queries 50 \
  --max-document-chars 4000 \
  --allow-live
```

주요 옵션:
- `--tasks`: 실행할 MTEB task name 목록.
- `--all-english-reranking-tasks`: MTEB의 English native Reranking task 전체 사용.
- `--methods`: 실행할 method 목록.
- `--split`: 기본 `test`.
- `--max-queries`: 비용 제어용 query 수 제한.
- `--max-document-chars`: candidate text 최대 길이. 기본 `4000`.
- `--output-dir`: 결과 저장 directory.
- `--overwrite`: 기존 output directory가 있을 때 삭제 후 재생성.
- `--resume`: 기존 `query_results.jsonl`을 읽어 완료된 query-method를 건너뛴다.
- `--env-file`: 기본 `.env`.
- `--allow-live`: live Azure OpenAI 호출 허용.
- `--strict-failed-query-policy`: MVP는 `zero_score`만 지원.
- `--include-repaired-diagnostics`: repaired diagnostic metric 저장.
- `--input-token-price-per-1m`: cost 추정용 input token 단가. 자동 기본값 없음.
- `--output-token-price-per-1m`: cost 추정용 output token 단가. 자동 기본값 없음.
- `--list-tasks`: MTEB English native Reranking task와 adapter 지원 여부 출력.
- `--inspect-task-schema`: 특정 task의 raw schema를 출력하고 실행하지 않음.

`--overwrite`와 `--resume`은 함께 사용할 수 없다.

Method name 정책:
- 출력/로그/README의 공식 이름은 `rankgpt_sliding_window@N`이다.
- CLI는 편의상 `sliding@N`을 받을 수 있지만 즉시 `rankgpt_sliding_window@N`으로 normalize한다.

CLI는 live 실행 전에 validation policy를 `stderr`에 출력한다.

```text
Strict validation is enabled.
Invalid LLM outputs are not repaired for main metrics.
Invalid query-method results will receive zero scores.
Repaired metrics, if enabled, are diagnostic only.
```

### 3.9 결과 파일
`--output-dir` 아래에 다음 파일을 만든다.

```text
query_results.jsonl
task_summary.json
overall_summary.json
result_tables.md
metadata.json
```

예시 기본 경로:

```text
/Users/skiiwoo/Documents/New project 2/benchmark-results/mteb-reranking/2026-05-18-native-mteb-reranking/
  query_results.jsonl
  task_summary.json
  overall_summary.json
  result_tables.md
  metadata.json
```

`query_results.jsonl` 예시:
```json
{
  "experiment_id": "ranksmith_native_mteb_reranking",
  "task": "AskUbuntuDupQuestions",
  "split": "test",
  "query_id": "AskUbuntuDupQuestions:test:42",
  "method": "rankgpt_sliding_window@100",
  "candidate_doc_ids": ["d1", "d2", "d3"],
  "candidate_labels": {"d1": 0.0, "d2": 1.0, "d3": 0.0},
  "strict_valid": true,
  "strict_failure_type": null,
  "final_doc_ids_strict": ["d2", "d1", "d3"],
  "metrics_strict": {
    "ndcg@10": 1.0,
    "mrr@10": 1.0,
    "map": 1.0,
    "recall@10": 1.0
  },
  "llm_calls": 9,
  "llm_latency_ms": 9700.0,
  "usage": {
    "input_tokens": 12450,
    "output_tokens": 740,
    "total_tokens": 13190,
    "estimated_cost": 0.038525,
    "usage_unavailable_reason": null
  }
}
```

`metadata.json`에는 main result의 validation policy를 반드시 기록한다.

```json
{
  "validation": {
    "main": "strict",
    "invalid_output_policy": "zero_score",
    "repair_enabled_for_main": false,
    "repair_enabled_for_diagnostic": true
  }
}
```

`result_tables.md` 상단에는 같은 정책을 사람이 읽을 수 있게 기록한다.

```markdown
## Validation Policy

Main results use strict validation. Invalid LLM outputs are not repaired.
If a query-method fails validation, its main metrics are set to 0.
Repaired metrics are reported only as diagnostics and are not mixed into the main result.
```

Main/Fairness/Scope table에는 다음 컬럼을 반드시 포함한다.

- `Invalid %`
- `Failed Queries`
- `Evaluated Queries`
- `Mean Latency/q`
- `P50 Latency/q`
- `P95 Latency/q`
- `Mean Tokens/q`
- `Mean Cost/q`
- `Total Cost`

### 3.10 실험 매트릭스
Reference:
- `original`
- `direct@20`
- `rankgpt_sliding_window@20`
- `rankgpt_sliding_window@100`

Fairness:
- `direct@20`
- `rankgpt_sliding_window@20`

Scope ablation:
- `rankgpt_sliding_window@20`
- `rankgpt_sliding_window@50`
- `rankgpt_sliding_window@100`

README에는 이 세 관점을 분리해서 기록한다.

- Method reference: method별 사용 상황, candidate scope, LLM call 수, latency/cost 성향.
- Native MTEB result: task macro average metric과 invalid output 비율.
- Interpretation guide: 결과를 보편적 순위가 아니라 선택 참고 자료로 해석하는 방법.

Optional:
- window/step grid
- candidate order shuffle robustness
- repaired diagnostic metric
- token/cost instrumentation

### 3.11 README reporting policy
README는 benchmark 결과를 "어떤 method가 항상 더 좋다"는 식으로 표현하지 않는다.

README에 넣을 문구:

```text
These results are intended as practical reference points, not a universal ranking.
Results depend on dataset, model, candidate count, latency budget, and invalid output rate.
This benchmark measures reranking over fixed native MTEB candidate sets, not first-stage retrieval.
```

README의 method reference table은 다음 컬럼을 포함한다.

```text
Method | Best for | Candidate Scope | LLM Calls/q | Tokens/q | Cost/q | Latency/q | Quality Signal
```

권장 해석:
- `original`: LLM 비용이 없는 기준선. reranking algorithm 성능을 뜻하지 않는다.
- `direct@20`: 낮은 latency/cost가 중요한 경우의 기본 LLM reranking 후보.
- `rankgpt_sliding_window@20`: `direct@20`과 같은 candidate scope에서 sliding 절차 자체를 비교하는 참고값.
- `rankgpt_sliding_window@100`: 더 넓은 candidate scope를 검사하는 품질/비용 trade-off 후보.

README의 metric table은 다음 컬럼을 포함한다.

```text
Method | NDCG@10 | MRR@10 | MAP | Recall@10 | Invalid % | Failed Queries | Calls/q | Tokens/q | Cost/q | Latency/q
```

README 결과 블록에는 수치와 함께 다음 재현 정보를 반드시 표시한다.

```text
Model | Date | Git Commit | MTEB Version | Tasks | Split | Max Queries | Max Document Chars | Validation Policy
```

주의 문구:

```text
Compare `direct@20` and `rankgpt_sliding_window@20` when you want the same candidate scope.
Compare `direct@20` and `rankgpt_sliding_window@100` only as a quality/cost trade-off.
```

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- **공통 컴포넌트 식별 (Shared Components)**
  - 기존 `Document`와 provider protocol을 재사용한다.
  - `direct`와 `rankgpt_sliding_window`의 window 이동 규칙은 기존 `ListwiseStrategy`와 동등해야 한다.
  - 기존 `_metrics.py`의 `ndcg_at_k`, `mrr_at_k`, `recall_at_k`는 재사용한다.
  - `map` 계산 helper는 `_metrics.py`에 추가한다.
  - 기존 `scripts/compare_reranking.py`와 env loading, live opt-in, report 구조를 맞춘다.
- **추상화 방안 (Abstraction Plan)**
  - `scripts/evaluate_mteb_reranking.py`: CLI, MTEB loading, report writing.
  - `src/ranksmith/_mteb_eval.py`: private schema, adapter, method runner, metric aggregation.
  - `tests/test_mteb_eval.py`: MTEB dependency 없이 synthetic raw rows로 adapter와 metric을 검증한다.
  - MTEB는 core dependency가 아니라 dev/optional dependency로 둔다.

## 5. 에러 핸들링 (Error Handling)
- `mteb` 미설치: CLI 시작 시 설치 안내와 함께 실패한다.
- `--allow-live` 없음: LLM method가 포함된 경우 실행을 거부한다.
- token price 옵션이 하나만 제공됨: input/output 가격을 둘 다 넣으라고 실패한다.
- token price가 음수임: 실패한다.
- `--overwrite`와 `--resume`이 함께 제공됨: 실패한다.
- unknown task name: MTEB error를 task name과 함께 보여준다.
- unsupported raw schema: task name, split, row index를 포함해 실패한다.
- `--all-english-reranking-tasks`에 unsupported task가 포함됨: unsupported task 목록을 출력하고 실패한다.
- candidate 2개 미만: 해당 row를 숨기지 않고 실패한다.
- candidate text가 비어 있음: task name, query id, doc id를 포함해 실패한다.
- candidate text가 `max_document_chars`를 초과함: task name, query id, doc id, 길이를 포함해 실패한다.
- invalid method string: 허용 method 목록을 출력하고 실패한다.
- `step > window_size`: `RerankInputError`와 같은 의미의 CLI validation error로 실패한다.
- multi-window method에서 한 window의 LLM output이 invalid: query-method를 failed로 기록하고 main metric은 0으로 처리한다.
- output directory가 이미 있고 `--overwrite` 또는 `--resume`이 없으면 실패한다.

## 6. 테스트 계획 (Test Plan)
- **성공 케이스 (Happy Paths)**
  - synthetic MTEB raw row를 `MtebRerankingSample`로 변환한다.
  - `--list-tasks`와 `--inspect-task-schema`는 live LLM 호출 없이 동작한다.
  - `original`은 candidate order를 유지한다.
  - `direct@20`은 head만 재정렬하고 tail은 유지한다.
  - `rankgpt_sliding_window@20`은 candidate가 20개 이하일 때 LLM call 1회만 수행한다.
  - `rankgpt_sliding_window@50`과 `@100`은 예상 window 수를 기록한다.
  - graded label의 `ndcg@10`과 binary relevance metric을 분리 계산한다.
  - provider usage가 있으면 token/cost를 계산한다.
  - provider usage가 없으면 token/cost를 `null`로 기록하고 reason을 남긴다.
  - query-level JSONL과 summary JSON schema를 검증한다.
  - `sliding@20` alias가 `rankgpt_sliding_window@20`으로 normalize된다.
  - `--resume`은 완료된 query-method를 건너뛴다.
- **엣지/실패 케이스 (Edge & Failure Cases)**
  - invalid ranking은 main metric 0으로 처리된다.
  - multi-window 중간 실패는 query-method 전체 실패로 처리된다.
  - repaired metric은 diagnostic에만 기록된다.
  - unsupported raw schema는 fast fail한다.
  - `max_document_chars` 초과는 truncation 없이 fast fail한다.
  - `--allow-live` 없이 LLM method를 실행하면 실패한다.
  - `--overwrite`와 `--resume` 동시 사용은 실패한다.
  - `step > window_size`, `rank_end < 1`, `max_queries < 1`은 실패한다.
- **공통 Reranking Smoke/Benchmark**
  - 이 CLI는 MTEB native task용이므로 기존 fixture JSONL을 입력으로 받지 않는다.
  - 단위 테스트는 MTEB 네트워크 다운로드 없이 synthetic row로 수행한다.
  - live Azure + real MTEB 실행은 수동 opt-in 검증으로 둔다.
  - README에 기록되는 결과는 method 선택 참고 자료로 표현하고, 보편적 leaderboard로 표현하지 않는다.

---

## 7. 작업 태스크 추적 (Task Checklist)

### Phase 1: 컨텍스트 및 설계 확인
- [x] 관련 기존 코드베이스 및 Wiki 문서 확인
- [x] MTEB native Reranking만 사용하는 범위 확정
- [x] 사용자와 spec 검토 및 승인

### Phase 2: 로직 구현 (Implementation)
- [x] `src/ranksmith/_metrics.py`: `map` 계산 helper 추가
- [x] `src/ranksmith/_mteb_eval.py`: private schema, adapter, method runner, metric aggregation 구현
- [x] `scripts/evaluate_mteb_reranking.py`: CLI, env loading, live opt-in, output writing 구현
- [x] `scripts/evaluate_mteb_reranking.py`: token price 옵션과 estimated cost 계산 추가
- [x] `scripts/evaluate_mteb_reranking.py`: `--list-tasks`, `--inspect-task-schema`, `--overwrite`, `--resume` 구현
- [x] `pyproject.toml`: MTEB dependency를 dev 또는 optional group으로 추가
- [x] `.gitignore`: `/benchmark-results/` 추가

### Phase 3: 검증 (Verification)
- [x] `tests/test_mteb_eval.py`: adapter 정상 케이스 테스트 추가
- [x] `tests/test_mteb_eval.py`: direct/sliding runner와 failure policy 테스트 추가
- [x] `tests/test_mteb_eval.py`: metric aggregation과 JSON output 테스트 추가
- [x] `tests/test_mteb_eval.py`: latency/token/cost summary 테스트 추가
- [x] `tests/test_mteb_eval.py`: evaluation parser와 core ranking validation 계약의 parity 테스트 추가
- [x] `tests/test_mteb_eval.py`: method alias normalization, resume, max document length 테스트 추가
- [ ] `./scripts/verify.sh` 스크립트를 통한 린트/타입/전체 테스트 통과 확인
- [ ] 수동 opt-in 명령으로 작은 `--max-queries` live smoke 결과 기록

### Phase 4: 완료 및 정리
- [x] `README.md` 또는 `README.ko.md`: MTEB evaluation CLI 사용법과 method 선택 참고 표 추가
- [x] `result_tables.md`에 strict validation과 `zero_score` 정책이 표시되는지 확인
- [x] 결과 예시와 "reference point, not universal ranking" caveat 문서화
- [ ] 본 문서 최상단의 **상태**를 `Completed`로 변경
