# Spec: Confidence-Aware Reranking (초안)

## 1. 개요 (Overview)
- **작업 목적**: 구현 완료된 confidence 신호(`ranksmith.confidence`)를 실제 reranking 결정에 연결한다. runtime readiness 스펙이 "CBDR-ready confidence signal"까지 완성해 두었으므로, 이 스펙은 그 신호를 소비하는 첫 모듈을 정의한다.
- **Reference**:
  - `docs/wiki/references/structural_confidence.md` (Trust in One Round — 구현된 confidence 신호의 근간)
  - `docs/wiki/references/cbdr_parametric_confidence_rag.md` (CBDR — 개념 로드맵. white-box 전제라 방식 직접 이식 불가)
  - `docs/wiki/references/llm_confidence_reranker.md` (LCR — training-free 정렬 규칙의 후보)
  - `docs/specs/spec_confidence_runtime_readiness.md` (소비할 signal contract)
- **상태**: `[ ] Draft` | `[x] In Progress` | `[ ] Completed`

Q004-1(경로 A), Q004-2(새 Strategy)가 확정됐고, 순위 규칙은 논문 충실도를 위해 **confidence 변화(answer_confidence) 기반**으로 최종 결정됐다(초기 signed-judgment 방식을 교체). `AnswerConfidenceRerankStrategy` / `AsyncAnswerConfidenceRerankStrategy`를 구현했다. 남은 것은 쓸 만한 scorer artifact(현재는 스모크용)와 벤치마크.

## 확정 설계 (구현됨) — confidence 변화 기반
CBDR 논문의 핵심은 confidence 변화다: `Inc(Q, D) = Conf(질문+문서) − Conf(질문)`. 문서가 답변 확신을 얼마나 높이는가로 순위를 매긴다. 우리는 이를 black-box(구조적 confidence)로 구현한다.

- **형태**: 새 Strategy. `AzureOpenAIReranker(strategy=AnswerConfidenceRerankStrategy(estimator=...))`.
- **answer 획득**: 문서당 LLM `answer(query, document)` 1회 — 그 문서로 질문에 답하게 함. confidence 측정에는 LLM을 쓰지 않는다(로컬 encoder+scorer). LCR처럼 문서당 반복 샘플링하지 않는다.
- **confidence**: `estimator.score(AnswerConfidenceInput(context=document, answer=answer)).score` — "이 답변이 맞을 확률"(structural confidence, 로컬).
- **순위 규칙**: `C_i = Conf(문서_i + 답변_i)` 내림차순. **기준선 생략의 근거**: CBDR은 `Inc = C_i − C_0`로 정렬하는데, 한 질문 안에서 `C_0`(질문만)는 모든 문서에 동일한 상수라 **순위에 영향이 없다**. 게다가 `answer_confidence`는 빈 context를 스코어링할 수 없어(`_require_non_empty`) `C_0` 자체를 못 구한다. 따라서 `C_i`로 정렬 = 질문 내 confidence 변화 정렬.
- **비용**: 문서 N개 → answer N회(리랭킹 신호) + 로컬 confidence N회(LLM 0회). 기준선 호출 없음.
- **async**: answer 호출은 `asyncio.gather`로 동시 실행, scoring은 순차(CPU/encoder).

### 통합 지점 (구현 완료)
- `src/ranksmith/model.py`: `ModelClient.answer` / `AsyncModelClient.answer` 추가(generation 파이프라인의 answer 프롬프트와 일치 — 학습/추론 일관성).
- `src/ranksmith/parsing.py`: `parse_answer_response` 추가.
- `src/ranksmith/strategies/confidence.py`: `AnswerConfidenceRerankStrategy` / `AsyncAnswerConfidenceRerankStrategy`.
- estimator는 Protocol로 타입해 strategies가 torch를 강제 import하지 않는다.

### 검증 (2026-07-05)
- 단위 테스트 8개(synthetic answer client + fake estimator): confidence 내림차순 정렬, 동점 보존, top_k, answer 부재/비-answer estimator/invalid JSON fast fail, sync/async.
- 엔드투엔드 + 벤치마크(LM Studio qwen3.5-9b answer + SQuAD로 학습한 answer_confidence artifact, macOS env 2개). held-out 15개 질문(각: 정답 문맥 1 + 무관 distractor 3), 정답 문맥 복원율:

  | 학습 데이터 | test roc_auc | rerank accuracy@1 | MRR |
  | --- | --- | --- | --- |
  | 100 샘플 | 0.333 | 0.000 | 0.294 |
  | 500 샘플 | 0.875 | **0.800** | **0.878** |
  | 랜덤 기준 | 0.5 | 0.250 | 0.521 |

  **데이터 규모가 관건이다.** 100샘플에선 과적합으로 랜덤 이하(정답 문맥이 오히려 바닥). 500샘플에선 강한 판별력(정답 문맥 12/15가 1위). 접근 자체는 유효하다.

- **Baseline 비교 (같은 15개 세팅) — 정직한 결과: 기존 Listwise한테 진다.**

  | 전략 | accuracy@1 | MRR | LLM 호출/쿼리 |
  | --- | --- | --- | --- |
  | ConfidenceRerank (500) | 0.800 | 0.878 | 4.0 |
  | `ListwiseStrategy` | **1.000** | **1.000** | **1.0** |

  이 세팅에서 Listwise가 완벽하면서 호출은 1/4. confidence 리랭커는 더 나쁘고 4배 비싸다. **현재까지 이 방법이 기존 전략보다 나은 세팅을 하나도 입증하지 못했다.** 이길 가능성이 있는 곳은 listwise window를 초과하는 대규모 후보 등 — 미측정.

- **측정 조건(정직한 한계)**: distractor가 무관 랜덤 SQuAD 문맥이라 모델이 NO_ANSWER를 내는 쉬운 세팅(그래서 Listwise가 자명하게 만점). BM25 hard negative(주제 유사) 같은 현실 세팅은 더 어렵다. 평가셋 15개로 신뢰구간이 넓다. README 성능 claim은 하지 않는다.

### 표준 벤치마크 편입 (구현됨) + 특이사항
`scripts/compare_reranking.py`에 `answer_confidence` 알고리즘을 추가했다(opt-in).
`--algorithm answer_confidence --answer-confidence-artifact <path> --allow-live`로
다른 전략과 같은 qrels 기반 지표(NDCG/Recall/MRR)로 비교할 수 있다.

**특이사항 (반드시 인지):**
- **이 벤치마크는 IR(qrels)이라 answer_confidence scorer를 학습할 수 없다.** 학습
  라벨은 "모델 답변이 gold answer와 일치했나"인데, IR 벤치마크(AskUbuntu/SciFact
  qrels)에는 free-text gold answer가 없다. 따라서 artifact는 **별도 QA 데이터로
  학습해서 경로로 넘겨야** 한다.
- **다른 도메인(예: SQuAD)에서 학습한 artifact를 넘기면 도메인 shift를 측정하는
  것이지 공정한 비교가 아니다.** 결과에 그 사실을 라벨링해야 한다. (CLI help에도
  명시.)
- 스크립트에 `RANKSMITH_OPENAI_BASE_URL`(+`_MODEL`/`_API_KEY`) 경로를 추가해
  LM Studio 등 OpenAI 호환 엔드포인트로도 돌릴 수 있다(로컬/테스트용). Azure는
  그 env가 없을 때의 기본 경로. `mypy src`만 CI 게이트이고 scripts는 타입체크
  대상이 아니다.
- **README의 표준 벤치마크(AskUbuntu 361쿼리, BM25 top-20, gpt-5.4-nano)에
  answer_confidence를 공정하게 올리는 건 이 개발 환경에선 불가**: (1) AskUbuntu
  corpus 캐시(`.benchmark-cache/askubuntu-bm25`)가 gitignore라 리포에 없고,
  (2) Azure가 VNet 차단이며, (3) AskUbuntu는 gold answer가 없어 answer_confidence
  artifact 자체를 학습할 수 없다. → **AskUbuntu corpus + Azure 접근이 있는 다른
  환경에서 PR을 받아 실행**해야 한다. 그 환경에서도 artifact는 QA 데이터로 별도
  학습해야 하며, SQuAD 등 도메인 밖 artifact를 쓰면 참고 수치(도메인 shift)일 뿐
  README 표에 넣을 공정 비교가 아니다.
- **(2026-07-16 갱신) 그 실행을 턴키로 만드는 도구가 추가됐다** — 절차 전체는
  `docs/benchmarks/answer_confidence_askubuntu.md` 러닝북 참고:
  - `scripts/build_answer_confidence_training_data.py`: SQuAD v1.1 train에서
    질문당 gold 문맥 + BM25 hard negative 문맥 쌍을 만드는 결정적 빌더
    (`benchmarks/bm25.py` 순수 파이썬 BM25 포함). 추론 분포(관련/무관 후보
    양쪽의 답변 채점)와 학습 분포를 일치시킨다.
  - `scripts/train_answer_confidence.py`: generation(라이브 모델) → labeling →
    training → artifact 내보내기 턴키. test roc_auc < 0.6이면 fast fail
    (100샘플 과적합 사고 재발 방지 게이트).
  - `scripts/compare_reranking.py`: `--algorithm` 반복 지정으로 여러 메소드를
    동일 케이스에서 한 번에 실행 가능. answer_confidence 선택 시 artifact
    경로/존재를 실행 전에 검증(fast fail).
  - `scripts/merge_benchmark_reports.py`: 커밋된 v3.merged.json 같은 기존
    결과와 새 단독 run을 병합. 벤치마크 정체성과 알고리즘별 query_id 집합
    일치를 검증해 조건 불일치 병합을 거부한다.
  - 전 경로(빌드→생성→학습→벤치마크→병합)는 실제 SQuAD 다운로드 + 결정적
    fake OpenAI 호환 서버 + 로컬 소형 encoder로 기계 검증 완료. LLM 품질
    수치만 실행 환경 몫이다.

#### 예비 진단 (LM Studio, SciFact, 도메인 밖 artifact — README에 넣지 말 것)
official 스크립트를 LM Studio(qwen3.5-9b)로 SciFact 15케이스(oracle+random,
후보 10)에 돌린 참고 수치. artifact는 SQuAD 학습(도메인 밖), distractor는 무관
랜덤(쉬운 세팅):

| 전략 | NDCG@5 | MRR@5 | Recall@5 | 호출/쿼리 |
| --- | --- | --- | --- | --- |
| answer_confidence (SQuAD art.) | 0.311 | 0.212 | 0.656 | 10 |
| rankgpt_sliding_window | 1.000 | 1.000 | 1.000 | 1 |

listwise가 완벽하고 answer_confidence는 훨씬 나쁘며 10배 비싸다. 단 위의 불리한
조건(도메인 밖 + 쉬운 distractor + 작은 로컬 모델) 때문이며, 공정 비교가 아니다.

### 남은 작업
- ~~README 표 편입 실행~~: 2026-07-20 완료. AskUbuntu 361쿼리, BM25 top-20,
  gpt-5.4-nano 환경에서 러닝북(빌드→학습→벤치마크→병합) 전체를 실행했다.
  `answer_confidence` NDCG@5=0.1722 / MRR@5=0.2862 / Recall@5=0.1435,
  361/361 valid, invalid_rate 0.000. BM25 baseline(NDCG@5=0.3520)보다 낮다 —
  SQuAD 학습 → AskUbuntu 도메인 밖이므로 예상된 결과이며, 이기도록 튜닝하지
  않고 측정된 그대로 README에 반영했다. 근거:
  `benchmark-results/live/askubuntu-bm25-top20-default-live.v4.merged.json`.
  실행 환경 노트: 이 macOS 환경에서 torch로 BERT를 인코딩한 뒤 같은 프로세스에서
  LightGBM `.fit()`을 호출하면 세그폴트가 재현됐다(torch/lightgbm의 OpenMP
  런타임 충돌로 추정, `OMP_NUM_THREADS=1`/`KMP_DUPLICATE_LIB_OK=TRUE`로도
  해결되지 않음). feature 추출과 LightGBM 학습을 별도 프로세스로 분리해
  우회했다 — 동일 하드웨어/라이브러리 조합에서 이 러닝북을 재현하려는
  사람을 위해 기록.
- IR에 맞는 변형: qrels로 학습 가능한 judgment_confidence 리랭커(초기 구현 후
  교체됨)를 되살리면 이 IR 벤치마크로 공정 비교 가능.
- 학습된 artifact는 커밋하지 않는다(`.gitignore`). 배포 시 사용자가 자기 도메인
  데이터로 학습.

### 범위에 대한 사실 정리
- CBDR 논문의 CBDR 메커니즘 자체는 "retrieval을 할지 결정하는 트리거"다. ranksmith에는 retriever가 없으므로 **retrieval 트리거는 이 스펙의 범위 밖**이며 caller 애플리케이션 책임이다.
- CBDR 논문의 confidence 추출(LLM hidden state 직접 접근)과 reranker fine-tuning은 ranksmith의 폐쇄형 모델 / training-free 전제와 충돌한다. 이 스펙이 논문에서 가져오는 것은 **"confidence 신호로 문서 우선순위를 조정한다"는 설계 패턴**뿐이다.

## 2. 요구 사항 및 제약 (Requirements & Constraints)

> **초안 단계 기록 (superseded).** 아래 §2~§7은 설계 초안 시점의 요구/계획
> 기록이다. 최종 확정·구현된 설계는 상단 "확정 설계 (구현됨)" 섹션이 정본이며,
> 초안과의 주요 차이는: ① confidence 소스가 `judgment_confidence`가 아니라
> **`answer_confidence`**(문서로 답변 생성 → 답변 신뢰도 채점)로 바뀌었고,
> ② LCR binning/threshold/환원 gate는 **미채택** — 구현은 confidence 원점수
> 내림차순 + 동점 시 원래 순서 유지다(채택 여부는 실측 후 재검토, Q004 참조).

### 포함 범위
- 기존 reranking 결과(`list[RerankResult]`) 또는 첫 단계 순서를 입력으로 받아, 후보별 confidence로 순위를 보정하는 모듈 1개.
- confidence 소스는 **경로 A로 확정**(Q004-1 resolved). 호출 비용 때문에 경로 B(LCR/MSCP)는 채택하지 않는다.
  - ~~**경로 A 초안**: `StructuralConfidenceEstimator.score_batch(...)`의 `judgment_confidence` 신호.~~ → **교체됨(구현)**: 문서당 `ModelClient.answer` 1회로 답변을 만들고 `estimator.score(AnswerConfidenceInput(context, answer))`로 채점하는 `answer_confidence` 신호. candidate identity 결합은 Strategy 내부 책임.
  - ~~경로 B (LCR/MSCP)~~: 문서당 K회+ 호출 비용으로 미채택.
- ~~정렬 규칙은 LCR 방식을 기본 후보로 한다: `T_upper`/`T_lower` binning + StableSort, query-level gate.~~ → **교체됨(구현)**: confidence 원점수 내림차순 + 동점 시 원래 순서 유지. binning/gate는 미채택(실측 후 재검토).
- ~~기존 순위로의 환원성 보장: gate 조건에서 원래 순위를 그대로 반환.~~ → gate 미채택으로 해당 없음. fast-fail 원칙은 유지(아래 제약).

### 제외 범위
- retrieval 트리거 (CBDR 본래 의미) — caller 책임.
- reranker/scorer fine-tuning — 경로 A의 artifact 학습은 기존 `confidence_training` 스펙 소관.
- 벤치마크 수치 주장 — 측정 전에는 README에 성능 claim을 쓰지 않는다.

### 제약
- fast fail 원칙 유지. silent fallback 금지 — confidence 계산 실패 시 원래 순위로 조용히 돌아가지 않고 에러를 낸다.
- ~~경로 B 채택 시 sampling 전용 요청 경로가 선행 결정 사항이다.~~ → 경로 B 미채택으로 해당 없음.
- 경로 A는 학습된 scorer artifact가 선행 조건이다. 리포에 커밋된 artifact는 없다(정책) — `scripts/build_answer_confidence_training_data.py` → `scripts/train_answer_confidence.py`로 만들어야 한다.

## 3. 상세 설계 (Architecture & Design) — 초안 기록 (superseded)

> 확정 메커니즘은 상단 "확정 설계 (구현됨)"가 정본이다. 아래는 초안 시점 골격.

### 동작 메커니즘 (초안 골격 — binning은 미채택됨)
1. 입력: `query`, 순서 있는 후보 목록(기존 순위 = PrevRank), confidence 파라미터.
2. 후보별 confidence 계산 (경로 A: score_batch 순서 보존 결과를 zip / 경로 B: MSCP).
3. ~~binning: `c >= T_upper → +1`, `c <= T_lower → −1`, 그 외 0.~~
4. ~~StableSort by (bin desc, PrevRank asc). gate 조건이면 PrevRank 그대로.~~
5. 출력: `list[RerankResult]`, metadata에 confidence 점수 포함(구현: `answer_confidence` 키).

### 통합 지점 (초안 후보 → 형태 1로 확정)
- **형태 1: 새 Strategy — 채택됨** (`AnswerConfidenceRerankStrategy`, `docs/wiki/08_custom_strategy_extension.md` 패턴). 다른 Strategy와 조합하려면 caller가 2단계 호출.
- ~~형태 2: `list[RerankResult]`를 받는 post-processor 함수~~ — 미채택 (Q004-2 resolved).

## 5. 에러 핸들링 (Error Handling)
- confidence 소스 실패(artifact 불일치 등): 해당 confidence 에러로 fast fail.
- ~~threshold 검증: `0 <= T_lower <= T_upper <= 1` 위반 시 `ValueError`.~~ → threshold 미채택으로 해당 없음. 구현된 검증: estimator `task_type` 불일치·invalid answer JSON·빈/공백 문서·잘못된 `top_k`는 `RerankError` 계열로 fast fail.
- 빈 후보 목록: 빈 결과 반환 (기존 Strategy들과 동일).

## 6. 테스트 계획 (Test Plan) — 초안 (superseded)
> 실제 수행된 검증은 상단 "검증 (2026-07-05)" 및 `tests/test_confidence_rerank.py`,
> `tests/test_model_architecture.py`, `tests/test_public_protocols.py`의
> answer 경로 테스트가 정본이다. ~~binning 경계값·환원성·StableSort bin 테스트~~는
> binning 미채택으로 함께 폐기됐다.

## 7. 열린 결정 (`docs/wiki/05_open_questions.md` Q004) — 전부 해소됨
1. ~~confidence 소스~~ — **경로 A(structural + LightGBM)로 확정**. 호출 비용으로 경로 B 미채택.
2. ~~형태~~ — **새 Strategy로 확정** (`AnswerConfidenceRerankStrategy` / async 변형, 구현 완료).
3. ~~score → 순위 반영 규칙~~ — **원점수 내림차순 + 동점 시 원래 순서 유지로 확정**. binning/threshold는 미채택(실측 후 재검토).
4. ~~artifact 학습 데이터~~ — **gold answer가 있는 QA 데이터로 확정**. SQuAD v1.1 경로가 도구화됨(`scripts/build_answer_confidence_training_data.py`), ≥30 샘플 요건은 기본 500행으로 충족.

## 스모크 검증 기록 (2026-07-05)
경로 A 파이프라인을 LM Studio(qwen3.5-9b) + libomp 설치 환경에서 부분 관통했다.
- generation: `generate_judgment_confidence_dataset`가 fixture 15개를 엔드투엔드 생성 (LM Studio provider).
- 추론 절반: `FrozenAutoEncoder`(bert-base-uncased) 로드 + 70차원 `structural-v1` 특징 추출 확인.
- 학습: `split.py`의 `MIN_TOTAL_SAMPLES = 30` 가드에 막힘 — 리포 fixture는 15개뿐. (이후 SciFact 실데이터 56개, 그리고 SQuAD 빌더 기본 500행으로 해소 — Q004 참조.)

## 작업 태스크 추적 (Task Checklist)
### Phase 0: 결정
- [x] Q004-1 confidence 소스 = 경로 A
- [x] Q004 나머지 (형태 = 새 Strategy, 순위 규칙 = 원점수 내림차순, 학습 데이터 = QA/SQuAD)
### Phase 1 이후
- [x] ≥30개 실제 라벨 데이터로 scorer artifact 생성 (SciFact 56개 스모크 → SQuAD 500행 도구화)
- [x] 구현: `AnswerConfidenceRerankStrategy` / async 변형 + `ModelClient.answer` + 단위 테스트
- [x] 표준 벤치마크 편입 도구 + 러닝북 (`docs/benchmarks/answer_confidence_askubuntu.md`)
- [x] README 표 편입 실행 (AskUbuntu 캐시 + gpt-5.4-nano 환경에서 러닝북 실행, 도메인 라벨 필수) — 2026-07-20 완료. `answer_confidence` NDCG@5=0.1722, MRR@5=0.2862, Recall@5=0.1435, 361/361 valid, invalid_rate 0.000. BM25 baseline보다 낮음(예상대로, SQuAD 학습 → AskUbuntu 도메인 밖).
