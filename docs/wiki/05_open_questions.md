# 열린 질문

## Q001 Pairwise Ranking Prompting 구현 범위
Status: resolved

Resolved Decision:
- 첫 구현 algorithm은 `PRP-Sliding-K`로 제한한다.
- 신규 public API로 `PairwiseStrategy` / `AsyncPairwiseStrategy`를 추가한다.
- 기존 listwise JSON permutation 계약은 유지하고, pairwise 전용 JSON 응답 계약을 추가한다.
- invalid output은 fast fail한다. 단, A/B와 B/A가 모두 유효하지만 서로 충돌하는 경우만 tie로 처리한다.
- `passes=10`을 기본값으로 둔다.
- pairwise `compare()`를 지원하지 않는 provider는 `RerankInputError`로 fast fail한다.

Impact:
- `docs/specs/spec_pairwise_ranking_prompting.md` 기준으로 구현을 진행할 수 있다.
- 구현 전 사용자 최종 승인 단계는 아직 남아 있다.

Needed From User:
- 스펙 검토 후 개발 착수 승인

## Q002 AcuRank public API 및 prior 정책
Status: resolved

Resolved Decision:
- Public API 이름은 `AcuRankStrategy` / `AsyncAcuRankStrategy`로 둔다.
- First-stage score는 `Document.metadata["score"]`에서 읽는다.
- score가 모든 문서에 있으면 `mu=score`, `sigma=score/3`으로 초기화한다.
- score가 모든 문서에 없으면 standard TrueSkill prior를 사용한다.
- score가 일부 문서에만 있거나 numeric이 아니면 `RerankInputError`로 실패한다.
- 결과 metadata에는 `mu`, `sigma`, `top_k_probability`, `reranker_calls`를 포함한다.
- 기본 stopping criterion은 uncertain candidate count이며, `max_adaptive_reranker_calls`로 adaptive refinement budget만 제한할 수 있다.

Impact:
- `docs/specs/spec_acurank.md` 기준으로 구현 완료.

Needed From User:
- 없음

## Q003 Setwise 구현 범위
Status: resolved

Missing:
- 없음

Impact:
- 새 public API인 `SetwiseStrategy` / `AsyncSetwiseStrategy`를 추가했다.
- 이번 구현은 `setwise.heapsort`만 포함하고, `setwise.bubblesort`와 logits 기반 `listwise.likelihood`는 제외한다.

Needed From User:
- 없음

## Q004 Confidence-aware reranking 설계 결정
Status: resolved

Context:
- `docs/specs/spec_confidence_aware_reranking.md` 참조.
- CBDR 논문은 white-box hidden state 전제라 방식 직접 이식이 불가하고, 설계 패턴만 가져온다.
- LCR 논문은 training-free지만 temperature 1 sampling K회가 필요해 현재 ModelClient 계약(temperature 0, JSON-only)과 충돌한다.

Resolved Decision:
1. **confidence 소스 = 경로 A (structural confidence + LightGBM scorer artifact)**. 사용자 결정 근거: 경로 B(LCR/MSCP)는 문서당 K회+ LLM 호출이 필요해 리랭킹 호출 비용을 크게 늘린다. 경로 A는 추론 시 answer 1회 + 로컬 CPU 연산이므로 호출 비용이 낮다.
2. **형태 = 새 Strategy.** `AnswerConfidenceRerankStrategy` / `AsyncAnswerConfidenceRerankStrategy`로 구현 완료 (`src/ranksmith/strategies/confidence.py`).
3. **순위 규칙 = confidence 원점수 내림차순 + 동점 시 원래 순서 유지.** LCR의 binning/이전-순위-환원 gate는 미채택 — 실측에서 필요성이 확인되면 별도 결정으로 재검토.
4. **scorer 학습 데이터 = gold answer가 있는 QA 데이터.** SQuAD v1.1 경로가 `scripts/build_answer_confidence_training_data.py` + `scripts/train_answer_confidence.py`로 도구화됨. 도메인 밖 artifact 라벨링 규칙은 스펙과 `docs/benchmarks/answer_confidence_askubuntu.md` 참조.

Smoke 실행에서 확인된 사실 (2026-07-05, LM Studio qwen3.5-9b + libomp 설치 후):
- **경로 A 전체 관통 성공** — BEIR/SciFact 실데이터 56개(qrels positive 14 + 어휘 유사 hard negative 42) → LM Studio judgment 생성(정답 일치 44 / 불일치 12) → bert 특징 추출 → LightGBM 학습 + sigmoid 보정 → **실제 scorer artifact(scorer.joblib) 생성** → held-out test 특징에 confidence 점수 산출 확인.
- generation 파이프라인: `generate_judgment_confidence_dataset`가 LM Studio provider로 엔드투엔드 동작.
- 추론 절반: `FrozenAutoEncoder`(bert-base-uncased) 로드 + 70차원 `structural-v1` 특징 생성 확인.

당시 발견한 문제와 처리 결과:
1. **로더 버그 (`load_lightgbm_scorer`)** — joblib 아티팩트가 metadata_path 지정 시 booster 텍스트 포맷으로 오인되던 문제. main의 PR #8(`d60cafb`)에서 수정 후 이 브랜치에 머지됨.
2. **macOS ARM 환경 제약 (ranksmith 코드 문제 아님)**: torch(인코더)와 lightgbm을 **한 프로세스**에서 같이 쓰면 이중 OpenMP 충돌로 segfault/hang 가능. 우회는 특징 추출(torch)과 학습/예측(lightgbm)의 프로세스 분리. 환경 제약으로 기록 유지.
3. ~~≥30 샘플 요건~~ — SciFact 실데이터 56개로 해소, 이후 SQuAD 빌더가 기본 500행 생산.

Impact:
- 구현 완료: 전략 2종, `ModelClient.answer`/`AsyncModelClient.answer`, `parse_answer_response`, opt-in 벤치마크 통합.
- 남은 실행 항목(질문 아님): README 표준 벤치마크 편입 실행 — `docs/benchmarks/answer_confidence_askubuntu.md` 러닝북 참조.

Needed From User:
- 없음

## 형식
```markdown
## Q001 <topic>
Status: blocked | needs-user-decision | resolved

Missing:
- <부족한 concept/API/evaluation/license detail>

Impact:
- <막힌 구현 결정>

Needed From User:
- <필요한 reference 또는 사용자 결정>
```
