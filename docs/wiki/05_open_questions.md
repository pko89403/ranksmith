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

## 형식
```markdown
## Q004 Confidence-aware reranking 설계 결정
Status: partially-resolved

Context:
- `docs/specs/spec_confidence_aware_reranking.md` (Draft) 참조.
- CBDR 논문은 white-box hidden state 전제라 방식 직접 이식이 불가하고, 설계 패턴만 가져온다.
- LCR 논문은 training-free지만 temperature 1 sampling K회가 필요해 현재 ModelClient 계약(temperature 0, JSON-only)과 충돌한다.

Resolved Decision:
1. **confidence 소스 = 경로 A (structural confidence + LightGBM scorer artifact)**. 사용자 결정 근거: 경로 B(LCR/MSCP)는 문서당 K회+ LLM 호출이 필요해 리랭킹 호출 비용을 크게 늘린다. 경로 A는 추론 시 judgment 1회 + 로컬 CPU 연산이므로 호출 비용이 낮다.

Needed From User (경로 A 확정에 따른 잔여 결정):
2. 형태: 새 Strategy vs `list[RerankResult]` post-processor.
3. threshold/score→순위 반영 규칙 (bin 방식 유지 여부, 기본값).
4. scorer artifact 학습에 쓸 실제 데이터셋 소스 (아래 스모크 실행에서 확인된 ≥30 샘플 요건).

Smoke 실행에서 확인된 사실 (2026-07-05, LM Studio qwen3.5-9b + libomp 설치 후):
- **경로 A 전체 관통 성공** — BEIR/SciFact 실데이터 56개(qrels positive 14 + 어휘 유사 hard negative 42) → LM Studio judgment 생성(정답 일치 44 / 불일치 12) → bert 특징 추출 → LightGBM 학습 + sigmoid 보정 → **실제 scorer artifact(scorer.joblib) 생성** → held-out test 특징에 confidence 점수 산출 확인.
- generation 파이프라인: `generate_judgment_confidence_dataset`가 LM Studio provider로 엔드투엔드 동작.
- 추론 절반: `FrozenAutoEncoder`(bert-base-uncased) 로드 + 70차원 `structural-v1` 특징 생성 확인.

발견한 문제 (구현 착수 전 처리 필요):
1. **로더 버그 (`load_lightgbm_scorer`)**: 학습 파이프라인은 `joblib.dump({"metadata":..., "scorer":...})` 형식으로 저장하는데, 로더는 **metadata_path가 주어지면** 아티팩트를 원시 LightGBM Booster 텍스트 파일로 간주(`_load_lightgbm_booster_scorer`)해 로드 실패("Unknown model format"). 그런데 `spec_confidence_runtime_readiness.md` §3의 `from_artifact`는 `load_lightgbm_scorer(path, metadata_path=metadata_path)`로 문서화돼 있고 `export_scorer_artifact`는 sidecar `write_metadata_json`까지 제공 → 스펙대로 만든 아티팩트를 스펙대로 로드하면 깨진다. 올바른 호출은 `from_artifact(path)`(metadata는 joblib 내장) — 확인함. YAGNI 리뷰 finding #1(로더가 4개 포맷 지원, 학습은 1개만 생산)과 동일 원인. 수정: 로더가 metadata_path 유무가 아니라 **아티팩트 내용**으로 포맷을 판별하도록 하거나, 미사용 booster 포맷을 제거.
2. **macOS ARM 환경 제약 (ranksmith 코드 문제 아님)**: torch(인코더)와 lightgbm을 **한 프로세스**에서 같이 쓰면 이중 OpenMP 충돌로 segfault/hang. `score_batch`(인코더+scorer 동시)가 이 박스에서 완료되지 않음. 우회: 특징 추출(torch)과 학습/예측(lightgbm)을 별도 프로세스로 분리 — 이 방식으로 artifact를 생성함. scoring 로직 자체는 lightgbm 단독 프로세스에서 정상 동작 확인.
3. ~~≥30 샘플 요건~~ — SciFact 실데이터 56개로 해소.

## Q001 <topic>
Status: blocked | needs-user-decision | resolved

Missing:
- <부족한 concept/API/evaluation/license detail>

Impact:
- <막힌 구현 결정>

Needed From User:
- <필요한 reference 또는 사용자 결정>
```
