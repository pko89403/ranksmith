# Reference: AcuRank

## Source
- Paper: AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking
- Local PDF: `docs/wiki/references/AcuRank- Uncertainty-Aware Adaptive Computation for Listwise Reranking.pdf`
- Repo: https://github.com/soyoung97/AcuRank
- License: 공식 Repo의 license가 확정되지 않은 상태로 확인되었으므로 구현 코드를 복사하지 않는다.

## 적용 영역
- Bayesian / TrueSkill 기반 reranking
- Uncertainty-aware adaptive computation
- Listwise reranker output aggregation
- Top-k boundary 중심 candidate refinement

## 핵심 메커니즘
AcuRank는 각 문서의 relevance를 단일 점수가 아니라 TrueSkill 기반 분포 `(mu_i, sigma_i)`로 유지한다. 각 반복에서 문서가 top-k threshold를 넘을 확률 `s_i = P(x_i > t(k))`를 계산하고, `epsilon < s_i < 1 - epsilon` 범위에 있는 불확실한 문서만 다시 listwise reranking한다.

Listwise reranker가 반환한 batch 내부 순위는 최종 순위가 아니라 TrueSkill posterior를 갱신하는 evidence로 사용된다. 반복이 진행되면서 확실한 문서는 uncertain set에서 빠지고, 계산은 top-k 경계 근처의 애매한 문서에 집중된다. 최종 순위는 posterior mean `mu_i` 내림차순으로 만든다.

기본 논문 설정:
- 초기화: `mu_i = first_stage_score_i`, `sigma_i = mu_i / 3`
- target cutoff: `k = 10`
- reranker capacity: `m = 20`
- uncertainty threshold: `epsilon = 0.01`
- stopping threshold: `tau = 10`
- high-precision variant: `epsilon = 0.0001`, `tau = 5`

## ranksmith 매핑
- Strategy: `AcuRankStrategy`, `AsyncAcuRankStrategy`
- Algorithm: `acurank`
- ModelClient contract: 기존 `rank(query, documents) -> {"ranking": [...]}` 재사용
- Public API 영향:
  - `AcuRankStrategy`, `AsyncAcuRankStrategy` 공개
  - first-stage score는 `Document.metadata["score"]`에서 선택적으로 읽음
  - 후보 수가 `target_rank`보다 작으면 effective target rank를 문서 수로 제한
  - 결과 metadata에 `mu`, `sigma`, `top_k_probability`, `reranker_calls` 포함
- Error 동작:
  - invalid listwise permutation은 `RerankParseError`
  - 문서 길이 초과는 `DocumentTooLongError`
  - 잘못된 `tolerance`, `uncertain_threshold`, budget 값은 `ValueError`
  - 불충분한 first-stage score 정책은 spec에서 결정
- 구현 보정:
  - `max_adaptive_reranker_calls`는 adaptive refinement phase만 제한한다.
  - initial pass는 공식 repo처럼 별도 단계로 수행하고, 호출 수는 결과 metadata의 `reranker_calls`에 함께 집계한다.
  - uncertain candidate 수가 `uncertain_threshold`보다 작아지면 `P(x_i > t) > tolerance` 후보를 한 번 더 final rerank한 뒤 종료한다.
  - adaptive 후보 batch는 공식 repo 재현성을 위해 `top_k_probability` 내림차순을 따른다.
  - `batch_parallelism`은 같은 AcuRank iteration 안에서 이미 선택된 독립 batch 호출만 병렬화한다. posterior update는 race를 막기 위해 batch 순서대로 적용한다.
- 추가할 테스트:
  - TrueSkill 초기화 테스트
  - top-k threshold / uncertainty score 계산 테스트
  - uncertain candidate selection 테스트
  - batch partitioning 테스트
  - reranker output 기반 score update 테스트
  - stopping criterion 테스트
  - sync/async strategy 테스트
  - fixture smoke test

## 현재 설계와 충돌
- 기존 `ListwiseStrategy`는 reranker permutation을 직접 window ordering에 반영한다. AcuRank는 permutation을 TrueSkill update evidence로만 사용하고, 최종 정렬은 `mu_i` 기준으로 한다.
- 기존 `Document`에는 first-stage retrieval score 전용 필드가 없다. score를 `Document.metadata`에서 읽을지, strategy parameter로 받을지, 입력 순서 기반 prior를 지원할지 결정해야 한다.
- ranksmith는 hidden truncation과 silent correction을 금지한다. 공식 repo의 invalid permutation 복구 로직은 그대로 따르지 않는다.
- `trueskill>=0.4.5` 의존성을 사용한다.

## 재현성 및 의도적 차이

| 항목 | 논문 / 공식 Repo | ranksmith 구현 | 판단 |
| --- | --- | --- | --- |
| TrueSkill posterior | batch ranking을 TrueSkill evidence로 사용 | 동일하게 `trueskill.rate()`로 posterior 갱신 | 재현 |
| first-stage prior | score가 있으면 `mu=score`, `sigma=score/3` | 모든 문서에 numeric metadata score가 있을 때만 동일 적용 | 재현, partial score는 fast fail |
| initial pass | adaptive budget과 별도 단계 | `initial_pass=True`에서 별도 수행, adaptive budget 미포함 | 재현 |
| adaptive budget | 공식 Repo `hard_constraint`는 adaptive 호출 제한 | `max_adaptive_reranker_calls`로 adaptive refinement만 제한 | 재현 |
| uncertain 감소 종료 | threshold 미만이면 final rerank 후 종료 | `P(x_i > t) > tolerance` 후보 final rerank 후 종료 | 재현 |
| adaptive 후보 순서 | probability 내림차순 정렬 후 chunking | `top_k_probability` 내림차순 사용 | 재현성 우선 |
| invalid permutation | 공식 Repo는 일부 missing/additional index를 복구 | strict JSON permutation 검증 후 실패 | 의도적 차이 |
| prompt / parsing | 공식 Repo 자체 prompt/parser 사용 | ranksmith `ModelClient.rank()`와 `parse_ranking_response()` 재사용 | 의도적 차이 |
| 문서 길이 처리 | Repo 구현 정책에 의존 | `max_document_chars` 초과 시 실패, 숨은 truncation 없음 | 의도적 차이 |
| 병렬 처리 | 공식 알고리즘 의미상 같은 iteration batch는 독립 | `batch_parallelism`으로 batch 호출만 병렬화, posterior update는 순서 고정 | 확장, 알고리즘 의미 유지 |

## 독립 구현 경계
- 논문에 공개된 알고리즘 설명, 수식, 기본 hyperparameter, 공식 Repo의 관찰 가능한 실행 흐름을 설계 입력으로 사용한다.
- 공식 Repo의 Python 함수, prompt template, parsing/cleanup 구현, 데이터 처리 코드는 복사하지 않는다.
- 공식 Repo와 같은 동작을 따르는 경우에도 ranksmith의 기존 abstraction(`Document`, `ModelClient.rank()`, `parse_ranking_response()`, `RerankError`)으로 다시 구현한다.
- 재현성보다 프로젝트 원칙이 우선하는 영역은 명시적으로 deviation으로 기록한다. 현재 deviation은 invalid output repair 미지원, hidden truncation 미지원, ranksmith JSON contract 유지다.

## Do Not Copy
- 공식 GitHub repo의 구현 코드를 복사하지 않는다.
- Repo license가 확정되지 않았으므로 논문에 공개된 알고리즘 설명과 수식, 그리고 관찰 가능한 high-level algorithm flow만 기반으로 독립 구현한다.
- 공식 repo의 prompt parsing, invalid output cleanup, missing index restoration은 ranksmith의 strict JSON / fast fail 정책과 맞지 않으므로 재사용하지 않는다.

## 부족한 정보
- 확인된 부족 정보 없음.
