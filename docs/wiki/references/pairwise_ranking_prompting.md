# Reference: Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting

## Source
- Paper: "Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting" (Qin et al., arXiv:2306.17563v2)
- Blog: -
- Repo: 논문 내 code/data release 계획 언급은 있으나, 현재 로컬 reference에는 구현 저장소가 포함되지 않음
- License: 로컬 PDF 기준 명시 확인 불가

## 적용 영역
- Pairwise Reranking
- Pairwise prompt contract
- Pairwise aggregation algorithm

## 핵심 메커니즘
Pairwise Ranking Prompting(PRP)은 query와 두 문서만 LLM에 제시해 어느 문서가 더 관련 있는지 비교한다. 비교 단위를 작게 만들어 listwise permutation 생성 부담과 pointwise score calibration 문제를 줄인다.

논문은 같은 문서 쌍을 A/B 순서와 B/A 순서로 두 번 비교해 위치 편향을 완화한다. 두 비교가 같은 문서를 선호하면 승패를 확정하고, 충돌하면 동률로 본다.

제안된 ranking 절차는 세 가지다.
- `PRP-Allpair`: 모든 문서 쌍을 비교하고 승수 기반 score로 정렬한다. 동률은 초기 순위로 해소한다. 비용은 `O(N^2)`이다.
- `PRP-Sorting`: pairwise comparator를 정렬 알고리즘에 연결한다. 논문은 Heapsort를 사용한다. 비용은 `O(N log N)`이다.
- `PRP-Sliding-K`: 뒤에서 앞으로 인접 문서 쌍을 비교하고 필요하면 swap한다. K pass를 수행해 top-K 품질을 노린다. 비용은 `O(KN)`이다.

## ranksmith 매핑
- Strategy: 신규 `PairwiseStrategy` 후보
- Algorithm: `prp_allpair`, `prp_sorting`, `prp_sliding_k` 후보
- Public API 영향: 1차 구현에서 `PairwiseStrategy`와 `AsyncPairwiseStrategy`를 공개 API에 추가한다.
- Provider 영향: 기존 `LLMProvider.rank(query, documents)`는 listwise JSON permutation 계약으로 유지하고, pairwise 전용 `compare()` 계약을 추가한다.
- Error 동작: ranksmith 원칙상 invalid generation을 동률로 조용히 처리하지 않고 `RerankParseError`로 실패시키는 방향이 더 일관적이다. 단, A/B와 B/A가 모두 유효하지만 선호가 충돌하는 경우는 동률로 처리 가능하다.
- 기본값: 1차 구현의 `prp_sliding_k`는 논문 기준과 맞춰 `passes=10`을 기본값으로 둔다.
- 호출 수: query당 `2 * passes * max(document_count - 1, 0)`회 provider 호출이 필요하다.
- 추가할 테스트:
  - pairwise 비교가 A/B와 B/A 두 번 호출되는지 검증
  - consistent preference가 승점 또는 swap에 반영되는지 검증
  - conflicting preference가 동률로 처리되는지 검증
  - invalid pairwise 응답이 fast fail하는지 검증
  - `PRP-Allpair` 또는 `PRP-Sliding-K`의 최종 rank, original_index 보존 검증

## 현재 설계와 충돌
- 현재 공식 범위는 `ListwiseStrategy`뿐이고, `PairwiseStrategy`는 향후 후보로만 기록되어 있다.
- 현재 provider는 listwise JSON permutation만 요구한다. 논문은 binary choice prompt를 사용한다.
- 논문은 generation 실패 또는 충돌을 동률로 완화할 수 있지만, ranksmith는 조용한 보정과 숨은 fallback을 금지한다.
- `PRP-Allpair`는 구현이 단순하고 reference에 충실하지만 API 호출 수가 많다.
- `PRP-Sliding-K`는 비용이 낮지만 입력 순서 민감도가 더 크다.
- `PRP-Sorting`은 효율적이지만 비전이성 비교에서 정렬 알고리즘 결과가 흔들릴 수 있다.

## Do Not Copy
- 외부 구현 코드가 제공되지 않았으므로 코드를 복사하지 않는다.
- 논문 prompt 문장은 그대로 고정하지 않고 ranksmith의 응답 계약에 맞게 재작성한다.
- 논문 실험 데이터나 예시 JSON을 repository에 복사하지 않는다.

## 부족한 정보
- 사용자 결정 완료: 1차 구현은 신규 `PairwiseStrategy` / `AsyncPairwiseStrategy`와 `PRP-Sliding-K`로 제한한다.
- 사용자 결정 완료: pairwise 전용 JSON 응답 계약을 추가한다.
- 사용자 결정 완료: invalid output은 fast fail한다. 단, A/B와 B/A가 모두 유효하지만 서로 충돌하는 경우만 tie로 처리한다.
- 사용자 결정 완료: `passes=10`을 기본값으로 둔다.
- 사용자 결정 완료: pairwise `compare()`를 지원하지 않는 provider는 `RerankInputError`로 fast fail한다.
