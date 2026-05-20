# Reference: TourRank

## Source
- Paper: TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy
- Venue: WWW 2025
- Local PDF: `docs/wiki/references/TourRank- Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy.pdf`
- License: ACM publication. Do not copy implementation code.

## 적용 영역
- Tournament-inspired reranking
- Selection-based ModelClient contract
- Multi-stage group selection
- Multi-round point aggregation

## 핵심 메커니즘
TourRank는 후보 문서를 토너먼트 참가자로 보고, 각 stage에서 group별로 LLM이 top-`m` 문서를 선택하게 한다. 선택되어 다음 stage로 진출한 문서는 점수 `+1`을 받고, 여러 tournament round의 점수를 합산해 최종 순위를 만든다.

논문 기본 실험 설정은 top-100 후보에 대해 `100 -> 50 -> 20 -> 10 -> 5 -> 2` stage를 사용하며, `TourRank-r`은 이 tournament를 `r`번 수행한다.

## ranksmith 매핑
- Strategy: `TourRankStrategy`, `AsyncTourRankStrategy`
- Algorithm: `tourrank_r`
- ModelClient contract: `select(query, documents, top_m) -> {"selected": [...]}`
- Public API 영향: `TourRankStrategy`, `AsyncTourRankStrategy`, `TourRankStageConfig`, `parse_selection_response()`
- Error 동작: stage와 문서 수 불일치, invalid selection은 fast fail
- 추가할 테스트: parser, sync/async TourRank, fixture smoke, example 실행

## 현재 설계와 충돌
- 기존 listwise `rank()`는 전체 permutation을 반환하지만, TourRank는 group별 selected indexes가 필요하다.
- 따라서 기존 `ListwiseStrategy.algorithm` 확장이 아니라 새 Strategy와 `ModelClient.select()` 계약이 필요하다.

## Do Not Copy
- 외부 TourRank repository 구현 코드를 복사하지 않는다.
- 논문에 공개된 알고리즘 설명과 prompt 의도만 ranksmith의 strict JSON 정책에 맞게 재구현한다.

## 부족한 정보
- 논문은 group 내부 shuffle seed를 고정 API로 정의하지 않는다. ranksmith는 재현성을 위해 `shuffle_seed=13`을 기본값으로 둔다.
