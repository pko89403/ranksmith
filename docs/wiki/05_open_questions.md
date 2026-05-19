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
