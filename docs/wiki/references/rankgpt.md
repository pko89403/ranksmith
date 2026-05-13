# Reference: Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents

## Source
- Paper: "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents" (Sun et al.)
- Blog: -
- Repo: https://github.com/sunnweiwei/RankGPT
- License: MIT (for standard implementations of RankGPT)

## 적용 영역
- Listwise Reranking
- Sliding Window Algorithm

## 핵심 메커니즘
LLM의 입력 토큰 한계를 극복하기 위해, 전체 문서 집합을 뒤에서부터 앞으로(Back-to-first) 슬라이딩 윈도우로 처리합니다. 각 윈도우 내에서 LLM이 순위를 매기며, 상위 문서들은 다음(앞쪽) 윈도우의 평가 대상으로 올라가고(Bubble up), 하위 문서들은 최종 순위에서 그대로 탈락/고정되는 방식입니다.

## ranksmith 매핑
- Strategy: `ListwiseStrategy`
- Algorithm: `sliding_window` (또는 기존 방식과 구분되는 `rankgpt_sliding_window`)
- Public API 영향: `ListwiseStrategy`에 `algorithm="rankgpt_sliding_window"` 추가 가능성. 외부 노출 API(`AzureOpenAIReranker`) 변경 없음.
- Error 동작: LLM이 지정된 문서들의 순위를 반환하지 않으면 `RerankParseError` 발생.
- 추가할 테스트: Back-to-first 윈도우 이동 시, 이전 윈도우의 1위 문서가 다음 윈도우에 정상적으로 포함되는지 확인하는 테스트.

## 현재 설계와 충돌
- **방향성 및 집계 방식:** 현재 `ranksmith`의 `sliding_window`는 앞에서부터 뒤로 진행하며 점수를 누적 합산(Aggregation)하지만, 논문은 뒤에서 앞으로 진행하며 최상위 문서를 밀어 올리는(Bubble) 승자 진출 방식입니다. 현재 설계와 근본적으로 다른 작동 방식을 가집니다.
- **출력 형식:** 논문은 `[2] > [3] > [1]` 형태를 사용하지만, 우리는 JSON 포맷 강제를 유지하므로 프롬프트(Prompt) 구조만 다르게 가져가야 합니다.

## Do Not Copy
- 기존 RankGPT 공식 리포지토리의 코드를 그대로 복사하지 않는다.
- 오직 Back-to-first 윈도우 이동과 승자 진출(Bubble) 메커니즘의 수학적 논리만 참고한다.

## 부족한 정보
- 확인된 부족분 없음 (RankGPT 리포지토리 코드 확인 완료).
  - **순위 고정 규칙:** 윈도우가 뒤에서 앞으로(step만큼) 이동할 때, 윈도우 내 하위 `step` 개수의 문서들은 평가 직후의 위치에서 최종 순위가 확정(고정)되며 다음 평가에서 제외됨. 상위 `window_size - step` 개의 문서들만이 다음 윈도우 평가에 참여(Bubble-up)함.
