# Reference: A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models

## Source
- Paper: "A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models" (Zhuang et al., SIGIR 2024)
- Local PDF: `docs/wiki/references/A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models.pdf`
- Repo: `https://github.com/ielab/llm-rankers`
- License: Paper text states Creative Commons Attribution-NonCommercial-ShareAlike 4.0. Do not copy external implementation code.

## 적용 영역
- Setwise reranking
- Selection-based ModelClient contract
- Sorting-based top-k reranking
- Heapsort-based efficient reranking

## 핵심 메커니즘
Setwise prompting은 query와 여러 candidate document를 한 번에 제시하고, LLM이 그중 가장 관련 높은 문서 1개를 선택하게 한다.

`setwise.heapsort`는 이 선택 호출을 heap sort의 비교 단계에 넣는다. 각 heapify 단계에서 parent와 child 여러 개를 함께 비교하고, 선택된 문서가 parent가 아니면 parent와 해당 child를 교환한다. Pairwise heapsort가 한 번에 2개만 비교하는 것보다 LLM 호출 수를 줄일 수 있다.

논문은 set size `c`를 hyperparameter로 둔다. `c`가 커지면 호출 수는 줄지만 prompt 길이 때문에 문서 truncation 압력이 커져 효과가 낮아질 수 있다. 기본 실험은 주로 `c=3`을 사용한다.

## ranksmith 매핑
- Strategy: `SetwiseStrategy`, `AsyncSetwiseStrategy`
- Algorithm: `setwise_heapsort`
- ModelClient contract: 기존 `select(query, documents, top_m=1) -> {"selected": [index]}` 재사용
- Public API 영향: `SetwiseStrategy`, `AsyncSetwiseStrategy` export 추가 완료
- Error 동작:
  - `set_size < 3`이면 `ValueError`
  - `max_document_chars < 1`이면 `ValueError`
  - provider가 `select()`를 지원하지 않으면 `RerankInputError`
  - invalid selection response는 `RerankParseError`
  - 문서 길이 초과는 `DocumentTooLongError`
- 추가할 테스트:
  - setwise heapsort가 점수 기반 mock provider로 top 문서를 추출하는지 검증
  - `top_k`가 LLM 호출 수를 줄이는지 검증
  - invalid selection과 provider capability 부족이 fast fail하는지 검증
  - async strategy 동작 검증
  - fixture 기반 smoke test 추가 검토

## 현재 설계와 충돌
- 확인된 충돌 없음.
- 논문의 `listwise.likelihood`는 logits 접근이 필요하다. 현재 `ModelProvider`/Azure API 계약은 logits를 제공하지 않으므로 이번 구현 범위에서 제외한다.
- 논문의 `setwise.bubblesort`도 이번 사용자 결정 범위에서 제외한다.

## Do Not Copy
- 외부 repository 구현 코드를 복사하지 않는다.
- 논문 Figure prompt 문장을 그대로 고정하지 않고, ranksmith의 strict JSON selection 계약에 맞게 재작성한다.
- 논문 실험 수치 또는 benchmark 수치를 README에 추가하지 않는다.

## 부족한 정보
- 없음.

## 구현 및 benchmark 상태
- 구현 완료: `SetwiseStrategy`, `AsyncSetwiseStrategy`
- 기본 public strategy 설정: `set_size=3`
- README benchmark alias: `setwise_hs_s10`
- README benchmark 설정: BM25 top-20 후보, `set_size=10`, `top_k=5`, `@5` 평가
- Benchmark artifact: `benchmark-results/live/askubuntu-bm25-top20-default-live.v3.merged.json`
