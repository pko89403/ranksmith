# References Index

## 등록된 Reference
- [Is ChatGPT Good at Search? (RankGPT)](references/rankgpt.md): Paper / Listwise Reranking, Sliding Window / 요약 완료
- [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](references/pairwise_ranking_prompting.md): Paper / Pairwise Reranking, PRP / 요약 완료, 사용자 결정 대기

## 예상 카테고리
- listwise reranking
- sliding-window reranking
- pairwise tournament
- bayesian aggregation
- confidence-based reranking
- evaluation datasets and metrics
- Azure OpenAI structured output behavior

## Reference 추가 방법
Reference 처리 전에 `docs/wiki/03_reference_processing.md`를 읽는다.

1. 사용자가 source를 업로드하거나 링크한다.
2. Codex가 `docs/wiki/references/<slug>.md`를 만든다.
3. Codex가 이 index에 다음 정보를 추가한다.
   - reference 이름
   - source 유형
   - 관련 ranksmith 영역
   - 구현 상태
