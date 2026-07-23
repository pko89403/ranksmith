# Reference: LLM-Confidence Reranker (LCR)

## Source
- Paper: LLM-Confidence Reranker: A Training-Free Approach for Enhancing Retrieval-Augmented Generation Systems (Zhipeng Song et al., arXiv:2602.13571v1, 2026-02-14, Preprint submitted to Elsevier)
- Local PDF: `docs/wiki/references/LLM-Confidence Reranker- A Training-Free Approach for Enhancing Retrieval-Augmented Generation Systems.pdf`
- Repo: 논문에 명시되지 않음
- License: 논문에 명시되지 않음 (코드 공개 언급 없음)

## 적용 영역
- confidence-based reranking (CBDR 방향의 후보 접근 중 하나)
- black-box confidence 추정 (샘플링 기반, hidden state 불필요)
- 기존 reranker 뒤에 붙는 post-reranking 단계

## 핵심 메커니즘
Training-free, plug-and-play 리랭커. LLM을 순수 black-box(입출력만 접근)로 두고 confidence를 추정한다.

**MSCP (Maximum Semantic Cluster Proportion)**: query(또는 query+document)에 대해 temperature 1로 K개(실험 K=10) 답변을 multinomial sampling한다. 같은 LLM에게 답변 쌍의 양방향 entailment를 판정시켜 semantic cluster를 만들고(greedy, O(K·M) 호출), 가장 큰 클러스터의 비율 `max|s_m| / K`를 confidence로 쓴다.

**2단계 LCR 알고리즘** (hyperparameter 3개: `T_query`, `T_upper`, `T_lower`):
1. Binning: 문서별 joint confidence `C(q,d)`를 High(≥T_upper, +1) / Medium(0) / Low(≤T_lower, −1)으로 구간화.
2. Ranking: query confidence `C(q) < T_query`이면 [bin 점수 내림차순, 기존 점수 내림차순]으로 StableSort. `C(q) ≥ T_query`이면 **원래 순위 유지** (LLM이 이미 답을 알면 문서 신호가 약하다는 근거). `T_query=0`이면 알고리즘 전체가 원래 순위로 환원되어 baseline 이하로 떨어지지 않는다.

**보고된 결과** (NDCG@5, BEIR 6종 + TREC DL19/20, top-10 재정렬): BM25 계열에서 상대 개선 최대 20.6%(YesNo+BEIR 0.1737→0.2095), Contriever 계열 최대 32.4%(YesNo+BEIR 0.2281→0.3021), 전체 평균 약 3.6%. 논문 주장 기준 "+LCR" 구성에서 하락 사례 없음. Qwen2.5-7B-Instruct 기본, 7–9B 4종 모델 모두에서 개선(InternLM 최고, Qwen 최저). MSCP가 Semantic Entropy보다 일관되게 우수.

## ranksmith 매핑 (제안, 미확정)
- Strategy: 신규 후보 `ConfidenceRerankStrategy` 류의 post-reranking Strategy (기존 Strategy 출력 뒤에 적용하는 형태)
- Algorithm: 미정
- Public API 영향: 미정 — 아래 "현재 설계와 충돌" 해소가 선행되어야 함
- Error 동작: fast fail 원칙 유지 (entailment 응답이 3개 라벨 밖이면 파싱 에러)
- 추가할 테스트: MSCP 클러스터링(양방향 entailment, greedy), binning 경계값, T_query=0 환원성(원래 순위 보존), StableSort 안정성

## 현재 설계와 충돌
- **temperature 계약**: ranksmith `ModelClient`는 temperature 0 고정(결정적 JSON)이다. MSCP는 temperature 1 multinomial sampling K회가 필수라서, sampling 전용 요청 경로가 필요하다.
- **호출 수**: 문서당 K=10 샘플 + O(K·M) entailment 판정 + query 자체 confidence. 논문은 총 호출 수 공식을 제시하지 않지만(명시되지 않음) 기존 전략 대비 호출량이 크게 늘어난다. 호출 수 추정 정책(advisor method-guide)과의 정합 필요.
- **출력 계약**: MSCP의 답변 샘플은 자유 텍스트다. 현재 ModelClient의 JSON-only 계약과 다르다.
- **기존 점수 필요**: Stage 2 정렬이 PrevScore(기존 retriever/reranker 점수)를 요구한다. `RerankResult.rank` 또는 `Document.metadata["score"]`를 입력으로 받는 adapter 설계 필요.

## Do Not Copy
- 논문에 공개 코드가 없으므로 복사 문제는 없으나, 논문 라이선스가 명시되지 않았으므로 수식·알고리즘 설명만 설계 입력으로 사용한다.
- 프롬프트 3종(answer w/o doc, answer w/ doc, entailment)은 논문 p.10에 공개된 것을 참고하되 ranksmith 계약에 맞게 재작성한다.

## 부족한 정보
- Table 2–3을 만든 `T_query`/`T_upper`/`T_lower` 실제 값 (논문은 "tuning 후 최적"만 언급; 가이드는 UT≈0.9, LT≈0.1–0.4)
- query당 총 LLM 호출 수 / 지연 / 토큰 비용 (논문 미제시)
- 샘플링 디코딩 파라미터 상세 (T=1 외 top-p/top-k/max tokens 미명시)
- greedy 클러스터링의 대표 샘플 선택 방식 (초기 샘플 vs embedding centroid — 실험에서 무엇을 썼는지 미명시)
- 코드/라이선스
