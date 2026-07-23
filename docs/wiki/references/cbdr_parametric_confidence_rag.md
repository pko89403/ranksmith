# Reference: Rethinking LLM Parametric Knowledge as Confidence (CBDR)

## Source
- Paper: Rethinking LLM Parametric Knowledge as Confidence for Effective and Efficient Retrieval-Augmented Generation (Anonymous ACL submission, 16pp)
- Local PDF: `docs/wiki/references/Rethinking LLM Parametric Knowledge as Confidence for Effective and Efficient Retrieval-Augmented Generation.pdf`
- Repo: 논문에 명시되지 않음 (anonymous submission, 코드 링크 없음)
- License: 논문에 명시되지 않음

## 적용 영역
- confidence-aware reranking의 개념적 로드맵 (runtime readiness 스펙이 "CBDR-ready signal"이라 부른 출처)
- confidence 변화(Inc) 기반 preference 신호 설계
- retrieval 트리거 결정 (ranksmith 범위 밖 — 아래 충돌 참조)

## 핵심 메커니즘
세 가지 구성 요소:

1. **Confidence detection model E**: target LLM이 질문 처리 중 만든 내부 hidden state(본문 기준 Mid_Layer, 첫 답변 토큰 생성 직전)를 입력으로 받아 "정답 가능 확신" 확률을 내는 5-layer MLP(2M 파라미터). NQ 질문의 정답/오답 여부를 라벨로 학습.
2. **Confidence-change 기반 reranker fine-tuning**: context 유무에 따른 confidence 변화 `Inc(Q, Ci) = Conf(H_{M,Q+Ci}) − Conf(H_{M,Q})`를 계산해, 상승 Top-5를 positive / 하락 Top-5를 negative로 하는 preference dataset(NQ_Rerank, 7,622 train)을 만들고 bge-reranker-v2-m3를 InfoNCE로 fine-tuning. 즉 "의미 유사도"가 아니라 "downstream LLM의 confidence를 올리는 문서"를 위로 올리는 reranker.
3. **CBDR (Confidence-Based Dynamic Retrieval)**: 원 질문에 대한 confidence가 임계값 β를 넘으면 retrieval/reranking을 생략하고 직접 생성, 아니면 전체 RAG 파이프라인 실행. β=0.98이 실험 최적.

보고된 결과: fine-tuned reranker가 NQ_Rerank에서 원본 bge 대비 P@1 +5.19pp(91.20), Qwen3-Reranker-8B 대비 +3.95pp. RAG 정확도는 target LLM(Llama3-8B)에서 NQ Top-3 +4.7pp. CBDR β=0.98에서 정확도 67.8%로 전량 검색(β=1.00, 66.9%)보다 높으면서 retrieval cost 7.1pp 절감. Abstract의 "+5.6pp" 헤드라인은 어떤 셀 비교인지 본문에 도출식이 없음.

주의 — 논문 내부 불일치(원문 병기, 해명 없음):
- hidden state 위치: 본문 §3.1은 Mid_Layer(Layer/2), Algorithm 1은 `hidden_states[−1]`(마지막 층)
- NQ_Confidence 규모: §4.1은 1k/300/500, Appendix G는 2,000/1,000/500
- E 학습 epoch: Appendix B는 30, Appendix G는 100

## ranksmith 매핑 (제안, 미확정)
- Strategy: 직접 매핑 없음 — 이 논문은 개념적 로드맵으로만 사용
- Algorithm: 없음
- Public API 영향: 없음 (아래 충돌로 인해 방식 자체를 이식할 수 없음)
- Error 동작: 해당 없음
- 추가할 테스트: 해당 없음

## 현재 설계와 충돌
- **White-box 전제**: 이 방법은 학습(hidden state 수집에 target LLM forward 74,204회)과 추론(질문마다 hidden state 추출) 모두 target LLM 내부 접근이 필요하다. ranksmith는 폐쇄형 API 모델이 전제라 **직접 적용 불가**. ranksmith의 구현된 대안이 Trust in One Round 방식(출력 텍스트 → frozen proxy encoder → structural confidence)이며, 이 논문에서 가져올 것은 "confidence 신호로 문서 선호를 정한다"는 설계 패턴뿐이다.
- **Reranker fine-tuning 필요**: ranksmith core는 training-free reranking 지향. preference dataset 구축 + fine-tuning 경로는 core 밖이다.
- **CBDR = 검색 트리거**: CBDR 자체는 reranker가 아니라 "retrieval을 할지 결정하는 스위치"다. ranksmith에는 retriever가 없으므로 트리거 부분은 라이브러리 범위 밖이고, caller 애플리케이션 몫이다.
- **target LLM 종속**: 논문 스스로 한계로 명시 — reranker가 특정 downstream LLM preference에 강결합되어 LLM 교체 시 재학습 필요.

## Do Not Copy
- 공개 코드 없음(복사 대상 자체가 없음). 라이선스 미명시이므로 수식/알고리즘 서술만 설계 입력으로 사용한다.

## 부족한 정보
- retriever 종류/corpus, Algorithm 1의 top_k 배수 r, InfoNCE τ와 negative 수 N
- Mid_Layer vs 마지막 층 불일치의 실제 구현
- "+5.6pp" 헤드라인의 산출 근거
- RAG accuracy 채점 방식(EM/substring 여부)
- 코드/라이선스
