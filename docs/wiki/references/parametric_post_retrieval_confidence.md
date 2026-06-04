# Reference: Rethinking LLM Parametric Knowledge as Post-retrieval Confidence for Dynamic Retrieval and Reranking

## Source
- Paper: Rethinking LLM Parametric Knowledge as Post-retrieval Confidence for Dynamic Retrieval and Reranking
- Local PDF: `docs/wiki/references/Rethinking LLM Parametric Knowledge as Post-retrieval Confidence for Dynamic Retrieval and Reranking`
- arXiv: 2509.06472v2
- Authors: Haoxiang Jin, Ronghan Li, Zixiang Lu, Qiguang Miao
- License: arXiv non-exclusive distribution license

## 적용 영역
- confidence-based reranking
- post-retrieval context filtering
- future CBDR-style retrieval routing

## 핵심 메커니즘
논문은 LLM 내부 hidden state로 confidence detector를 학습하고, retrieved context 주입 전후의 confidence 변화량을 preference signal로 사용한다.

핵심 수식:

```text
Inc(Q, C_i) = Conf(H_M,Q+C_i) - Conf(H_M,Q)
```

`Inc(Q, C_i) > 0`이면 해당 context는 target LLM의 answerability를 높이는 positive context로 보고, `Inc(Q, C_i) < 0`이면 negative context로 본다.

논문 원형은 이 preference dataset으로 reranker를 fine-tune하고, 별도로 CBDR을 통해 query-only confidence가 높으면 retrieval/reranking을 skip한다.

## ranksmith 매핑
- Strategy: 새 `ConfidenceGainStrategy` 후보
- Algorithm: `confidence_gain`
- Public API 영향:
  - `ranksmith.confidence`에 query-only/contextual answerability input type 추가 필요
  - `ranksmith.confidence_generation`에 answerability confidence dataset generation 추가 필요
  - `ranksmith.confidence_training` task type 확장 필요
  - `ranksmith.strategies`에 confidence gain 기반 Strategy 추가 가능
- Error 동작:
  - confidence artifact/task mismatch는 fast fail
  - confidence score가 finite probability가 아니면 fast fail
  - 문서별 confidence scoring 실패는 조용히 fallback ranking하지 않고 실패
- 추가할 테스트:
  - base confidence와 context confidence 차이 계산
  - confidence gain 내림차순 정렬
  - 동점 시 original_index 유지
  - query-only/contextual task mismatch 실패
  - scorer score 범위 검증

## 현재 설계와 충돌
- 논문 원형은 open-source LLM hidden state 접근을 전제한다.
- 논문 원형은 confidence detector 학습과 reranker fine-tuning을 포함한다.
- ranksmith는 closed LLM API, hidden state/logits/attention 비의존, runtime training-free reranking을 지향한다.

따라서 논문 원형을 그대로 구현하지 않는다.
ranksmith에서는 이미 구현된 structural confidence layer를 closed model proxy confidence로 사용한다.
runtime reranking은 training-free지만, confidence scorer artifact는 사전에 생성/학습되어 있어야 한다.

## Do Not Copy
- 논문 구현 코드가 제공되더라도 그대로 복사하지 않는다.
- ranksmith 구현은 public API, error policy, metadata validation, optional dependency 정책을 따른다.

## 부족한 정보
- query-only answerability confidence의 canonical input schema를 확정해야 한다.
- query+context answerability confidence의 canonical input schema를 확정해야 한다.
- generation label을 exact match로만 둘지, 별도 evaluator를 도입할지 결정해야 한다.
- CBDR retrieval skip은 이번 strategy 구현과 같은 범위에 넣을지 별도 spec으로 분리할지 결정해야 한다.
