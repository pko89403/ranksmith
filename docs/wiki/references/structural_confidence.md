# Reference: Trust in One Round

## Source
- Paper: Trust in One Round: Confidence Estimation for Large Language Models via Structural Signals
- Blog:
- Repo:
- License:

## 적용 영역
- `ranksmith.confidence`
- closed model output confidence estimation

## 핵심 메커니즘
closed model의 hidden state를 직접 보지 않는다. 대신 closed model이 만든 `context + answer` 또는 ranksmith용 `query + document + judgment`를 frozen encoder에 넣어 token-level hidden trajectory를 만든다.

이 trajectory를 spectral stability, local variation, shape coherence 같은 structural feature로 요약하고, 이미 학습된 lightweight scorer artifact가 confidence probability를 추정한다.

## ranksmith 매핑
- Strategy: 추가하지 않음
- Algorithm: 추가하지 않음
- Public API 영향: `ranksmith.confidence` submodule 추가
- Error 동작: confidence-specific error로 fast fail
- 추가할 테스트: feature schema, scorer artifact validation, HF token handling, numeric stability

## 현재 설계와 충돌
- ranksmith core는 closed API 기반 training-free reranking을 지향한다.
- Trust 방식은 scorer 학습이 필요하므로 Phase 1은 inference-only로 제한한다.
- Phase 1은 학습된 compatible scorer artifact가 이미 있다는 전제에서만 confidence를 계산한다.
- Phase 1은 benchmark claim이나 semantic feature fusion을 제공하지 않는다.

## Do Not Copy
- 외부 reference 구현 코드를 복사하지 않는다.
- 논문이 명시하지 않은 feature 세부 계산은 ranksmith `structural-v1` schema로 고정한다.

## 부족한 정보
- Phase 2 training dataset schema
- Phase 2 label 생성 방식
- Phase 2 artifact save/export helper
