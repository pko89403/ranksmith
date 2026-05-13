# AGENTS.md

## 답변 스타일 및 소통 원칙
- **인간의 이해 최우선 (Human-Centric)**: 사용자(인간)에게 직접 질문하거나 답변을 할 때, 그리고 스펙(spec) 등 모든 문서를 작성할 때는 **항상 인간이 읽고 이해하기 가장 좋은 형태(가독성)**에 완벽하게 초점을 맞춘다.
- 기계적인 장황한 서술을 피하고, 직관적이고 명확하게 소통한다.
- 한국어로 짧고 간결하고 깔끔하게 답한다.
- 의미 왜곡이나 중요한 생략은 하지 않는다.
- **장기 지속성 최우선 (Long-term Sustainability First)**: 사용자에게 문제 해결을 위한 선택지를 제시할 때에는, 당장의 작업량이나 난이도와 무관하게 **장기적인 유지보수성과 아키텍처의 건전성에 가장 유리한 올바른 방향을 무조건 1순위 선택지로 제안**한다. (단기적 우회책은 후순위로 미루거나 지양한다)

## 언어 정책
- 사용자와의 대화는 한국어로 한다.
- 개발자/Codex 협업 문서(`AGENTS.md`, `docs/wiki/*`)는 한국어 우선으로 작성한다.
- Public API, 코드 식별자, 패키지 메타데이터, `README.md`는 영어로 유지한다.
- `README.ko.md`는 한국어로 유지한다.
- 외부 reference 요약은 한국어로 작성하되, 핵심 원문 용어는 필요하면 괄호로 병기한다.
- 커밋 메시지는 영어를 권장한다.

## 필수 컨텍스트 로딩
설계나 코드 변경 전 반드시 읽는다:
1. `docs/wiki/00_context.md`
2. `docs/wiki/01_decisions.md`
3. `docs/wiki/02_architecture.md`
4. `docs/wiki/03_reference_processing.md`
5. `docs/wiki/04_references_index.md`
6. `docs/wiki/06_verification_policy.md`
7. 관련 `docs/wiki/references/` 파일

## 사용자 통제 규칙
업로드된 reference가 부족하면 구현을 멈추고 사용자에게 묻는다.

하지 않는다:
- 부족한 algorithm 세부사항을 추론해서 구현
- 사용자 명시 요청 없는 외부 조사
- 사용자 승인 없는 public API 또는 scope 확장
- 외부 reference 구현 코드 복사

허용한다:
- 확인된 부족분 요약
- 구현 영향 설명
- 사용자에게 물어볼 정확한 질문 제안

## 개발 파이프라인 (Development Pipeline)
코딩 어시스턴트는 반드시 다음의 **7단계 파이프라인**을 엄격하게 준수하여 개발을 진행해야 합니다. 절대 사용자의 명시적인 검토 및 승인 없이 임의로 개발 단계로 넘어가지 않습니다.

1. **스펙 생성**: 요구사항을 바탕으로 `docs/specs/TEMPLATE.md` 기반의 스펙(Spec) 문서를 생성합니다.
2. **사용자와 대화**: 생성된 스펙에 대해 사용자에게 피드백을 요청하고 질문 및 대화를 나눕니다.
3. **사용자 최종 승인**: 사용자가 스펙과 설계 방향에 대해 명시적으로 '승인'을 완료할 때까지 대기합니다. (시스템 자동 승인을 무시하고 사용자의 실제 의도를 확인합니다)
4. **개발**: 승인된 스펙 문서의 `Task Checklist`를 기반으로 코드를 작성합니다.
5. **테스트**: 코드를 작성한 후, 관련 단위 테스트를 추가하고 통과시킵니다.
6. **사용자 체크**: 구현 및 테스트 결과를 사용자에게 공유하여 피드백을 받습니다.
7. **작업 정리**: `verify.sh` 등을 통해 모든 검증을 마치고, 문서(`walkthrough.md`, `README` 등)를 정리하여 작업을 마무리합니다.

## 프로젝트 원칙
- 조용한 보정보다 fast fail을 우선한다.
- 숨은 truncation을 하지 않는다.
- 숨은 ranking correction을 하지 않는다.
- Public API는 작게 유지한다.
- `rank`는 1-based로 유지한다.
- `original_index`는 0-based로 유지한다.
- Strategy는 비교 단위다.
- Algorithm은 최종 순위 생성 절차다.
- 새로운 주요 기능(알고리즘 등) 개발 시, 반드시 `docs/specs/TEMPLATE.md`를 복사하여 `docs/specs/spec_[feature].md` 파일을 생성한다.
- 코딩 어시스턴트는 생성된 spec 파일의 `Task Checklist` 섹션을 활용하여 진행 상황을 `[x]`로 체크하며 개발을 진행 및 완료한다.

## 검증
완료를 주장하기 전에 실행한다:
- `./scripts/verify.sh`

Push 전에는 `.githooks/pre-push`와 같은 수준으로 `twine check`까지 확인한다.
