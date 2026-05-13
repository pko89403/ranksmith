# Reference 처리

## 목적
Reference는 구현 지시가 아니다. 먼저 ranksmith 설계 지식으로 변환한다.

## 필수 요약 형식
업로드된 reference 요약은 다음 구조를 따른다:

```markdown
# Reference: <name>

## Source
- Paper:
- Blog:
- Repo:
- License:

## 적용 영역
- <관련 ranksmith 영역>

## 핵심 메커니즘
<짧은 메커니즘 요약>

## ranksmith 매핑
- Strategy:
- Algorithm:
- Public API 영향:
- Error 동작:
- 추가할 테스트:

## 현재 설계와 충돌
- <충돌 내용 또는 "확인된 충돌 없음">

## Do Not Copy
- <라이선스 또는 구현 제약>

## 부족한 정보
- <사용자 입력이 필요한 gap>
```

## Reference 부족 규칙
Reference가 충분한 정보를 제공하지 않으면:
1. 구현 전에 멈춘다.
2. 무엇이 부족한지 분류한다.
3. gap을 `docs/wiki/05_open_questions.md`에 기록한다.
4. 사용자에게 필요한 reference 또는 결정을 요청한다.

사용자 명시 승인 없이 조사, 추론, 발명, 구현하지 않는다.
