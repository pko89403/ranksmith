# 검증 정책

## 원칙
작동하는 것처럼 보이게 만드는 구현은 실패로 간주한다.

Codex는 실제 검증 없이 완료, 성공, 동작을 주장하지 않는다.

## 완료 주장 조건
완료를 주장하려면:
1. 관련 테스트 또는 검증 명령이 있어야 한다.
2. 검증 명령을 실제로 실행해야 한다.
3. 출력 결과를 요약해야 한다.
4. 실패한 검증은 숨기지 않아야 한다.

## 금지 표현
검증 없이 다음 표현을 쓰지 않는다:
- 동작합니다.
- 구현 완료.
- 테스트 통과 예상.
- 문제없어 보입니다.
- 아마 됩니다.

## 허용 형식
```text
검증:
- `uv run pytest -q` -> 13 passed
- `uv run ruff check .` -> All checks passed
- `uv run mypy src` -> Success: no issues found
- `uv build` -> wheel/sdist built
```

검증하지 못한 경우:

```text
검증하지 못했습니다.
이유: <이유>
남은 위험: <위험>
다음 확인 방법: <명령>
```

## Hook 정책
- `pre-commit`: 빠른 품질 체크만 실행한다.
- `pre-push`: 테스트, lint, format, type check, build, twine check를 실행한다.
- CI: 로컬 hook 우회를 막는 최종 방어선이다.

설치:

```bash
./scripts/install-hooks.sh
```

## 로컬 Hook 한계
Git hook은 `--no-verify`로 우회할 수 있다.

따라서 GitHub에서는 `main` branch protection과 CI 필수 통과를 설정해야 한다.
