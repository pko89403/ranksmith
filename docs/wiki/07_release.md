# Release

## 배포 환경
- dev: PR/branch CI에서 test, lint, format, type check, build만 수행한다.
- stage: pre-release tag를 TestPyPI에 배포한다.
- release: stable tag를 PyPI에 배포한다.

로컬에서 `twine upload dist/*`로 API token을 입력해 배포하지 않는다.

## Trusted Publisher 설정
PyPI와 TestPyPI 프로젝트의 Publishing 설정에서 다음 GitHub publisher를 등록한다.

- Owner: `pko89403`
- Repository name: `ranksmith`
- Workflow name: `ci.yml`
- Environment name: 비워두거나 `(Any)`를 사용한다.

## Stage 절차: TestPyPI
pre-release version을 사용한다.

예:
- `0.1.1rc1`
- `0.1.1b1`
- `0.1.1a1`
- `0.1.1.dev1`

절차:
1. `pyproject.toml`의 version을 pre-release version으로 올린다.
2. `./scripts/verify.sh`를 실행한다.
3. `UV_CACHE_DIR=.uv-cache uv tool run twine check dist/*`를 실행한다.
4. 변경사항을 commit하고 `main`에 push한다.
5. `v<version>` 형식의 pre-release tag를 push한다.

```bash
git tag v0.1.1rc1
git push origin v0.1.1rc1
```

tag에 `a`, `b`, `rc`, `dev`가 포함되면 `.github/workflows/ci.yml`의
`publish-testpypi` job이 TestPyPI로 배포한다.

TestPyPI 설치 확인:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  ranksmith==0.1.1rc1
```

## Release 절차
stable version을 사용한다.

1. `pyproject.toml`의 version을 stable version으로 올린다.
2. `./scripts/verify.sh`를 실행한다.
3. `UV_CACHE_DIR=.uv-cache uv tool run twine check dist/*`를 실행한다.
4. 변경사항을 commit하고 `main`에 push한다.
5. `v<version>` 형식의 tag를 push한다.

```bash
git tag v0.1.1
git push origin v0.1.1
```

tag에 `a`, `b`, `rc`, `dev`가 없으면 `.github/workflows/ci.yml`의
`publish-pypi` job이 다음을 수행한다.

1. test matrix 통과 대기
2. `uv build`
3. `uv tool run twine check dist/*`
4. `pypa/gh-action-pypi-publish@release/v1`로 PyPI 업로드

## 주의
- GitHub Actions의 publish job은 `id-token: write` 권한이 필요하다.
- PyPI에 등록한 workflow 이름과 실제 workflow 파일명(`ci.yml`)이 일치해야 한다.
- tag 이름은 `v*` 패턴이어야 publish job이 실행된다.
- 같은 version은 PyPI/TestPyPI 각각에서 재업로드할 수 없다. 실패한 배포를 다시 시도하려면 version을 올린다.
