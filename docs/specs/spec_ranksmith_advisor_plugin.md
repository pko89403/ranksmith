# Spec: ranksmith Advisor (Claude Code 플러그인/스킬)

> **작성 가이드**: 이 문서는 코딩 어시스턴트의 작업 추적용이기도 하지만, **최우선적으로 사람(개발자)이 읽고 이해하기 가장 좋은 형태(가독성)**여야 합니다.

## 1. 개요 (Overview)
- **작업 목적**: 코딩 어시스턴트(Claude Code)가 `ranksmith` 사용자에게 ① **적절한 리랭킹 메소드 선택 가이드**와 ② **동작하는 코드 스니펫**을 정확하게 제공하도록, ranksmith 도메인 지식을 Claude Code 스킬/플러그인으로 레포에 내장한다.
- **핵심 가치**: 일반 LLM이 이 라이브러리에서 *반복적으로 틀리는* 안티패턴(미구현 provider 호출, `algorithm` 문자열 확장, confidence를 리랭커로 취급 등)을 **가드레일**로 차단한다. 추천 로직은 추론이 아니라 README 표 + 커밋된 벤치마크 근거로 인코딩한다.
- **Reference**:
  - Claude Code Skills: https://code.claude.com/docs/en/skills.md
  - Claude Code Plugins: https://code.claude.com/docs/en/plugins.md
  - Plugins Reference(매니페스트/검증): https://code.claude.com/docs/en/plugins-reference.md
  - 근거 문서(레포 내부): `README.md`(Recommended Use Cases·Benchmark), `docs/wiki/00_context.md`, `docs/wiki/02_architecture.md`, `docs/wiki/08_custom_strategy_extension.md`, `AGENTS.md`
- **상태**: `[ ] Draft` | `[x] In Progress` | `[ ] Completed`

## 2. 요구 사항 및 제약 (Requirements & Constraints)
- **입력 (Inputs)**: 사용자의 자연어 제약조건 — 후보 수, 비용/지연 예산, 품질 목표, 동기/비동기, top_k, first-stage score 보유 여부, 커스텀 로직 필요 여부.
- **출력 (Outputs)**: ① 전략+파라미터 추천(근거·콜 비용 포함), ② 검증된 `examples/`에 기반한 코드 스니펫, ③ 가드레일 준수.
- **스코프 (v1)**: **리랭킹 전략만**. confidence(`ranksmith.confidence` 등)는 "리랭킹 Strategy가 아님 + 별도 유틸"이라는 **짧은 포인터**로만 다룬다. confidence 사용/학습 가이드는 v2 후보.
- **배포 (단계적)**:
  - **1단계 — 프로젝트 로컬**: `.claude/skills/ranksmith-advisor/`. 이 레포에서 작업하는 기여자에게 설치 0으로 즉시 동작.
  - **2단계 — 배포형 플러그인**: 동일 스킬을 `.claude-plugin/plugin.json` + 루트 `skills/`로 노출하고 `.claude-plugin/marketplace.json`을 추가해, pip로 ranksmith를 설치한 외부 개발자가 자기 프로젝트에 설치 가능하게 한다.
- **제약 사항 (Constraints)**:
  - **드리프트 금지**: 스니펫 코드를 복제하지 않고, CI에서 실행·검증되는 `examples/*.py`(`tests/test_examples.py`)를 source of truth로 가리킨다.
  - **증거 정책**: AGENTS.md "README/Benchmark Evidence Policy"를 그대로 반영한다. 실행하지 않은 method의 metric을 만들지 않고, 콜 수는 estimate로 명시한다.
  - **fast-fail 철학 노출**: 어시스턴트가 조용한 truncation/ranking 보정 코드를 생성하지 않도록 명시한다.
  - **작은 표면적**: v1은 단일 스킬 `ranksmith-advisor` + 보조 `.md` 3개. 선택기/생성기 분리는 하지 않는다.
  - **패키지 산출물 비포함**: 본 작업은 `src/ranksmith/` public API를 변경하지 않는다. `pyproject.toml`의 wheel `exclude`에 영향이 없도록 레포 루트 도구 디렉터리에만 추가한다(빌드 산출물에 포함되지 않음).

## 3. 상세 설계 (Architecture & Design)

### 파일 구조
```text
# 단일 소스: 플러그인 루트 = 레포 루트
skills/ranksmith-advisor/              # 스킬 본문(유일 사본)
  SKILL.md                             # 라우팅 + 가드레일 요약 (상시 로드)
  method-guide.md                      # 전략 표 · 콜 비용 모델 · 벤치마크 근거 · 파라미터
  snippets.md                          # 시나리오 → examples/ 매핑 + 최소 스니펫
  guardrails.md                        # 하드룰(안티패턴)
.claude-plugin/
  plugin.json                          # 플러그인 매니페스트 (plugin root = repo root)
  marketplace.json                     # 자체 marketplace (plugin source "./")
.claude/settings.json                  # (예정) 기여자 자동 등록 — 라이브 세션 확인 후
tests/test_advisor_references.py       # 자동 참조 무결성 테스트
pyproject.toml                         # exclude += /.claude /.claude-plugin /skills (sdist 격리)
```
> **단일 소스 확정**: 스킬 본문은 `skills/ranksmith-advisor/` 한 곳에만 둔다. 레포 전체가 곧 플러그인(plugin root = repo root)이며, 자체 `marketplace.json`이 `source: "./"`로 이 플러그인을 가리킨다. 외부 사용자: `/plugin marketplace add pko89403/ranksmith` → `/plugin install ranksmith-advisor@ranksmith`. 기여자 자동 등록(`.claude/settings.json`의 `extraKnownMarketplaces`)은 라이브 Claude Code 세션에서 스키마 확인 후 추가하며, 그 전까지는 수동 `/plugin marketplace add .`로 대체한다.

### SKILL.md frontmatter (초안)
```yaml
---
name: ranksmith-advisor
description: >
  Guide ranksmith users to the right reranking strategy and generate correct,
  idiomatic code. Use when the user asks how to rerank with ranksmith, which
  strategy to pick (RankGPT/listwise, PRP/pairwise, Setwise, TourRank, AcuRank),
  how to configure window_size/passes/rounds/target_rank, async usage, custom
  strategies, or how to wire AzureOpenAIReranker / ModelClient / ModelProvider.
---
```
- `description`이 트리거 품질을 좌우하므로 핵심 키워드(ranksmith, rerank, 전략명, 파라미터명, AzureOpenAIReranker)를 앞에 배치한다.
- `/ranksmith-advisor`로 사용자 직접 호출도 가능하게 둔다(별도 `disable-model-invocation` 미설정).

### method-guide.md — 결정 절차 (근거 기반)
README "Recommended Use Cases" + Benchmark 표를 절차로 인코딩한다.

```text
비용 우선 / 기본값        → ListwiseStrategy("rankgpt_sliding_window")   window>=N 이면 single-call
품질 우선 / 예산 보통     → TourRankStrategy(rounds=2)   ★벤치 NDCG@5·Recall@5 최고
오프라인 최고 품질        → TourRankStrategy(rounds=10)
MRR 중시 / PRP 재현       → PairwiseStrategy             ★벤치 MRR@5 최고 (단 호출 多)
적응형 / top-k 경계 집중  → AcuRankStrategy(target_rank=k)  TrueSkill, metadata["score"] 프라이어
setwise top-k 추출        → SetwiseStrategy(set_size)    top-k 조기종료 유일 (벤치 품질 최저)
결정적 비즈니스 로직      → 커스텀 Strategy 클래스
고처리량(FastAPI)         → Async* 변형
```

콜 비용(AskUbuntu BM25 top-20, **estimate**, README 근거):

| Method | NDCG@5 | MRR@5 | Recall@5 | Nominal calls/query |
| --- | ---: | ---: | ---: | ---: |
| original_bm25 | 0.3520 | 0.5062 | 0.2862 | 0 |
| single_call_listwise@20 | 0.4082 | 0.5541 | 0.3345 | 1 |
| rankgpt_sw_w5 | 0.3973 | 0.5283 | 0.3366 | 9 |
| acurank_k5_b1 | 0.4053 | 0.5491 | 0.3377 | 2 |
| tourrank_r2 | 0.4236 | 0.5725 | 0.3601 | 8 |
| setwise_hs_s10 | 0.3653 | 0.5059 | 0.3005 | 12 |
| prp_sliding_p1 | 0.4065 | 0.5818 | 0.3277 | 38 |

+ 파라미터 기본값/제약: `window_size=20`, `stride=10`, `max_document_chars=4000`, `passes=10`, `set_size>=3`, `rounds=2`, `target_rank=10`, `tolerance(0<x<0.5)` 등.

### snippets.md — 검증된 예제 매핑 (복제 금지)
| 시나리오 | 참조 예제(테스트됨) |
| --- | --- |
| 동기 listwise(RankGPT) | `examples/rankgpt_sync.py` |
| 비동기(FastAPI류) | `examples/rankgpt_async.py` |
| pairwise PRP | `examples/pairwise_prp.py` |
| setwise heapsort | `examples/setwise_heapsort.py` |
| TourRank-r | `examples/tourrank.py` |
| AcuRank + score 프라이어 | `examples/acurank.py` |
| 커스텀 Strategy 계약 | `examples/custom_strategy.py` |
"이 예제에서 X만 바꿔라" 식 적응 지침 + provider/자격증명 주입(`AzureOpenAIReranker` vs `ModelClient` 주입) 설명.

### guardrails.md — 하드룰
1. **Azure만 실제 동작**. `OpenAIProvider`/`AnthropicProvider`/`GeminiProvider`는 stub → 호출 시 `RerankProviderError`. 이 stub들로 동작 코드를 생성하지 않는다.
2. **새 알고리즘 = 새 Strategy 클래스**. `ListwiseStrategy(algorithm="...")`에 새 문자열을 넣지 않는다.
3. `rank`=1-based, `original_index`=입력 기준 0-based.
4. 커스텀 Strategy: keyword-only `rerank(*, query, documents, model_client, top_k=None)`; model JSON은 `parse_ranking_response`/`parse_selection_response`로 검증; provider 실패는 `RerankProviderError`로 분류.
5. **fast-fail**: 문서 truncation/ranking 보정 금지(`DocumentTooLongError`).
6. **confidence는 리랭킹 Strategy가 아님**. 별도 `ranksmith.confidence` 유틸, optional deps(`pip install ranksmith[confidence]`).
7. **벤치 수치 날조 금지**. 커밋된 artifact만 인용, 콜 수는 estimate로 명시.
8. `top_k` 조기종료는 Setwise뿐. 나머지는 사후 슬라이스.
9. **Public API만 사용**. private 모듈(`ranksmith._providers`, `ranksmith.strategies._*`) 직접 접근 금지.

### plugin.json (2단계 초안)
```json
{
  "name": "ranksmith-advisor",
  "description": "Guide ranksmith users to the right reranking strategy and generate correct code.",
  "version": "0.1.0",
  "author": { "name": "ranksmith contributors" },
  "repository": "https://github.com/pko89403/ranksmith",
  "license": "MIT"
}
```

## 4. 재사용 및 모듈화 (Reusability & Modularization)
- **Source of truth 재사용**: 스니펫은 `examples/`(이미 `tests/test_examples.py`로 CI 실행), 추천 수치는 README/Benchmark, 계약은 `docs/wiki/08_custom_strategy_extension.md`를 가리킨다. 별도 미검증 사본을 만들지 않는다.
- **단일 스킬**: 보조 `.md` 3개는 progressive disclosure로 필요 시에만 로드. 선택기/생성기 분리는 하지 않는다.

## 5. 운영 리스크 및 대응 (Risks & Handling)
- **스킬 미발동**: `description` 키워드 미스. → 전략명·파라미터명·진입 클래스명을 description 앞쪽에 배치하고, `/ranksmith-advisor` 직접 호출 경로를 함께 제공.
- **숫자/기본값 드리프트**: 라이브러리 변경 시 method-guide.md가 낡음. → 수치 중복 최소화 + 출처 링크. 신규 Strategy 추가 시 **AGENTS.md 7단계 파이프라인 체크리스트에 "advisor 갱신" 항목 추가**(별도 PR로 제안).
- **예제 경로 변경**: snippets.md가 가리키는 `examples/*.py` 경로가 바뀌면 깨짐. → 테스트 계획의 "참조 무결성 체크"로 방어.
- **배포 산출물 오염**: 도구 파일이 wheel에 포함되지 않도록 루트 도구 디렉터리(`.claude/`, `.claude-plugin/`, `skills/`)에만 추가하고 빌드 영향 없음을 확인.

## 6. 테스트 계획 (Test Plan)
> 본 작업은 reranking 알고리즘이 아니므로 TEMPLATE의 "공통 Reranking Smoke/Benchmark"는 비적용. 문서/매니페스트 무결성 중심으로 검증한다.

- **성공 케이스**:
  - `skills/ranksmith-advisor/SKILL.md`가 유효한 frontmatter로 로드된다.
  - (2단계) `plugin.json`/`marketplace.json`이 스키마상 유효하다(가능 시 `claude plugin validate` 수준 확인).
  - snippets.md가 가리키는 모든 `examples/*.py` 경로가 실제 존재한다(참조 무결성).
  - method-guide.md의 수치가 README 표와 일치한다.
- **엣지/실패 케이스**:
  - 존재하지 않는 전략/파라미터를 추천하지 않는지(가드레일과 충돌 없음) 수기 검토.
  - guardrails.md 항목이 현재 코드(`_stubs.py`, `parsing.py`, `errors.py`, 전략 default)와 사실 일치하는지 대조.
- **검증 명령**: `./scripts/verify.sh`(기존 라이브러리 회귀 없음 확인) + 참조 무결성/일치 체크(스크립트화 여부는 구현 시 결정).

---

## 7. 작업 태스크 추적 (Task Checklist)
> **코딩 어시스턴트 필수 지침**: 완료 작업은 `[x]`로 표시하고 필요 시 하위 태스크를 추가한다.

### Phase 1: 컨텍스트 및 설계 확인
- [x] 기존 코드베이스·Wiki·README·examples·AGENTS.md 확인
- [x] Claude Code Skills/Plugins 작성 규격 확인
- [x] 본 스펙(배포 단계·스코프·구조) 사용자 검토 및 최종 승인

### Phase 2: 구현 — 스킬 본문 (단일 소스)
- [x] `skills/ranksmith-advisor/SKILL.md` (라우팅 + 가드레일 요약)
- [x] `skills/ranksmith-advisor/method-guide.md` (결정 절차·콜 비용·파라미터)
- [x] `skills/ranksmith-advisor/snippets.md` (examples 매핑 + 최소 스니펫)
- [x] `skills/ranksmith-advisor/guardrails.md` (하드룰)

### Phase 3: 구현 — 배포 패키징
- [x] `.claude-plugin/plugin.json` (plugin root = repo root)
- [x] `.claude-plugin/marketplace.json` (source `"./"`)
- [x] `pyproject.toml` exclude += `/.claude` `/.claude-plugin` `/skills` (sdist 격리)
- [ ] `.claude/settings.json` 기여자 자동 등록 — 라이브 세션 스키마 확인 후

### Phase 4: 검증 (Verification)
- [x] `tests/test_advisor_references.py`: snippets ↔ `examples/*.py` 참조 무결성 + public symbol 검증
- [x] guardrails.md ↔ 현재 코드(`_stubs.py`/`parsing.py`/`errors.py`/전략 default) 사실 대조
- [ ] `./scripts/verify.sh` 통과(라이브러리 회귀 없음)
- [ ] 실제 Claude Code 세션에서 추천/스니펫 트리거 + 매니페스트 로드 수기 검증

### Phase 5: 완료 및 정리
- [ ] 필요 시 `README`/`docs/wiki/`에 advisor 안내 추가
- [ ] 본 문서 최상단 **상태**를 `Completed`로 변경
