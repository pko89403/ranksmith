# ranksmith 작업 컨텍스트

## 목표
LLM 기반 reranking을 위한 작고 신뢰성 있는 Python 패키지를 만든다.

## 현재 범위
- Azure OpenAI provider만 지원한다.
- Zero-shot listwise reranking만 지원한다.
- `direct`, `sliding_window`, `rankgpt_sliding_window` algorithm만 지원한다.
- indexing은 하지 않는다.
- vector search는 하지 않는다.
- LangChain/LlamaIndex adapter는 아직 만들지 않는다.

## 절대 규칙
- Fast fail.
- 문서를 조용히 자르지 않는다.
- 잘못된 ranking을 조용히 보정하지 않는다.
- 부족한 reference를 근거로 algorithm 세부사항을 추론하지 않는다.
- 사용자 승인 없이 public API를 확장하지 않는다.

## Public API
- `AzureOpenAIReranker`
- `Document`
- `RerankResult`
- `ListwiseStrategy`
- `RerankError`
- `RerankInputError`
- `RerankParseError`
- `RerankProviderError`
- `DocumentTooLongError`

## 현재 기본값
- Python `3.10+`
- Package/import name: `ranksmith`
- Build: `uv + hatchling`
- License: MIT
- `rank`: 1-based
- `original_index`: 0-based
- `ListwiseStrategy.algorithm`: `sliding_window`
- `window_size`: `20`
- `stride`: `10`
- `max_document_chars`: `4000`

## Codex 읽기 순서
1. `docs/wiki/00_context.md`
2. `docs/wiki/01_decisions.md`
3. `docs/wiki/02_architecture.md`
4. `docs/wiki/03_reference_processing.md`
5. `docs/wiki/04_references_index.md`
6. `docs/wiki/06_verification_policy.md`
7. 관련 reference summary
8. source code
