# Guardrails (hard rules)

Apply these to every recommendation and every line of code you generate. They
encode ranksmith's contracts. The canonical statements live in the repository;
this file points to them instead of re-deriving them, so fix the sources, not a
second copy.

1. **Azure is the only bundled provider.** Generate Azure-backed code
   (`AzureOpenAIReranker`, `AzureAOAIProvider`) or a user-supplied
   `ModelClient` / custom `ModelProvider` for other vendors.
   Source: `src/ranksmith/providers/`, README "Model Provider Architecture".

2. **A new ranking method is a new Strategy class**, never a patch on the
   built-in strategy classes.
   Source: `docs/wiki/08_custom_strategy_extension.md`, `docs/wiki/01_decisions.md` (D006).

3. **Indexing contract.** `rank` is 1-based (display); `original_index` is
   0-based against the input list, even after reordering.
   Source: `docs/wiki/00_context.md` "현재 기본값", README "Result Model".

4. **Custom strategy contract.** Keyword-only
   `rerank(*, query, documents, model_client, top_k=None) -> list[RerankResult]`.
   Validate model JSON with `parse_ranking_response()` /
   `parse_selection_response()`. Classify provider failures inside the strategy
   as `RerankProviderError`.
   Source: `docs/wiki/08_custom_strategy_extension.md`, `examples/custom_strategy.py`.

5. **Fail fast.** Never silently truncate a long document (raise / expect
   `DocumentTooLongError`) or repair an invalid ranking. JSON `true` / `false`
   are not integers in a ranking.
   Source: `src/ranksmith/parsing.py`, `src/ranksmith/errors.py`, AGENTS.md "프로젝트 원칙".

6. **Confidence is not reranking.** `ranksmith.confidence`,
   `ranksmith.confidence_generation`, and `ranksmith.confidence_training` are
   optional utilities (`pip install "ranksmith[confidence]"`), not a Strategy or
   Algorithm. Do not present them as a reranker.
   Source: `docs/wiki/02_architecture.md` "Confidence".

7. **Evidence only.** Quote metrics and call counts from the committed README /
   benchmark artifacts, label call counts as estimates, and never fabricate
   numbers for an unmeasured method.
   Source: AGENTS.md "README / Benchmark Evidence Policy".

8. **`top_k` early stopping is Setwise-only.** Other strategies slice to `top_k`
   after computing the full order.
   Source: `src/ranksmith/strategies/setwise.py`.

9. **Public API only.** Use root `ranksmith` exports in user code; do not reach
   into private modules (`ranksmith.providers.azure`, `ranksmith.strategies.acurank`, …).
   Source: `docs/wiki/02_architecture.md`.
