---
name: ranksmith-advisor
description: >-
  Advise users of the ranksmith Python library on LLM reranking. Use whenever
  the user asks how to rerank with ranksmith, which strategy or parameters to
  pick (RankGPT/listwise, PRP/pairwise, Setwise heapsort, TourRank, AcuRank),
  how to tune window_size/passes/rounds/target_rank, sync vs async, writing a
  custom strategy, or wiring AzureOpenAIReranker / ModelClient / ModelProvider.
---

# ranksmith Advisor

Help users of the `ranksmith` library (training-free LLM reranking over closed
APIs) in two ways: **recommend a method** and **generate correct code**. Keeping
ranksmith's contracts intact matters — those are the most common failures.

## How to use this skill

1. **"Which strategy / parameters should I use?"** → read `method-guide.md`, ask
   for the missing constraints (candidate count, cost/latency budget, quality
   target, sync vs async, `top_k`, first-stage scores), then recommend with the
   call-cost shape and the committed evidence.
2. **"Write or fix ranksmith code"** → read `snippets.md`, start from the
   matching tested example, adapt it, and keep every guardrail.
3. **Always** apply `guardrails.md`. These are hard rules, not suggestions.

## Non-negotiable guardrails (full list with sources in `guardrails.md`)

- Only the **Azure** provider actually runs. `OpenAIProvider`,
  `AnthropicProvider`, and `GeminiProvider` are reserved stubs that raise
  `RerankProviderError`. Never build working code on them.
- A new ranking method is a **new Strategy class**, never a new
  `ListwiseStrategy(algorithm="...")` string.
- `rank` is 1-based; `original_index` is 0-based against the input list.
- ranksmith **fails fast**: never silently truncate documents or repair an
  invalid ranking.
- `ranksmith.confidence` is **not** a reranking strategy — it is a separate
  optional utility.

## Evidence rule

Quote benchmark numbers and call counts only from `method-guide.md` (sourced
from the committed README and benchmark artifacts), and label call counts as
estimates. Never invent metrics for a method that was not measured.
