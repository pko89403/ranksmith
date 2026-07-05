# Method selection guide

Source of truth: the repository `README.md` ("Supported Strategies & Algorithms"
and "Benchmarking") and `docs/wiki/00_context.md` ("현재 기본값"). Keep this file
in sync with them; do not invent numbers.

## Quick decision

| If the user wants… | Recommend | Why |
| --- | --- | --- |
| The default, lowest-friction reranker | `ListwiseStrategy("rankgpt_sliding_window")` | Fewest calls. With `window_size >= N` it becomes one-shot listwise. |
| Best quality on a moderate budget | `TourRankStrategy(rounds=2)` | Best NDCG@5 and Recall@5 in the committed benchmark. |
| Quality-focused offline / paper-style | `TourRankStrategy(rounds=10)` | Highest built-in call cost; strongest listwise-style quality. |
| Pairwise preference / reproduce PRP / best MRR | `PairwiseStrategy(...)` | Best MRR@5 in the benchmark, but many calls. |
| Adaptive spend near the top-k boundary | `AcuRankStrategy(target_rank=k)` | TrueSkill-based; uses `metadata["score"]` as a first-stage prior when present. |
| Setwise top-k extraction with early stopping | `SetwiseStrategy(set_size=...)` | The only strategy with true `top_k` early stopping; lowest quality in the benchmark. |
| Deterministic business logic / a new research method | A custom Strategy class | You own the ranking contract. |
| High-throughput (FastAPI, etc.) | The `Async*` variant of any of the above | Same contracts, `await`ed. |

## Committed benchmark evidence

AskUbuntuDupQuestions test set, BM25 top-20 candidates, `@5`, 361 queries,
Azure deployment `gpt-5.4-nano`. Call counts are **nominal estimates**, not
exact provider-call telemetry.

| Method | NDCG@5 | MRR@5 | Recall@5 | Nominal LLM calls/query |
| --- | ---: | ---: | ---: | ---: |
| original_bm25 (no rerank) | 0.3520 | 0.5062 | 0.2862 | 0 |
| single_call_listwise@20 | 0.4082 | 0.5541 | 0.3345 | 1 |
| rankgpt_sw_w5 | 0.3973 | 0.5283 | 0.3366 | 9 |
| acurank_k5_b1 | 0.4053 | 0.5491 | 0.3377 | 2 |
| tourrank_r2 | 0.4236 | 0.5725 | 0.3601 | 8 |
| setwise_hs_s10 | 0.3653 | 0.5059 | 0.3005 | 12 |
| prp_sliding_p1 | 0.4065 | 0.5818 | 0.3277 | 38 |

`tourrank_r2` led NDCG@5 and Recall@5; `prp_sliding_p1` led MRR@5.

## Parameters & defaults

### ListwiseStrategy / AsyncListwiseStrategy — RankGPT
- `window_size=20`,
  `stride=10`, `max_document_chars=4000`.
- Cost shape: roughly the number of windows; `window_size >= N` → 1 call.
- Model op: `rank()` (a full listwise permutation).

### PairwiseStrategy / AsyncPairwiseStrategy — PRP
- `passes=10` (expensive), `max_document_chars=4000`.
- Async compares both pair orders concurrently.
- Cost shape: about `passes * (N-1) * 2` compare calls; lower `passes` to cut cost.
- Model op: `compare()` (A/B winner); each adjacent pair is compared in both directions.

### SetwiseStrategy / AsyncSetwiseStrategy — Setwise heapsort
- `set_size=3` (minimum 3), `max_document_chars=4000`.
- The only strategy that truly early-stops on `top_k` (extracts just the needed top).
- Larger `set_size` → fewer calls but a harder selection prompt.
- Model op: `select()` (best of a set).

### TourRankStrategy / AsyncTourRankStrategy — TourRank-r
- `rounds=2`, `shuffle_seed=13`, configurable
  `stage_configs` (`TourRankStageConfig`), `max_document_chars=4000`.
- Deterministic seeded shuffling. Sync runs groups serially; async allows a
  semaphore-limited or unbounded value.
- Model op: `select()` per group; scores accumulate across rounds.

### AcuRankStrategy / AsyncAcuRankStrategy — AcuRank
- `target_rank=10`, `window_size=20`, `tolerance=0.01`
  (must satisfy `0 < tolerance < 0.5`), `uncertain_threshold=10`,
  `initial_pass=True`, score prior via `metadata["score"]`, async `batch_parallelism=1`,
  `max_adaptive_reranker_calls=None`.
- Uses TrueSkill ratings; reranks only uncertain candidates near the top-k
  boundary until convergence or budget. Align `target_rank` with your eval cutoff.
- First-stage prior: if every `Document` has a numeric `metadata["score"]`, it
  seeds the prior. All-or-none — partial or boolean scores fail fast.
- Model op: `rank()` over batches.

## top_k behavior
Only `SetwiseStrategy` stops early at `top_k`. Every other strategy computes the
full order and slices to `top_k` at the end. `top_k` must be `>= 0`.

## Sync vs async
Use `AzureOpenAIReranker` for scripts and batch jobs; `AsyncAzureOpenAIReranker`
for async servers. The strategy class must match the reranker (`TourRankStrategy`
with the sync reranker, `AsyncTourRankStrategy` with the async one, etc.).
