# MTEB Reranking Benchmark

This page records the evidence behind the README benchmark table.

The benchmark measures reranking only. It uses fixed native MTEB candidate sets,
not first-stage retrieval results.

## Run Command

```bash
UV_NATIVE_TLS=true uv run python scripts/evaluate_mteb_reranking.py \
  --tasks AskUbuntuDupQuestions \
  --methods \
    original \
    rankgpt_sliding_window@20 \
    prp_sliding_k@20:p1 \
    tourrank_r@20:r2 \
    tourrank_r@20:r10 \
  --output-dir benchmark-results/mteb-reranking/askubuntu-full-tourrank-prp-20260520 \
  --max-document-chars 4000 \
  --shuffle-candidates --shuffle-seed 13 \
  --rankgpt-window-size 20 --rankgpt-step 10 \
  --concurrency 4 \
  --retry-invalid-outputs 3 \
  --input-token-price-per-1m 2.50 \
  --output-token-price-per-1m 10.00 \
  --allow-live
```

Failed rows were later retried with `--resume --retry-failed-results`.

## Run Scope

- Dataset: `AskUbuntuDupQuestions`, `test` split
- Queries: `361`
- Candidates: MTEB-provided `top_ranked` candidates, `20` per query
- Candidate order: shuffled with seed `13`
- Model: Azure OpenAI deployment `gpt-5.4-nano`
- Validation: strict JSON; invalid outputs are zero-scored
- Artifact: `benchmark-results/mteb-reranking/askubuntu-full-tourrank-prp-20260520`
- Evidence files: `metadata.json`, `overall_summary.json`, `task_summary.json`, `result_tables.md`

## Results

| Method | NDCG@10 | MRR@10 | MAP | Recall@10 | p50 latency | Invalid rate | LLM calls/query | Total calls | Queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | 0.3926 | 0.4594 | 0.3711 | 0.4993 | 0.0 ms | 0.000 | 0.0 | 0 | 361 |
| `rankgpt_sliding_window@20` | 0.6908 | 0.7470 | 0.6355 | 0.7671 | 1820.5 ms | 0.008 | 1.0 | 374 | 361 |
| `tourrank_r@20:r2` | 0.7023 | 0.7642 | 0.6421 | 0.7785 | 8297.1 ms | 0.000 | 8.0 | 2,888 | 361 |
| `tourrank_r@20:r10` | 0.7135 | 0.7734 | 0.6597 | 0.7836 | 39026.4 ms | 0.006 | 39.9 | 14,409 | 361 |

`tourrank_r@20:r10` had the strongest scores in this run, while
`tourrank_r@20:r2` stayed close with far fewer calls and lower latency.

Full `prp_sliding_k@20` with the default `passes=10` was not run in this
full-query benchmark. It would require `380` calls/query, or `137,180` calls
over all 361 queries, so no quality or latency metrics are reported for that
setting.

The auxiliary `prp_sliding_k@20:p1` run completed over the same 361 queries only
as a call-budget reference near `tourrank_r@20:r10`: NDCG@10 `0.5360`, MRR@10
`0.7261`, MAP `0.4983`, Recall@10 `0.5773`, p50 latency `19919.1 ms`,
invalid rate `0.000`, `38.0` calls/query, `13,718` total calls.

## Call Accounting

`evaluate_mteb_reranking.py` records measured LLM calls. The broader comparison
runner, `scripts/compare_reranking.py`, estimates live reranking calls before
execution.

- `rankgpt_sliding_window`: one LLM call per back-to-front RankGPT window.
- `prp_sliding_k`: `2 * passes * max(document_count - 1, 0)` pairwise LLM calls per query.
- `tourrank_r`: `tourrank_rounds * sum(stage.group_count)` selection LLM calls per query.

For exactly 100 candidates, the runner uses the paper top-100 TourRank stages.
For other candidate counts, it uses an explicit single-group halving stage plan.
With the paper top-100 stages, TourRank-2 uses 26 calls/query and TourRank-10
uses 130 calls/query.

The runner does not create first-stage candidates, embeddings, or communities.
If candidate generation uses embeddings or community-building LLM calls, report
those costs separately from reranking calls.

## BEIR/SciFact Comparison Runner

`ranksmith` also includes a qrels-backed comparison runner. BEIR mode requires a
first-stage candidate TSV, because qrels alone are not a valid reranking
benchmark.

Expected cache layout:

```text
.benchmark-cache/scifact/
  corpus.jsonl
  queries.jsonl
  qrels/test.tsv
```

Candidate TSV rows must start with `query_id` and `document_id`:

```text
query_id    document_id    rank
```

Example:

```bash
python scripts/compare_reranking.py \
  --dataset beir-scifact \
  --cache-dir .benchmark-cache/scifact \
  --split test \
  --candidates path/to/candidates.tsv \
  --algorithm all \
  --top-k 10 \
  --window-size 20 \
  --stride 10 \
  --output benchmark-results/scifact.json \
  --allow-live
```

Raw benchmark artifacts should not be committed unless explicitly reviewed.
Publish summary artifacts and state the run scope.

