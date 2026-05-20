# MTEB Reranking Results

Strict validation: invalid LLM outputs receive zero scores.
Zero-score policy applies to main metrics; `valid-only` columns report metrics computed over the valid subset only.

## Overall

| Method | ndcg@10 | ndcg@10 (valid-only) | mrr@10 | map | recall@10 | p50_ms | p95_ms | invalid_rate | llm_calls/query | llm_calls_total | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 0.3926 | 0.3926 | 0.4594 | 0.3711 | 0.4993 | 0.0 | 0.0 | 0.000 | 0.0 | 0 | 361 |
| prp_sliding_k@20:p1 | 0.5360 | 0.5360 | 0.7261 | 0.4983 | 0.5773 | 19919.1 | 22276.7 | 0.000 | 38.0 | 13718 | 361 |
| rankgpt_sliding_window@20 | 0.6908 | 0.6966 | 0.7470 | 0.6355 | 0.7671 | 1820.5 | 2309.3 | 0.008 | 1.0 | 374 | 361 |
| tourrank_r@20:r10 | 0.7135 | 0.7175 | 0.7734 | 0.6597 | 0.7836 | 39026.4 | 43152.7 | 0.006 | 39.9 | 14409 | 361 |
| tourrank_r@20:r2 | 0.7023 | 0.7023 | 0.7642 | 0.6421 | 0.7785 | 8297.1 | 9798.3 | 0.000 | 8.0 | 2888 | 361 |

## Task: AskUbuntuDupQuestions

| Method | ndcg@10 | ndcg@10 (valid-only) | mrr@10 | map | recall@10 | p50_ms | p95_ms | invalid_rate | llm_calls/query | llm_calls_total | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 0.3926 | 0.3926 | 0.4594 | 0.3711 | 0.4993 | 0.0 | 0.0 | 0.000 | 0.0 | 0 | 361 |
| prp_sliding_k@20:p1 | 0.5360 | 0.5360 | 0.7261 | 0.4983 | 0.5773 | 19919.1 | 22276.7 | 0.000 | 38.0 | 13718 | 361 |
| rankgpt_sliding_window@20 | 0.6908 | 0.6966 | 0.7470 | 0.6355 | 0.7671 | 1820.5 | 2309.3 | 0.008 | 1.0 | 374 | 361 |
| tourrank_r@20:r10 | 0.7135 | 0.7175 | 0.7734 | 0.6597 | 0.7836 | 39026.4 | 43152.7 | 0.006 | 39.9 | 14409 | 361 |
| tourrank_r@20:r2 | 0.7023 | 0.7023 | 0.7642 | 0.6421 | 0.7785 | 8297.1 | 9798.3 | 0.000 | 8.0 | 2888 | 361 |
