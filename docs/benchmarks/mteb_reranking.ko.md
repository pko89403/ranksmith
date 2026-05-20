# MTEB Reranking 벤치마크

이 문서는 README benchmark 표의 근거를 기록합니다.

이 benchmark는 reranking만 측정합니다. first-stage retrieval 결과가 아니라
MTEB가 제공하는 고정 후보 집합을 사용합니다.

## 실행 명령

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

실패 row는 이후 `--resume --retry-failed-results`로 재시도했습니다.

## 측정 범위

- Dataset: `AskUbuntuDupQuestions`, `test` split
- Queries: `361`
- Candidates: MTEB가 제공하는 `top_ranked` 후보, query당 `20`개
- Candidate order: seed `13`으로 shuffled
- Model: Azure OpenAI deployment `gpt-5.4-nano`
- Validation: strict JSON, invalid output은 zero-scored
- Artifact: `benchmark-results/mteb-reranking/askubuntu-full-tourrank-prp-20260520`
- Evidence files: `metadata.json`, `overall_summary.json`, `task_summary.json`, `result_tables.md`

## 결과

| Method | NDCG@10 | MRR@10 | MAP | Recall@10 | p50 latency | Invalid rate | LLM calls/query | Total calls | Queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | 0.3926 | 0.4594 | 0.3711 | 0.4993 | 0.0 ms | 0.000 | 0.0 | 0 | 361 |
| `rankgpt_sliding_window@20` | 0.6908 | 0.7470 | 0.6355 | 0.7671 | 1820.5 ms | 0.008 | 1.0 | 374 | 361 |
| `tourrank_r@20:r2` | 0.7023 | 0.7642 | 0.6421 | 0.7785 | 8297.1 ms | 0.000 | 8.0 | 2,888 | 361 |
| `tourrank_r@20:r10` | 0.7135 | 0.7734 | 0.6597 | 0.7836 | 39026.4 ms | 0.006 | 39.9 | 14,409 | 361 |

이 run에서는 `tourrank_r@20:r10`이 가장 높은 점수를 냈습니다.
`tourrank_r@20:r2`는 더 적은 호출과 낮은 latency로 근접한 결과를 냈습니다.

기본 `passes=10`의 full `prp_sliding_k@20`은 이 full-query benchmark에서
실행하지 않았습니다. query당 `380`회, 전체 `137,180`회 호출이 필요하므로 이
설정의 품질/latency metric은 보고하지 않습니다.

보조 측정인 `prp_sliding_k@20:p1`은 `tourrank_r@20:r10`과 비슷한 호출 예산을
보기 위해 같은 361개 query에서만 실행했습니다: NDCG@10 `0.5360`, MRR@10
`0.7261`, MAP `0.4983`, Recall@10 `0.5773`, p50 latency `19919.1 ms`,
invalid rate `0.000`, query당 `38.0`회, 전체 `13,718`회 호출.

## 호출 수 산정

`evaluate_mteb_reranking.py`는 실제 측정된 LLM 호출 수를 기록합니다. 더 넓은
비교 runner인 `scripts/compare_reranking.py`는 live reranking 호출 수를 실행
전에 추정해 출력합니다.

- `rankgpt_sliding_window`: RankGPT back-to-front window마다 LLM 1회 호출
- `prp_sliding_k`: query마다 `2 * passes * max(document_count - 1, 0)` pairwise LLM 호출
- `tourrank_r`: query마다 `tourrank_rounds * sum(stage.group_count)` selection LLM 호출

후보가 정확히 100개이면 논문 top-100 TourRank stage를 사용합니다. 그 외 후보
수에는 명시적인 single-group halving stage plan을 사용합니다. 논문 top-100
stage 기준 TourRank-2는 query당 26회, TourRank-10은 query당 130회 호출합니다.

runner는 first-stage candidate, embedding, community를 생성하지 않습니다.
candidate generation에 embedding이나 community-building LLM 호출을 사용했다면,
그 비용은 reranking 호출과 분리해서 기록해야 합니다.

## BEIR/SciFact 비교 Runner

`ranksmith`에는 qrels 기반 비교 runner도 포함되어 있습니다. BEIR mode에서는
qrels만으로는 유효한 reranking benchmark가 아니므로 first-stage candidate TSV가
필요합니다.

예상 cache layout:

```text
.benchmark-cache/scifact/
  corpus.jsonl
  queries.jsonl
  qrels/test.tsv
```

Candidate TSV row는 `query_id`, `document_id`로 시작해야 합니다.

```text
query_id    document_id    rank
```

예시:

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

raw benchmark artifact는 명시적으로 검토한 경우가 아니라면 커밋하지 않습니다.
summary artifact와 run scope를 함께 공개합니다.

