# BM25 Top-20 Reranking Benchmark

이 문서는 기존 benchmark dataset의 corpus/query/qrels를 유지하고, candidate generation만 Pyserini BM25로 수행하는 reranking 비교 기준을 기록합니다.

이 benchmark는 reranking만 측정합니다. Pyserini는 BM25 top-20 candidate run artifact를 만드는 upstream 도구이며, ranksmith는 그 후보 순서를 입력으로 받습니다.

## Candidate Input

Expected cache layout:

```text
.benchmark-cache/<dataset>/
  corpus.jsonl
  queries.jsonl
  qrels/test.tsv
```

Pyserini TREC run format:

```text
q1 Q0 d3 1 9.5 pyserini
q1 Q0 d2 2 8.0 pyserini
```

Simple TSV format도 지원합니다.

```text
query_id    document_id    rank    score
```

`score`가 있으면 `BenchmarkDocument.score`로 보존되고, AcuRank 실행 시 `Document.metadata["score"]` prior로 전달됩니다.

## Method Set

기본 비교는 BM25 top-20 후보를 입력으로 받고 `@5`로 평가합니다. top-k 조기 종료를 지원하는 method는 평가 대상 top-5까지만 추출할 수 있습니다.

| Method | 설정 | 비교에서의 의미 | Calls/query estimate |
| --- | --- | --- | ---: |
| `original_bm25` | BM25 top-20 그대로 | first-stage baseline | 0 |
| `single_call_listwise@20` | `N=20` | one-shot listwise baseline | 1 |
| `rankgpt_sw_w5` | `N=20`, `w=5`, `stride=2` | true sliding-window listwise baseline | 9 |
| `acurank_k5_b1` | `N=20`, `target_rank=5`, `w=20`, adaptive budget 1 | evaluate@5-aligned low-cost AcuRank baseline | 2 |
| `tourrank_r2` | `N=20`, `r=2` | moderate-cost setwise baseline | 8 |
| `setwise_hs_s10` | `N=20`, `set_size=10`, `top_k=5` | practical setwise heapsort baseline for long-context chat models | 12 |
| `prp_sliding_p1` | `N=20`, `passes=1` | pairwise quality/cost reference | 38 |

Optional methods:

| Method | 설정 | 비교에서의 의미 | Calls/query estimate |
| --- | --- | --- | ---: |
| `acurank_k5_b4` | `N=20`, `target_rank=5`, `w=20`, adaptive budget 4 | stronger evaluate@5-aligned AcuRank reference | 5 |
| `acurank_b1` | `N=20`, `target_rank=10`, `w=20`, adaptive budget 1 | legacy AcuRank top-10-boundary reference | 2 |
| `acurank_b4` | `N=20`, `target_rank=10`, `w=20`, adaptive budget 4 | legacy stronger AcuRank top-10-boundary reference | 5 |
| `tourrank_r10` | `N=20`, `r=10` | high-cost setwise reference | 40 |
| `prp_sliding_p3` | `N=20`, `passes=3` | stronger but expensive PRP | 114 |
| `answer_confidence` | `N=20`, 학습된 scorer artifact 필요 | confidence-change reranker (도메인 밖 artifact 라벨링 필수) | 20 |

`answer_confidence`는 학습된 artifact가 전제조건이며 절차와 보고 규칙이
별도 러닝북에 있습니다: `docs/benchmarks/answer_confidence_askubuntu.md`.

`prp_sliding_p1`은 asymptotic하게 O(n)이지만, 현재 `PairwiseStrategy`는 인접 pair마다 양방향 비교를 수행합니다. 그래서 20개 후보에서는 `2 * (20 - 1) = 38` calls/query로 추정합니다.

## Run Command

기본 method set:

```bash
UV_NATIVE_TLS=true uv run python scripts/compare_reranking.py \
  --dataset benchmark-cache \
  --dataset-name askubuntu-bm25 \
  --cache-dir .benchmark-cache/askubuntu-bm25 \
  --split test \
  --candidates benchmark-results/pyserini/askubuntu-bm25-top20.trec \
  --candidate-count 20 \
  --algorithm all \
  --top-k 5 \
  --window-size 20 \
  --stride 10 \
  --output benchmark-results/askubuntu-bm25-top20-reranking.json \
  --allow-live
```

Optional method는 별도 run으로 실행합니다.

```bash
UV_NATIVE_TLS=true uv run python scripts/compare_reranking.py \
  --dataset benchmark-cache \
  --dataset-name askubuntu-bm25 \
  --cache-dir .benchmark-cache/askubuntu-bm25 \
  --split test \
  --candidates benchmark-results/pyserini/askubuntu-bm25-top20.trec \
  --candidate-count 20 \
  --algorithm tourrank_r10 \
  --top-k 5 \
  --window-size 20 \
  --stride 10 \
  --output benchmark-results/askubuntu-bm25-top20-tourrank-r10.json \
  --allow-live
```

## Reporting Rules

- README benchmark 수치는 실제 summary artifact가 있을 때만 옮깁니다.
- `original_bm25`와 reranking method는 같은 BM25 top-20 candidate set에서 비교합니다.
- BM25 retrieval 품질과 reranking 품질을 같은 숫자로 섞지 않습니다.
- `call_estimates`는 실제 metric과 분리해서 estimate로 표기합니다.
- smoke/partial run은 품질 benchmark로 표현하지 않습니다.
