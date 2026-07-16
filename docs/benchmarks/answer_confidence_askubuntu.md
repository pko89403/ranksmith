# answer_confidence를 README 표준 벤치마크에 올리는 러닝북

이 문서는 `answer_confidence` reranker를 README의 표준 벤치마크 표
(AskUbuntuDupQuestions 361쿼리, Pyserini BM25 top-20, `@5` 평가,
`gpt-5.4-nano`)에서 **기존 메소드들과 같은 조건으로** 측정하는 절차를
기록합니다. 기존 7개 메소드의 결과는
`benchmark-results/live/askubuntu-bm25-top20-default-live.v3.merged.json`에
커밋되어 있으므로 다시 돌릴 필요가 없습니다. `answer_confidence`만 같은
candidate/qrels/모델로 돌린 뒤 병합합니다.

## 왜 별도 학습 데이터가 필요한가 (정직성 요구사항)

- `answer_confidence`는 학습된 scorer artifact가 전제조건입니다. 학습
  라벨은 "모델 답변이 gold answer와 일치했는가"인데, **AskUbuntu에는
  qrels만 있고 free-text gold answer가 없어 artifact를 AskUbuntu로 학습할
  수 없습니다.**
- 따라서 artifact는 QA 데이터(SQuAD v1.1)로 학습하며, 결과는 **도메인
  밖 artifact 측정**입니다. 표에 올릴 때 artifact 학습 도메인을 반드시
  함께 표기합니다(스펙
  `docs/specs/spec_confidence_aware_reranking.md`의 보고 규칙).
- 학습 데이터는 질문마다 gold 문맥 1개 + BM25 hard negative 문맥
  1개(기본)로 구성합니다. 추론 시 estimator는 관련/무관 후보 문서 양쪽에서
  나온 답변을 채점하므로, 학습 분포도 같아야 합니다.

## 전제조건

- `.benchmark-cache/askubuntu-bm25/` 캐시(corpus/queries/qrels). 리포에는
  gitignore되어 없습니다. 이 캐시는 MTEB `AskUbuntuDupQuestions`에서
  export한 upstream artifact이며, README 벤치마크를 만든 환경에 있습니다.
- 후보 run은 리포에 커밋되어 있습니다:
  `benchmark-results/pyserini/askubuntu-bm25-top20.trec`.
- 모델 접근: 기존 표와 같은 `gpt-5.4-nano` Azure 배포(.env), 또는 로컬
  스모크용 OpenAI 호환 endpoint(`RANKSMITH_OPENAI_BASE_URL`).
- `confidence-train` extra 설치 (torch/transformers/lightgbm/scikit-learn).

## 1단계 — 학습 데이터 빌드 (LLM 호출 없음, 1회)

```bash
uv run python scripts/build_answer_confidence_training_data.py \
  --download \
  --output .benchmark-cache/answer-confidence/answer_train.jsonl
```

- 기본값: 질문 250개 × (gold 1 + hard negative 1) = 500행. 스펙 기록상
  100행은 과적합(랜덤 이하), 500행은 유효했습니다.
- 결정적(seed 13)이며 `answer_train.report.json`에 통계와 SQuAD 파일
  sha256이 남습니다.

## 2단계 — artifact 학습 (LLM 호출: 행 수만큼, 기본 500회)

**벤치마크와 같은 배포로 generation하세요.** estimator는 추론 시점에 그
모델이 만든 답변을 채점하므로, 학습 답변도 같은 모델이어야 공정합니다.

```bash
# Azure(gpt-5.4-nano)는 .env에서 읽습니다. 로컬 스모크는
# RANKSMITH_OPENAI_BASE_URL(+_MODEL)로 대체할 수 있습니다.
uv run python scripts/train_answer_confidence.py \
  --input .benchmark-cache/answer-confidence/answer_train.jsonl \
  --workdir .benchmark-cache/answer-confidence/work \
  --artifact .benchmark-cache/answer-confidence/answer_confidence.joblib \
  --allow-truncation \
  --allow-live
```

- 끝에 test split 지표가 출력되고, **test roc_auc < 0.6이면 스크립트가
  실패합니다.** 그 artifact로는 벤치마크를 돌리지 마세요(행 수를 늘리거나
  `generated.jsonl` 라벨을 점검).
- artifact는 커밋하지 않습니다(`.gitignore` 정책 유지).

## 3단계 — answer_confidence 단독 run (LLM 호출: 361 × 20 = 7,220회)

기존 optional method 관행대로 별도 run으로 실행합니다.

```bash
uv run python scripts/compare_reranking.py \
  --dataset benchmark-cache \
  --dataset-name askubuntu-bm25 \
  --cache-dir .benchmark-cache/askubuntu-bm25 \
  --split test \
  --candidates benchmark-results/pyserini/askubuntu-bm25-top20.trec \
  --candidate-count 20 \
  --algorithm answer_confidence \
  --answer-confidence-artifact .benchmark-cache/answer-confidence/answer_confidence.joblib \
  --top-k 5 \
  --window-size 20 \
  --stride 10 \
  --output benchmark-results/askubuntu-bm25-top20-answer-confidence.json \
  --allow-live
```

- 플래그(`--dataset-name`, `--candidate-count`, `--top-k`)는 커밋된
  v3 결과와 정확히 같아야 4단계 병합 검증을 통과합니다.
- `--algorithm`은 이제 반복 지정으로 여러 메소드를 한 번에 같은 케이스에서
  돌릴 수도 있습니다(예: `--algorithm original_bm25 --algorithm
  answer_confidence` 재검증 run).

## 4단계 — 커밋된 기존 결과와 병합

```bash
uv run python scripts/merge_benchmark_reports.py \
  benchmark-results/live/askubuntu-bm25-top20-default-live.v3.merged.json \
  benchmark-results/askubuntu-bm25-top20-answer-confidence.json \
  --output benchmark-results/live/askubuntu-bm25-top20-default-live.v4.merged.json
```

병합 스크립트는 벤치마크 정체성(dataset, candidate 파일명, candidate 수,
case 수, top_k)과 **알고리즘별 query_id 집합의 완전 일치**를 검증하고,
동일하지 않으면 거부합니다. 수동 병합으로 생기던 조건 불일치 사고를
막기 위한 것입니다.

## 보고 규칙

- README 표에 올릴 때 method 이름에 artifact 도메인을 병기합니다. 예:
  `answer_confidence` *(scorer: SQuAD v1.1, 도메인 밖)*.
- Nominal LLM calls/query = 20 (후보당 answer 1회). 여기에 더해 로컬
  encoder+scorer 추론이 문서당 1회 있습니다(LLM 아님).
- artifact 학습 generation 비용(행당 1회)은 reranking 호출과 분리해서
  보고합니다.
- 스모크/부분 run(예: `--max-cases`)은 품질 수치로 보고하지 않습니다
  (`docs/benchmarks/bm25_top20_reranking.md` 규칙 동일).

## 이 러닝북이 검증된 범위

전 단계(빌드 → generation → 학습 → artifact → 단독 run → 다중
알고리즘 run → 병합)는 실제 SQuAD 다운로드 + 결정적 fake
OpenAI-호환 서버 + 로컬 소형 encoder로 기계적으로 관통 확인됐습니다
(2026-07-16, 원격 개발 환경). LLM 품질에 좌우되는 수치는 이 러닝북을
실행하는 환경에서 나옵니다.
