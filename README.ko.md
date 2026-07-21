# ranksmith

<p align="center">
  <img src="https://raw.githubusercontent.com/pko89403/ranksmith/main/assets/ranksmith-icon.png" alt="ranksmith icon" width="160">
</p>

후보 문서를 더 나은 순서로 벼리는 LLM reranking 패키지입니다.

[English README](https://github.com/pko89403/ranksmith/blob/main/README.md)

`ranksmith`는 LLM 기반 reranking을 위한 작은 Python 패키지입니다. 현재 패키지는
Azure OpenAI 기반 zero-shot candidate reranking에 집중합니다.

주요 특징:

- listwise RankGPT, pairwise PRP, tournament 방식 TourRank-r,
  uncertainty-aware AcuRank, confidence-gain built-in Strategy
- 커스텀 reranking 메소드를 위한 public Strategy contract
- vendor 독립 LLM 호출을 위한 `ModelClient` / `ModelProvider` 경계
- 엄격한 JSON parsing과 fast-fail 오류 정책
- sync/async Azure OpenAI reranker
- 근거 artifact가 커밋된 재현 가능한 benchmark 요약

## 설치

```bash
pip install ranksmith
```

## 빠른 시작

```python
from ranksmith import AzureOpenAIReranker, Document

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
)

results = reranker.rerank(
    query="What is listwise reranking?",
    documents=[
        Document(id="a", text="Listwise reranking compares candidates together."),
        Document(id="b", text="Vector search retrieves candidate documents."),
    ],
    top_k=2,
)

for result in results:
    print(result.rank, result.original_index, result.document.id)
```

`rank`는 사람이 읽기 쉬운 1-based 값입니다. `original_index`는 입력 리스트로
다시 매핑하기 쉽도록 0-based 값입니다.

## 지원하는 전략 및 알고리즘 (Strategy & Algorithm)

`ranksmith`는 평가 방식(Strategy)과 실행 절차(Algorithm)를 분리합니다.

### 추천 사용 시나리오

| Method | Strategy | 추천 상황 | 비용 / 위험 |
| --- | --- | --- | --- |
| `rankgpt_sliding_window` | `ListwiseStrategy` | production 또는 evaluation에서 기본 LLM reranker가 필요할 때 | 호출 수가 적지만, 한 번에 전체 순위를 출력해야 하므로 output format에 민감할 수 있음. `window_size >= N`이면 one-shot listwise reranking이 됨 |
| `prp_sliding_k` | `PairwiseStrategy` | pairwise preference 비교가 필요하거나 PRP 방식 재현이 필요할 때 | LLM 호출 수가 많고, 기본 `passes=10`은 비용이 큼 |
| `setwise_heapsort` | `SetwiseStrategy` | 실용적인 long-context 설정에서 pairwise PRP보다 적은 호출로 top-k 중심 setwise selection을 하고 싶을 때 | `set_size`에 따라 품질이 달라짐. 큰 set은 호출 수를 줄이지만 selection prompt가 어려워질 수 있음 |
| `tourrank_r`, `rounds=2` | `TourRankStrategy` | 중간 수준 호출 예산에서 listwise보다 강한 품질을 원할 때 | RankGPT보다 호출 수가 많지만 TourRank-10보다 훨씬 가벼움 |
| `tourrank_r`, `rounds=10` | `TourRankStrategy` | 품질 중심 offline reranking, 논문식 평가, 최종 reranking처럼 latency를 감수할 수 있을 때 | 일반 사용 기준 built-in 중 호출 비용이 가장 큼 |
| `acurank` | `AcuRankStrategy` | top-k 경계 근처의 불확실한 후보에 listwise 호출을 집중하고 싶을 때 | TrueSkill 상태를 사용하며, cap을 두지 않으면 기본 listwise보다 호출 수가 늘 수 있음 |
| `confidence_gain` | `ConfidenceGainStrategy` | query-only 및 query+context confidence scorer를 학습했고 `Conf(Q+C)-Conf(Q)`로 문서를 정렬하고 싶을 때 | scorer artifact와 answer generator hook이 필요함. 문서 수가 `N`이면 runtime에서 answer generation `N+1`회, confidence scoring `N+1`회를 수행함 |
| `cbdr` | `CBDRStrategy` | answerability confidence scorer를 학습했고 `Conf(Q)`가 충분히 높으면 context reranking을 건너뛰고, 낮으면 confidence gain으로 정렬하고 싶을 때 | scorer artifact와 answer generator hook이 필요함. skip path는 answer generation 1회와 confidence scoring 1회, rerank path는 각각 `N+1`회를 수행함 |
| Custom strategy | `RerankStrategy` / `AsyncRerankStrategy` | deterministic business logic, proprietary ranking, 새 research method가 필요할 때 | ranking contract와 validation을 직접 책임져야 함 |

### 전략 적용 방법

Strategy를 설정한 뒤 `AzureOpenAIReranker`에 전달합니다.

```python
from ranksmith import AzureOpenAIReranker, ListwiseStrategy

strategy = ListwiseStrategy(
    algorithm="rankgpt_sliding_window",
    window_size=20,
    stride=10,
    max_document_chars=4000,
)

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=strategy,
)

results = reranker.rerank("query", documents)
```

Pairwise PRP도 같은 reranker facade에 다른 Strategy를 주입해서 사용합니다.

```python
from ranksmith import AzureOpenAIReranker, PairwiseStrategy

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=PairwiseStrategy(passes=3),
)
```

TourRank-r도 같은 주입 지점을 사용합니다.

```python
from ranksmith import AzureOpenAIReranker, TourRankStrategy

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=TourRankStrategy(rounds=2, group_parallelism=1),
)
```

품질 중심 실행에서는 TourRank-10을 명시합니다.

```python
reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=TourRankStrategy(rounds=10),
)
```

AcuRank는 listwise reranker 호출 결과를 TrueSkill 기반 relevance 추정의
evidence로 사용합니다.

```python
from ranksmith import AcuRankStrategy, AzureOpenAIReranker

reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=AcuRankStrategy(
        target_rank=10,
        window_size=20,
        max_adaptive_reranker_calls=20,  # 선택적 adaptive phase budget cap.
        batch_parallelism=2,  # 선택 사항. provider thread-safety가 불확실하면 1 유지.
    ),
)
```

모든 `Document`에 numeric `metadata["score"]`가 있으면 AcuRank는 이를 first-stage
prior로 사용합니다. score가 전혀 없으면 standard TrueSkill prior를 사용합니다.
일부 문서에만 score가 있거나 boolean score 값이면 fast fail합니다.

후보 수가 작을 때는 `target_rank`를 문서 수로 자동 제한합니다.
`max_adaptive_reranker_calls`는 adaptive refinement phase만 제한하며, 선택적
initial pass 호출은 결과 metadata에서 별도로 함께 집계됩니다.
`batch_parallelism`은 같은 AcuRank iteration 안의 독립 batch를 병렬 호출하되,
posterior update는 deterministic batch order로 적용합니다.

> **참고**: `strategy`를 명시하지 않으면 기본적으로 `ListwiseStrategy(algorithm="rankgpt_sliding_window")`가 자동으로 적용됩니다. Pairwise PRP, Setwise, TourRank-r, AcuRank는 기본 listwise보다 LLM 호출 수가 많을 수 있으므로 live benchmark 전 호출 수를 확인해야 합니다.

## 커스텀 Strategy

커스텀 reranking 메소드는 `ListwiseStrategy.algorithm`에 새 문자열 값을 추가하는
방식보다, 새 Strategy 클래스로 구현하는 방식을 권장합니다. Strategy는 정규화된
`Document` 목록, model client, 선택적 `top_k`를 받아 `RerankResult` 목록을 반환합니다.

```python
from collections.abc import Sequence

from ranksmith import (
    AzureOpenAIReranker,
    Document,
    RerankResult,
)


class LengthStrategy:
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
        model_client: object,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del query, model_client
        ordered_indexes = sorted(
            range(len(documents)),
            key=lambda index: len(documents[index].text),
            reverse=True,
        )
        results = [
            RerankResult(
                document=documents[original_index],
                rank=rank,
                original_index=original_index,
                metadata={"strategy": "length"},
            )
            for rank, original_index in enumerate(ordered_indexes, start=1)
        ]
        return results if top_k is None else results[:top_k]


reranker = AzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    strategy=LengthStrategy(),
)
```

model-backed Strategy와 async Strategy도 같은 public contract를 따릅니다.
자세한 확장 가이드는
[커스텀 Strategy 확장 가이드](https://github.com/pko89403/ranksmith/blob/main/docs/wiki/08_custom_strategy_extension.md)와
[custom strategy 예제](https://github.com/pko89403/ranksmith/blob/main/examples/custom_strategy.py)를
참고하세요.

## Model Provider Architecture

`ModelClient`는 ranksmith 도메인의 prompt와 `rank` / `compare` / `select`
계약을 담당합니다. `ModelProvider`는 vendor별 JSON completion 호출만 담당합니다.

| Layer | 책임 | Public methods |
| --- | --- | --- |
| `Strategy` | 최종 reranking 순서를 만든다. | `rerank(...)` |
| `ModelClient` | ranksmith prompt 생성, ranking 도메인 계약, usage 전달을 담당한다. | `rank(...)`, `compare(...)`, `select(...)` |
| `ModelProvider` | vendor SDK를 호출하고 JSON completion text를 반환한다. | `complete(...)` |

```python
from ranksmith import AzureAOAIProvider, ModelClient

provider = AzureAOAIProvider(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
    api_version="2024-08-01-preview",
)
model_client = ModelClient(provider=provider)
```

같은 `ModelClient`는 모든 built-in Strategy에 사용할 수 있습니다.

```python
from ranksmith import AzureOpenAIReranker, PairwiseStrategy

reranker = AzureOpenAIReranker(
    model_client=model_client,
    strategy=PairwiseStrategy(passes=3),
)
```

`OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`는 향후 SDK 구현을 위한
public stub입니다. 호출하면 `RerankProviderError`로 fast fail 합니다.

## 비동기 지원 (Async Support)

대규모 트래픽이나 FastAPI 같은 비동기 웹 프레임워크를 위해 async reranker를
제공합니다.

```python
from ranksmith import AsyncAzureOpenAIReranker

reranker = AsyncAzureOpenAIReranker(
    api_key="...",
    azure_endpoint="https://example.openai.azure.com",
    azure_deployment="gpt-4o-mini",
)

results = await reranker.rerank("query", documents)
```

## Structural Confidence

`ranksmith.confidence`는 frozen HuggingFace encoder, `structural-v1` feature,
학습된 compatible scorer artifact를 사용해 closed-model output에 대한 single-item
및 bounded batch sync confidence inference를 제공합니다.

선택 dependency 설치:

```bash
pip install "ranksmith[confidence]"
```

```python
from ranksmith.confidence import (
    AnswerConfidenceInput,
    StructuralConfidenceEstimator,
)

estimator = StructuralConfidenceEstimator.from_artifact(
    "structural-confidence.joblib",
)

result = estimator.score(
    AnswerConfidenceInput(context="...", answer="...")
)
print(result.score)

batch_results = estimator.score_batch(
    [AnswerConfidenceInput(context="...", answer="...")],
    batch_size=8,
    max_workers=1,
)
```

이 모듈은 scorer를 학습하지 않고, reranking Strategy를 추가하지 않으며, async
inference를 수행하지 않습니다. 병렬 batch scoring은 같은 encoder/scorer instance를
worker thread들이 공유하므로 thread-safe backend에서만 `max_workers>1`을 사용해야
합니다. 첫 worker error에서 pending work를 취소하지만, 이미 시작된 Python thread는
background에서 완료될 수 있습니다.

`ConfidenceGainStrategy`는 별도 sync reranking Strategy입니다. 두 개의 compatible
confidence estimator와 answer generator hook을 받아 사용합니다.

```python
from ranksmith.confidence import StructuralConfidenceEstimator
from ranksmith.strategies import ConfidenceGainStrategy

base_estimator = StructuralConfidenceEstimator.from_artifact(
    "query-answerability.joblib"
)
context_estimator = StructuralConfidenceEstimator.from_artifact(
    "query-context-answerability.joblib"
)

strategy = ConfidenceGainStrategy(
    base_estimator=base_estimator,
    context_estimator=context_estimator,
    answer_generator=my_answer_generator,
)
```

이 Strategy는 `Conf(Q+C)-Conf(Q)` 기준으로 정렬합니다. CBDR retrieval skip, async
reranking, scorer 학습은 구현하지 않습니다.

`CBDRStrategy`는 sync reranking-side router입니다. retriever와 통합하거나 upstream
retrieval 호출 자체를 멈추지는 않습니다. 이미 `rerank(...)`에 documents가 전달된
뒤 context reranking을 건너뛸지 결정합니다.

```python
from ranksmith.integrations import AzureAnswerGenerator
from ranksmith.strategies import CBDRStrategy

answer_generator = AzureAnswerGenerator.from_env()

strategy = CBDRStrategy.from_artifacts(
    base_artifact_path="query-answerability.joblib",
    context_artifact_path="query-context-answerability.joblib",
    answer_generator=answer_generator,
    skip_threshold=0.8,
)

results = strategy.rerank(query=query, documents=documents)
```

`Conf(Q) >= skip_threshold`이면 original document order를 보존하고
`metadata["cbdr_skipped"] == True`를 남깁니다. `Conf(Q) < skip_threshold`이면
모든 문서를 scoring한 뒤 `top_k`를 적용합니다.
`AzureAnswerGenerator`는 `ranksmith.confidence_generation`과 같은 no-answer
sentinel 계약을 사용하며, 답할 수 없으면 `{"answer":"__NO_ANSWER__"}`를
반환하게 합니다.

compatible scorer artifact가 있으면 benchmark runner에서도 CBDR을 명시적으로 실행할
수 있습니다.

```bash
uv run python scripts/compare_reranking.py \
  --dataset benchmark-cache \
  --cache-dir .benchmark-cache/askubuntu-bm25 \
  --candidates benchmark-results/pyserini/askubuntu-bm25-top20.trec \
  --algorithm cbdr \
  --cbdr-base-artifact query-answerability.joblib \
  --cbdr-context-artifact query-context-answerability.joblib \
  --cbdr-max-document-chars 4000 \
  --allow-live
```

`ranksmith.confidence_generation`은 raw answer/relevance/answerability 예시에 대해
closed model을 호출해 confidence training용 supervised canonical JSONL을 생성할 수
있습니다. 이 모듈은 reranking Strategy가 아니라 데이터 생성 utility입니다.

### compatible confidence scorer 학습

`ranksmith.confidence_training`은 supervised canonical JSONL에서 Phase 1 compatible
scorer artifact를 학습할 수 있습니다. label 생성, closed model 호출, dataset
adapter, reranking benchmark 수치 보고는 수행하지 않습니다.

학습 dependency 설치:

```bash
pip install "ranksmith[confidence-train]"
```

```python
from ranksmith.confidence_training import (
    ConfidenceTrainingConfig,
    train_confidence_scorer,
)

result = train_confidence_scorer(
    ConfidenceTrainingConfig(
        task_type="answer_confidence",
        dataset_path="answer_confidence.jsonl",
        output_dir="confidence-runs/answer-v1",
        export_path="artifacts/answer_confidence.joblib",
    )
)
print(result.export_path)
```

### LM Studio 로컬 confidence pipeline

CBDR에는 두 answerability scorer가 필요합니다. query-only 예시에서 `Conf(Q)`를,
query+context 예시에서 `Conf(Q+C)`를 학습합니다. LM Studio는 supervised label
생성에만 사용하며, scorer artifact는 그대로 `ranksmith.confidence_training`이
학습합니다.

로컬 OpenAI-compatible server를 시작하고 loaded model을 지정합니다.

```bash
lms server start
export LMSTUDIO_MODEL=google/gemma-4-12b
```

canonical JSONL dataset을 생성합니다.

```bash
uv run python scripts/generate_confidence_dataset.py \
  --task query_answerability_confidence \
  --provider lmstudio \
  --input runs/confidence/local/raw/query_answerability.jsonl \
  --output runs/confidence/local/canonical/query_answerability_confidence.jsonl \
  --resume

uv run python scripts/generate_confidence_dataset.py \
  --task query_context_answerability_confidence \
  --provider lmstudio \
  --input runs/confidence/local/raw/query_context_answerability.jsonl \
  --output runs/confidence/local/canonical/query_context_answerability_confidence.jsonl \
  --max-context-chars 8000 \
  --resume
```

scorer를 범용적으로 다루기 전에 source/group balance를 확인합니다.

```bash
uv run python scripts/report_confidence_dataset.py \
  --task query_answerability_confidence \
  --dataset runs/confidence/local/canonical/query_answerability_confidence.jsonl
```

CBDR-compatible scorer artifact를 학습합니다.

```bash
uv run python scripts/train_confidence_scorer.py \
  --task query_answerability_confidence \
  --dataset runs/confidence/local/canonical/query_answerability_confidence.jsonl \
  --output-dir runs/confidence/local/training/query_answerability \
  --export-path runs/confidence/local/artifacts/query_answerability.joblib \
  --encoder-name bert-base-uncased \
  --max-length 256

uv run python scripts/train_confidence_scorer.py \
  --task query_context_answerability_confidence \
  --dataset runs/confidence/local/canonical/query_context_answerability_confidence.jsonl \
  --output-dir runs/confidence/local/training/query_context_answerability \
  --export-path runs/confidence/local/artifacts/query_context_answerability.joblib \
  --encoder-name bert-base-uncased \
  --max-length 256
```

학습된 artifact를 LM Studio runtime과 함께 사용합니다.

```python
from ranksmith.integrations import LMStudioModelProvider, ProviderAnswerGenerator
from ranksmith.strategies import CBDRStrategy

answer_generator = ProviderAnswerGenerator(
    provider=LMStudioModelProvider(model="google/gemma-4-12b")
)

strategy = CBDRStrategy.from_artifacts(
    base_artifact_path="runs/confidence/local/artifacts/query_answerability.joblib",
    context_artifact_path="runs/confidence/local/artifacts/query_context_answerability.joblib",
    answer_generator=answer_generator,
    skip_threshold=0.8,
)
```

benchmark runner도 같은 provider를 사용할 수 있습니다. 이 명령은 live 실행이므로
`--allow-live`가 필요합니다. summary artifact를 만들고 커밋하기 전까지는 benchmark
품질 수치로 주장하지 않습니다.

```bash
uv run python scripts/compare_reranking.py \
  --dataset benchmark-cache \
  --cache-dir .benchmark-cache/askubuntu-bm25 \
  --candidates benchmark-results/pyserini/askubuntu-bm25-top20.trec \
  --algorithm cbdr \
  --cbdr-answer-provider lmstudio \
  --cbdr-base-artifact runs/confidence/local/artifacts/query_answerability.joblib \
  --cbdr-context-artifact runs/confidence/local/artifacts/query_context_answerability.joblib \
  --lmstudio-model google/gemma-4-12b \
  --allow-live
```

## 실전 가이드 (Examples)

실행 가능한 예제는 `examples/` 폴더에 있습니다.

- [rankgpt_sync.py](https://github.com/pko89403/ranksmith/blob/main/examples/rankgpt_sync.py): 동기 RankGPT 연동
- [rankgpt_async.py](https://github.com/pko89403/ranksmith/blob/main/examples/rankgpt_async.py): 비동기 RankGPT 연동
- [pairwise_prp.py](https://github.com/pko89403/ranksmith/blob/main/examples/pairwise_prp.py): pairwise PRP Strategy
- [setwise_heapsort.py](https://github.com/pko89403/ranksmith/blob/main/examples/setwise_heapsort.py): fake provider 기반 Setwise Heapsort
- [tourrank.py](https://github.com/pko89403/ranksmith/blob/main/examples/tourrank.py): fake provider 기반 TourRank-r
- [acurank.py](https://github.com/pko89403/ranksmith/blob/main/examples/acurank.py): first-stage score prior 기반 AcuRank
- [custom_strategy.py](https://github.com/pko89403/ranksmith/blob/main/examples/custom_strategy.py): custom Strategy 계약

## 벤치마크

아래 benchmark는 reranking만 측정합니다. Pyserini BM25는 고정된 first-stage
candidate를 만들고, `ranksmith`는 retrieval 없이 해당 후보만 재정렬합니다.
실행 조건은 `AskUbuntuDupQuestions` test data, query `361`개, query당 BM25
top-20 후보, `@5` 평가입니다. top-k 조기 종료를 지원하는 method는 평가 대상인
top-5만 출력할 수 있습니다. Live LLM 호출에는 Azure OpenAI deployment
`gpt-5.4-nano`를 사용했습니다.

잘못된 LLM 출력은 조용히 보정하거나 자동 복구하지 않았습니다. 대신 재호출했고,
끝까지 남은 invalid row는 invalid로 보고했습니다.

표에서는 algorithm 기준 nominal 호출 수와 retry를 포함한 row-level attempt 수를
분리합니다. row attempt는 retry accounting에는 유용하지만, multi-call method가
중간 단계에서 실패할 수 있으므로 정확한 provider-call telemetry는 아닙니다.
커밋된 근거 artifact는 다음과 같습니다.

- [`benchmark-results/live/askubuntu-bm25-top20-default-live.v3.merged.json`](https://github.com/pko89403/ranksmith/blob/main/benchmark-results/live/askubuntu-bm25-top20-default-live.v3.merged.json)
- [`benchmark-results/pyserini/askubuntu-bm25-top20.trec`](https://github.com/pko89403/ranksmith/blob/main/benchmark-results/pyserini/askubuntu-bm25-top20.trec)
- [`benchmark-results/askubuntu-bm25-top20-cbdr-live.json`](https://github.com/pko89403/ranksmith/blob/main/benchmark-results/askubuntu-bm25-top20-cbdr-live.json) (optional `cbdr` method, 별도 run)

| Method | NDCG@5 | MRR@5 | Recall@5 | Valid rows | Invalid rate | Nominal LLM calls/query | LLM row attempts/query incl. retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original_bm25` | 0.3520 | 0.5062 | 0.2862 | 361/361 | 0.000 | 0 | N/A |
| `single_call_listwise@20` | 0.4082 | 0.5541 | 0.3345 | 359/361 | 0.006 | 1 | 1.04 |
| `rankgpt_sw_w5` | 0.3973 | 0.5283 | 0.3366 | 361/361 | 0.000 | 9 | 1.01 |
| `acurank_k5_b1` | 0.4053 | 0.5491 | 0.3377 | 356/361 | 0.014 | 2 | 1.12 |
| `tourrank_r2` | 0.4236 | 0.5725 | 0.3601 | 361/361 | 0.000 | 8 | 1.03 |
| `setwise_hs_s10` | 0.3653 | 0.5059 | 0.3005 | 361/361 | 0.000 | 12 | 1.00 |
| `prp_sliding_p1` | 0.4065 | 0.5818 | 0.3277 | 361/361 | 0.000 | 38 | 1.00 |
| `cbdr` *(scorer: TriviaQA, 도메인 밖)* | 0.2259 | 0.3458 | 0.1867 | 361/361 | 0.000 | 21 | 1.00 |

`tourrank_r2`는 NDCG@5와 Recall@5가 가장 높았고, `prp_sliding_p1`은 MRR@5가
가장 높았습니다. `single_call_listwise@20`은 one-shot listwise baseline입니다.
`rankgpt_sw_w5`는 이 top-20 설정의 실제 sliding-window listwise baseline입니다.
`acurank_k5_b1`은 AcuRank uncertainty boundary를 `@5` 평가 cutoff와 맞춘
설정입니다. `setwise_hs_s10`은 20개 후보에서 평가 대상 top-5만 추출하는 실용적인
Setwise Heapsort 설정입니다. `cbdr`은 [LM Studio 로컬 confidence
pipeline](#lm-studio-로컬-confidence-pipeline)에 문서화된 `Conf(Q)`/`Conf(Q+C)`
스코어러 두 개를 사용하며, TriviaQA로 학습해 AskUbuntu 기준 도메인 밖입니다.
이 벤치마크에서는 BM25 baseline보다 낮은 점수를 기록했고, 이기도록 튜닝하지
않고 측정된 그대로 보고합니다.

왜 이기게 튜닝하지 않는가: 스코어러를 이 벤치마크 분포에 맞추면 알고리즘의
일반적인 품질이 아니라 AskUbuntu에 대한 과적합을 측정하게 됩니다. 이는
smoke/partial run이나 cherry-pick된 수치를 벤치마크 품질로 보고하지 않는다는
이 프로젝트의 보고 규칙([`docs/benchmarks/bm25_top20_reranking.md`](docs/benchmarks/bm25_top20_reranking.md#reporting-rules))과도
어긋납니다. 더 강한 domain-in 스코어러를 만드는 건 정당한 후속 작업이지만,
그건 더 나은 artifact를 학습해서 같은 커맨드를 다시 돌리는 것이지 보고
방식을 바꾸는 게 아닙니다.

재시도 후에도 `single_call_listwise@20` 2개 row와 `acurank_k5_b1` 5개 row가
invalid로 남았습니다. 이 row는 보정하지 않고 invalid rate에 반영했습니다.

## 결과 모델

```python
result.document        # Document
result.rank            # 1-based rank
result.original_index  # 0-based input index
result.metadata        # 전략별 metadata
```

## 에러 처리

`ranksmith`는 fast fail 정책을 따릅니다. 긴 문서를 조용히 자르거나,
잘못된 순위를 자동 보정하거나, 검증되지 않은 LLM 출력을 반환하지 않습니다.

```python
from ranksmith import (
    DocumentTooLongError,
    RerankParseError,
    RerankProviderError,
    RerankStrategyError,
)

try:
    results = reranker.rerank("query", documents)
except DocumentTooLongError:
    ...
except RerankParseError:
    ...
except RerankProviderError:
    ...
except RerankStrategyError:
    ...
```
