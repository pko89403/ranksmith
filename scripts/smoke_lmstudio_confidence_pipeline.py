#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksmith.confidence import StructuralConfidenceResult, TaskType  # noqa: E402
from ranksmith.confidence.encoder import FrozenAutoEncoder  # noqa: E402
from ranksmith.confidence_generation import (  # noqa: E402
    ConfidenceGenerationResult,
    QueryAnswerabilityGenerationConfig,
    QueryContextAnswerabilityGenerationConfig,
    generate_query_answerability_confidence_dataset,
    generate_query_context_answerability_confidence_dataset,
)
from ranksmith.confidence_training import (  # noqa: E402
    ConfidenceTrainingConfig,
    ConfidenceTrainingResult,
)
from ranksmith.confidence_training.artifact import (  # noqa: E402
    export_scorer_artifact,
    write_metadata_json,
)
from ranksmith.confidence_training.dataset import (  # noqa: E402
    load_canonical_dataset,
)
from ranksmith.confidence_training.dataset_report import (  # noqa: E402
    build_dataset_report,
)
from ranksmith.confidence_training.features import (  # noqa: E402
    extract_feature_rows,
)
from ranksmith.confidence_training.manifest import (  # noqa: E402
    build_dataset_manifest,
    build_split_manifest,
    json_hash,
    training_config_hash,
)
from ranksmith.confidence_training.pipeline import (  # noqa: E402
    _feature_row_json,
    _write_json,
    _write_model,
    _write_report_markdown,
)
from ranksmith.confidence_training.report import (  # noqa: E402
    generate_training_report,
)
from ranksmith.confidence_training.split import split_dataset  # noqa: E402
from ranksmith.confidence_training.train import (  # noqa: E402
    calibrate_classifier,
    train_lightgbm_classifier,
)
from ranksmith.confidence_training.types import (  # noqa: E402
    ConfidenceFeatureRow,
)
from ranksmith.integrations import (  # noqa: E402
    LMStudioModelProvider,
    ProviderAnswerGenerator,
)
from ranksmith.strategies import CBDRStrategy  # noqa: E402
from ranksmith.types import Document  # noqa: E402

TASK_QUERY = "query_answerability_confidence"
TASK_QUERY_CONTEXT = "query_context_answerability_confidence"
DEFAULT_ENCODER = "hf-internal-testing/tiny-random-bert"


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == ["__extract_features"]:
        return _extract_features_main(raw_argv[1:])
    if raw_argv[:1] == ["__rerank_smoke"]:
        return _rerank_smoke_main(raw_argv[1:])

    args = _parse_args(raw_argv)
    run_dir = _resolve_run_dir(args.output_dir)
    _prepare_run_dir(run_dir, overwrite=args.overwrite)

    provider = LMStudioModelProvider(
        model=args.lmstudio_model,
        base_url=args.lmstudio_base_url,
        api_key=args.lmstudio_api_key,
        timeout=args.timeout,
        max_tokens=args.lmstudio_max_tokens,
    )

    raw_dir = run_dir / "raw"
    canonical_dir = run_dir / "canonical"
    training_dir = run_dir / "training"
    artifacts_dir = run_dir / "artifacts"
    reports_dir = run_dir / "reports"
    for directory in (raw_dir, canonical_dir, training_dir, artifacts_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    live_query_raw = raw_dir / "live_query_answerability.jsonl"
    live_context_raw = raw_dir / "live_query_context_answerability.jsonl"
    live_query_canonical = canonical_dir / "live_query_answerability.jsonl"
    live_context_canonical = canonical_dir / "live_query_context_answerability.jsonl"
    training_query_canonical = canonical_dir / "training_query_answerability.jsonl"
    training_context_canonical = (
        canonical_dir / "training_query_context_answerability.jsonl"
    )

    _write_jsonl(live_query_raw, _live_query_rows())
    _write_jsonl(live_context_raw, _live_context_rows())

    live_query_result = generate_query_answerability_confidence_dataset(
        QueryAnswerabilityGenerationConfig(
            input_path=live_query_raw,
            output_path=live_query_canonical,
            provider=provider,
            overwrite=True,
            max_items=args.max_items,
            source="lmstudio-smoke-live",
        )
    )
    live_context_result = generate_query_context_answerability_confidence_dataset(
        QueryContextAnswerabilityGenerationConfig(
            input_path=live_context_raw,
            output_path=live_context_canonical,
            provider=provider,
            overwrite=True,
            max_items=args.max_items,
            max_context_chars=args.max_context_chars,
            source="lmstudio-smoke-live",
        )
    )

    _write_json(
        reports_dir / "live_query_report.json",
        _dataset_report(live_query_canonical, TASK_QUERY),
    )
    _write_json(
        reports_dir / "live_query_context_report.json",
        _dataset_report(live_context_canonical, TASK_QUERY_CONTEXT),
    )

    _write_jsonl(training_query_canonical, _training_query_rows())
    _write_jsonl(training_context_canonical, _training_context_rows())
    _write_json(
        reports_dir / "training_query_report.json",
        _dataset_report(training_query_canonical, TASK_QUERY),
    )
    _write_json(
        reports_dir / "training_query_context_report.json",
        _dataset_report(training_context_canonical, TASK_QUERY_CONTEXT),
    )

    base_artifact = artifacts_dir / "query_answerability.joblib"
    context_artifact = artifacts_dir / "query_context_answerability.joblib"
    base_training = _train_smoke_scorer(
        _training_config(
            task_type=TASK_QUERY,
            dataset_path=training_query_canonical,
            output_dir=training_dir / "query_answerability",
            export_path=base_artifact,
            args=args,
        )
    )
    context_training = _train_smoke_scorer(
        _training_config(
            task_type=TASK_QUERY_CONTEXT,
            dataset_path=training_context_canonical,
            output_dir=training_dir / "query_context_answerability",
            export_path=context_artifact,
            args=args,
        )
    )

    answer_generator = ProviderAnswerGenerator(provider=provider)
    strategy = CBDRStrategy.from_artifacts(
        base_artifact_path=base_training.export_path,
        context_artifact_path=context_training.export_path,
        answer_generator=answer_generator,
        skip_threshold=args.skip_threshold,
        cache_dir=args.cache_dir,
        device=args.device,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
        allow_truncation=args.allow_truncation,
    )
    rerank_results = _run_optional_rerank_smoke(
        include_rerank=args.include_rerank,
        run_dir=run_dir,
        base_artifact=base_training.export_path,
        context_artifact=context_training.export_path,
        args=args,
    )

    summary = {
        "run_dir": str(run_dir),
        "live_generation": {
            "query_answerability": _generation_summary(live_query_result),
            "query_context_answerability": _generation_summary(live_context_result),
        },
        "training_artifacts": {
            "query_answerability": str(base_training.export_path),
            "query_context_answerability": str(context_training.export_path),
        },
        "reports_dir": str(reports_dir),
        "cbdr_smoke": {
            "strategy_loaded": strategy.algorithm == "cbdr",
            "rerank_executed": args.include_rerank,
            "results": rerank_results,
        },
    }
    _write_json(reports_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _extract_features_main(argv: Sequence[str]) -> int:
    args = _parse_extract_args(argv)
    config = _training_config_from_extract_args(args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_canonical_dataset(config.dataset_path, task_type=config.task_type)
    split = split_dataset(
        samples,
        seed=config.seed,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
    )
    encoder = FrozenAutoEncoder.from_pretrained(
        encoder_name=config.encoder_name,
        encoder_revision=config.encoder_revision,
        tokenizer_name=config.tokenizer_name,
        tokenizer_revision=config.tokenizer_revision,
        hf_token=None,
        local_files_only=config.local_files_only,
        cache_dir=config.cache_dir,
        device="cpu",
        max_length=config.max_length,
        allow_truncation=config.allow_truncation,
    )
    for name, part in (
        ("train", split.train),
        ("valid", split.valid),
        ("test", split.test),
    ):
        rows = extract_feature_rows(part, encoder=encoder)
        _write_feature_rows(output_dir / f"features_{name}.jsonl", rows)
    return 0


def _rerank_smoke_main(argv: Sequence[str]) -> int:
    args = _parse_rerank_args(argv)
    provider = LMStudioModelProvider(
        model=args.lmstudio_model,
        base_url=args.lmstudio_base_url,
        api_key=args.lmstudio_api_key,
        timeout=args.timeout,
        max_tokens=args.lmstudio_max_tokens,
    )
    answer_generator = _LoggingAnswerGenerator(
        ProviderAnswerGenerator(provider=provider),
        enabled=True,
    )
    loaded = CBDRStrategy.from_artifacts(
        base_artifact_path=args.base_artifact,
        context_artifact_path=args.context_artifact,
        answer_generator=answer_generator,
        skip_threshold=args.skip_threshold,
        cache_dir=args.cache_dir,
        device=args.device,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
        allow_truncation=args.allow_truncation,
    )
    strategy = CBDRStrategy(
        base_estimator=_LoggingEstimator(loaded.base_estimator, label="base"),
        context_estimator=_LoggingEstimator(loaded.context_estimator, label="context"),
        answer_generator=answer_generator,
        skip_threshold=args.skip_threshold,
        max_document_chars=loaded.max_document_chars,
    )
    results = _execute_rerank_smoke(strategy)
    _write_json(args.output_json, {"results": results})
    return 0


def _train_smoke_scorer(config: ConfidenceTrainingConfig) -> ConfidenceTrainingResult:
    output_dir = Path(config.output_dir)
    export_path = Path(config.export_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    _run_feature_extraction_subprocess(config)
    train_rows = _read_feature_rows(output_dir / "features_train.jsonl")
    valid_rows = _read_feature_rows(output_dir / "features_valid.jsonl")
    test_rows = _read_feature_rows(output_dir / "features_test.jsonl")

    model = train_lightgbm_classifier(train_rows, seed=config.seed)
    scorer = calibrate_classifier(model, valid_rows)
    report = generate_training_report(
        model,
        scorer,
        valid_rows=valid_rows,
        test_rows=test_rows,
    )
    _write_json(output_dir / "report.json", report.to_dict())
    _write_report_markdown(output_dir / "report.md", report)

    samples = load_canonical_dataset(config.dataset_path, task_type=config.task_type)
    split = split_dataset(
        samples,
        seed=config.seed,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
    )
    dataset_manifest = build_dataset_manifest(
        Path(config.dataset_path),
        task_type=config.task_type,
        sample_count=len(samples),
    )
    split_manifest = build_split_manifest(
        split,
        seed=config.seed,
        task_type=config.task_type,
    )
    _write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    _write_json(output_dir / "split_manifest.json", split_manifest)
    _write_model(output_dir / "model.joblib", scorer)

    metadata = export_scorer_artifact(
        scorer,
        config=config,
        train_count=len(split.train),
        valid_count=len(split.valid),
        test_count=len(split.test),
        dataset_manifest_hash=json_hash(dataset_manifest),
        training_config_hash=training_config_hash(config),
    )
    metadata_path = output_dir / "metadata.json"
    write_metadata_json(metadata_path, metadata)

    return ConfidenceTrainingResult(
        output_dir=output_dir,
        export_path=export_path,
        report_path=output_dir / "report.json",
        metadata_path=metadata_path,
    )


def _run_feature_extraction_subprocess(config: ConfidenceTrainingConfig) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "__extract_features",
        "--task",
        config.task_type,
        "--dataset",
        str(config.dataset_path),
        "--output-dir",
        str(config.output_dir),
        "--encoder-name",
        config.encoder_name,
        "--max-length",
        str(config.max_length),
        "--seed",
        str(config.seed),
    ]
    if config.encoder_revision is not None:
        command.extend(["--encoder-revision", config.encoder_revision])
    if config.tokenizer_name is not None:
        command.extend(["--tokenizer-name", config.tokenizer_name])
    if config.tokenizer_revision is not None:
        command.extend(["--tokenizer-revision", config.tokenizer_revision])
    if config.cache_dir is not None:
        command.extend(["--cache-dir", config.cache_dir])
    if config.local_files_only:
        command.append("--local-files-only")
    if config.allow_truncation:
        command.append("--allow-truncation")
    subprocess.run(command, cwd=ROOT, check=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a live LM Studio smoke for confidence generation, "
            "training, and CBDR loading."
        )
    )
    parser.add_argument("--lmstudio-model", default=None)
    parser.add_argument("--lmstudio-base-url", default=None)
    parser.add_argument("--lmstudio-api-key", default=None)
    parser.add_argument("--lmstudio-max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-items", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER)
    parser.add_argument("--encoder-revision", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--include-rerank", action="store_true")
    parser.add_argument("--skip-threshold", type=float, default=0.8)
    args = parser.parse_args(argv)
    if args.max_items < 1:
        parser.error("--max-items must be >= 1")
    if args.max_context_chars < 1:
        parser.error("--max-context-chars must be >= 1")
    if args.lmstudio_max_tokens < 1:
        parser.error("--lmstudio-max-tokens must be >= 1")
    if args.skip_threshold < 0.0 or args.skip_threshold > 1.0:
        parser.error("--skip-threshold must be in [0, 1]")
    return args


def _parse_extract_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=argparse.SUPPRESS)
    parser.add_argument(
        "--task", required=True, choices=(TASK_QUERY, TASK_QUERY_CONTEXT)
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--encoder-name", required=True)
    parser.add_argument("--encoder-revision", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args(argv)


def _parse_rerank_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=argparse.SUPPRESS)
    parser.add_argument("--base-artifact", required=True, type=Path)
    parser.add_argument("--context-artifact", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--lmstudio-model", default=None)
    parser.add_argument("--lmstudio-base-url", default=None)
    parser.add_argument("--lmstudio-api-key", default=None)
    parser.add_argument("--lmstudio-max-tokens", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--skip-threshold", type=float, required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--allow-truncation", action="store_true")
    return parser.parse_args(argv)


def _training_config_from_extract_args(
    args: argparse.Namespace,
) -> ConfidenceTrainingConfig:
    return ConfidenceTrainingConfig(
        task_type=cast(TaskType, args.task),
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        export_path=args.output_dir / "unused.joblib",
        encoder_name=args.encoder_name,
        encoder_revision=args.encoder_revision,
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.tokenizer_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
        allow_truncation=args.allow_truncation,
        seed=args.seed,
    )


def _resolve_run_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(tempfile.gettempdir()) / f"ranksmith-lmstudio-smoke-{timestamp}"


def _prepare_run_dir(run_dir: Path, *, overwrite: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise SystemExit(f"output dir already exists: {run_dir}. Use --overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)


def _training_config(
    *,
    task_type: str,
    dataset_path: Path,
    output_dir: Path,
    export_path: Path,
    args: argparse.Namespace,
) -> ConfidenceTrainingConfig:
    return ConfidenceTrainingConfig(
        task_type=cast(TaskType, task_type),
        dataset_path=dataset_path,
        output_dir=output_dir,
        export_path=export_path,
        encoder_name=args.encoder_name,
        encoder_revision=args.encoder_revision,
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.tokenizer_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
        allow_truncation=args.allow_truncation,
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1,
    )


def _dataset_report(path: Path, task_type: str) -> Mapping[str, Any]:
    return build_dataset_report(path, cast(TaskType, task_type))


def _generation_summary(result: ConfidenceGenerationResult) -> Mapping[str, Any]:
    return {
        "output_path": str(result.output_path),
        "input_count": result.input_count,
        "generated_count": result.generated_count,
        "skipped_count": result.skipped_count,
        "positive_count": result.positive_count,
        "negative_count": result.negative_count,
    }


def _live_query_rows() -> list[Mapping[str, Any]]:
    return [
        {
            "id": "live-q-1",
            "query": "What is the capital of France?",
            "gold_answer": "Paris",
            "source": "lmstudio-smoke",
            "group_id": "live-q",
        },
        {
            "id": "live-q-2",
            "query": (
                "What exact color was the mayor's umbrella "
                "in the unpublished town memo?"
            ),
            "gold_answer": "__NO_ANSWER__",
            "source": "lmstudio-smoke",
            "group_id": "live-q",
        },
    ]


def _live_context_rows() -> list[Mapping[str, Any]]:
    return [
        {
            "id": "live-qc-1",
            "query": "Who played Karen in Married to the Mob?",
            "context": "Nancy Travis played Karen in the film Married to the Mob.",
            "gold_answer": "Nancy Travis",
            "source": "lmstudio-smoke",
            "group_id": "live-qc",
        },
        {
            "id": "live-qc-2",
            "query": "Who played Karen in Married to the Mob?",
            "context": "Michelle Pfeiffer starred in Married to the Mob as Angela.",
            "gold_answer": "__NO_ANSWER__",
            "source": "lmstudio-smoke",
            "group_id": "live-qc",
        },
    ]


def _training_query_rows() -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for index in range(15):
        rows.append(
            {
                "id": f"train-q-pos-{index}",
                "task_type": TASK_QUERY,
                "query": f"What is synthetic known fact {index}?",
                "answer": f"Known answer {index}",
                "gold_answer": f"Known answer {index}",
                "label": 1,
                "source": "synthetic-smoke",
            }
        )
        rows.append(
            {
                "id": f"train-q-neg-{index}",
                "task_type": TASK_QUERY,
                "query": f"What is synthetic unknown fact {index}?",
                "answer": "__NO_ANSWER__",
                "gold_answer": "__NO_ANSWER__",
                "label": 0,
                "source": "synthetic-smoke",
            }
        )
    return rows


def _training_context_rows() -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for index in range(15):
        rows.append(
            {
                "id": f"train-qc-pos-{index}",
                "task_type": TASK_QUERY_CONTEXT,
                "query": f"Who is the synthetic actor {index}?",
                "context": f"Context says Synthetic Actor {index} played the role.",
                "answer": f"Synthetic Actor {index}",
                "gold_answer": f"Synthetic Actor {index}",
                "label": 1,
                "source": "synthetic-smoke",
            }
        )
        rows.append(
            {
                "id": f"train-qc-neg-{index}",
                "task_type": TASK_QUERY_CONTEXT,
                "query": f"Who is the synthetic missing actor {index}?",
                "context": "This context does not contain the requested actor.",
                "answer": "__NO_ANSWER__",
                "gold_answer": "__NO_ANSWER__",
                "label": 0,
                "source": "synthetic-smoke",
            }
        )
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_feature_rows(path: Path, rows: Sequence[ConfidenceFeatureRow]) -> None:
    path.write_text(
        "".join(json.dumps(_feature_row_json(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_feature_rows(path: Path) -> list[ConfidenceFeatureRow]:
    rows: list[ConfidenceFeatureRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(ConfidenceFeatureRow(**json.loads(line)))
    return rows


class _LoggingAnswerGenerator:
    def __init__(self, inner: ProviderAnswerGenerator, *, enabled: bool) -> None:
        self._inner = inner
        self._enabled = enabled

    def answer_query(self, query: str) -> str:
        if self._enabled:
            print("[cbdr] answer_query start", flush=True)
        answer = self._inner.answer_query(query)
        if self._enabled:
            print("[cbdr] answer_query done", flush=True)
        return answer

    def answer_with_context(self, query: str, context: str) -> str:
        if self._enabled:
            print(
                f"[cbdr] answer_with_context start chars={len(context)}",
                flush=True,
            )
        answer = self._inner.answer_with_context(query, context)
        if self._enabled:
            print("[cbdr] answer_with_context done", flush=True)
        return answer


class _LoggingEstimator:
    def __init__(self, inner: Any, *, label: str) -> None:
        self._inner = inner
        self._label = label

    @property
    def task_type(self) -> str:
        return str(self._inner.task_type)

    def score(self, item: object) -> StructuralConfidenceResult:
        print(f"[cbdr] {self._label}_estimator score start", flush=True)
        result = cast(StructuralConfidenceResult, self._inner.score(item))
        print(
            f"[cbdr] {self._label}_estimator score done score={result.score}",
            flush=True,
        )
        return result


def _run_optional_rerank_smoke(
    *,
    include_rerank: bool,
    run_dir: Path,
    base_artifact: Path,
    context_artifact: Path,
    args: argparse.Namespace,
) -> list[Mapping[str, Any]]:
    if not include_rerank:
        return []
    output_json = run_dir / "reports" / "cbdr_rerank_smoke.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "__rerank_smoke",
        "--base-artifact",
        str(base_artifact),
        "--context-artifact",
        str(context_artifact),
        "--output-json",
        str(output_json),
        "--lmstudio-max-tokens",
        str(args.lmstudio_max_tokens),
        "--timeout",
        str(args.timeout),
        "--skip-threshold",
        str(args.skip_threshold),
        "--device",
        args.device,
        "--max-length",
        str(args.max_length),
    ]
    if args.lmstudio_model is not None:
        command.extend(["--lmstudio-model", args.lmstudio_model])
    if args.lmstudio_base_url is not None:
        command.extend(["--lmstudio-base-url", args.lmstudio_base_url])
    if args.lmstudio_api_key is not None:
        command.extend(["--lmstudio-api-key", args.lmstudio_api_key])
    if args.cache_dir is not None:
        command.extend(["--cache-dir", args.cache_dir])
    if args.local_files_only:
        command.append("--local-files-only")
    if args.allow_truncation:
        command.append("--allow-truncation")
    subprocess.run(command, cwd=ROOT, check=True)
    data = json.loads(output_json.read_text(encoding="utf-8"))
    return list(data["results"])


def _execute_rerank_smoke(strategy: CBDRStrategy) -> list[Mapping[str, Any]]:
    print("[cbdr] rerank smoke start", flush=True)
    results = strategy.rerank(
        query="Who played Karen in Married to the Mob?",
        documents=[
            Document(
                id="nancy-travis",
                text="Nancy Travis played Karen in the film Married to the Mob.",
            ),
            Document(
                id="michelle-pfeiffer",
                text="Michelle Pfeiffer starred in Married to the Mob as Angela.",
            ),
        ],
        top_k=2,
    )
    print("[cbdr] rerank smoke done", flush=True)
    return [
        {
            "rank": result.rank,
            "document_id": result.document.id,
            "original_index": result.original_index,
            "metadata": result.metadata,
        }
        for result in results
    ]


if __name__ == "__main__":
    raise SystemExit(main())
