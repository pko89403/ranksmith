#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ranksmith.confidence import TaskType  # noqa: E402
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
from ranksmith.confidence_training.dataset import load_canonical_dataset  # noqa: E402
from ranksmith.confidence_training.dataset_report import (  # noqa: E402
    build_dataset_report,
)
from ranksmith.confidence_training.features import extract_feature_rows  # noqa: E402
from ranksmith.confidence_training.manifest import (  # noqa: E402
    build_dataset_manifest,
    build_split_manifest,
    json_hash,
    training_config_hash,
)
from ranksmith.confidence_training.pipeline import (  # noqa: E402
    _feature_row_json,
    _write_model,
    _write_report_markdown,
)
from ranksmith.confidence_training.report import generate_training_report  # noqa: E402
from ranksmith.confidence_training.split import split_dataset  # noqa: E402
from ranksmith.confidence_training.train import (  # noqa: E402
    calibrate_classifier,
    train_lightgbm_classifier,
)
from ranksmith.confidence_training.types import ConfidenceFeatureRow  # noqa: E402
from ranksmith.integrations import (  # noqa: E402
    LMStudioModelProvider,
    ProviderAnswerGenerator,
)
from ranksmith.model import ModelProvider, ModelRequest, ModelResponse  # noqa: E402
from ranksmith.strategies import CBDRStrategy  # noqa: E402

TASK_QUERY = "query_answerability_confidence"
TASK_QUERY_CONTEXT = "query_context_answerability_confidence"


@dataclass(frozen=True)
class _JsonObjectRetryProvider:
    provider: ModelProvider
    max_attempts: int

    def complete(self, request: ModelRequest) -> ModelResponse:
        last_response: ModelResponse | None = None
        for _ in range(self.max_attempts):
            response = self.provider.complete(request)
            last_response = response
            if _is_valid_json_object(response.content):
                return response
        if last_response is not None:
            return last_response
        return self.provider.complete(request)


def _is_valid_json_object(content: str) -> bool:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    answer = parsed.get("answer")
    return isinstance(answer, str) and answer.strip() != ""


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == ["__extract_features"]:
        return _extract_features_main(raw_argv[1:])

    args = _parse_args(raw_argv)
    run_dir = Path(args.run_dir)
    if (
        run_dir.exists()
        and any(run_dir.iterdir())
        and not (args.overwrite or args.resume_generation)
    ):
        raise SystemExit(
            f"run dir already exists and is not empty: {run_dir}. "
            "Use --resume-generation or --overwrite."
        )

    raw_dir = run_dir / "raw"
    canonical_dir = run_dir / "canonical"
    report_dir = run_dir / "reports"
    training_dir = run_dir / "training"
    artifact_dir = run_dir / "artifacts"
    for directory in (raw_dir, canonical_dir, report_dir, training_dir, artifact_dir):
        directory.mkdir(parents=True, exist_ok=True)

    query_raw = (
        Path(args.query_raw)
        if args.query_raw is not None
        else raw_dir / "query_answerability_raw.jsonl"
    )
    query_context_raw = (
        Path(args.query_context_raw)
        if args.query_context_raw is not None
        else raw_dir / "query_context_answerability_raw.jsonl"
    )
    if args.build_qa_raw:
        _build_qa_raw_files(
            raw_dir=raw_dir,
            query_raw=query_raw,
            query_context_raw=query_context_raw,
            args=args,
        )
    _require_input_file(query_raw, "--query-raw")
    _require_input_file(query_context_raw, "--query-context-raw")

    provider: ModelProvider = _JsonObjectRetryProvider(
        provider=LMStudioModelProvider(
            model=args.lmstudio_model,
            base_url=args.lmstudio_base_url,
            api_key=args.lmstudio_api_key,
            timeout=args.timeout,
            max_tokens=args.lmstudio_max_tokens,
        ),
        max_attempts=args.generation_max_attempts,
    )

    query_canonical = canonical_dir / "query_answerability_confidence.jsonl"
    context_canonical = canonical_dir / "query_context_answerability_confidence.jsonl"

    query_generation = generate_query_answerability_confidence_dataset(
        QueryAnswerabilityGenerationConfig(
            input_path=query_raw,
            output_path=query_canonical,
            provider=provider,
            overwrite=args.overwrite,
            resume=args.resume_generation,
            max_items=args.max_items,
            source=args.source,
        )
    )
    context_generation = generate_query_context_answerability_confidence_dataset(
        QueryContextAnswerabilityGenerationConfig(
            input_path=query_context_raw,
            output_path=context_canonical,
            provider=provider,
            overwrite=args.overwrite,
            resume=args.resume_generation,
            max_items=args.max_items,
            max_context_chars=args.max_context_chars,
            source=args.source,
        )
    )

    query_report = build_dataset_report(query_canonical, cast(TaskType, TASK_QUERY))
    context_report = build_dataset_report(
        context_canonical,
        cast(TaskType, TASK_QUERY_CONTEXT),
    )
    _write_json(report_dir / "query_answerability_dataset_report.json", query_report)
    _write_json(
        report_dir / "query_context_answerability_dataset_report.json",
        context_report,
    )

    base_artifact = artifact_dir / "query_answerability.joblib"
    context_artifact = artifact_dir / "query_context_answerability.joblib"
    query_training = _train_confidence_scorer_isolated(
        _training_config(
            task_type=TASK_QUERY,
            dataset_path=query_canonical,
            output_dir=training_dir / "query_answerability",
            export_path=base_artifact,
            args=args,
        )
    )
    context_training = _train_confidence_scorer_isolated(
        _training_config(
            task_type=TASK_QUERY_CONTEXT,
            dataset_path=context_canonical,
            output_dir=training_dir / "query_context_answerability",
            export_path=context_artifact,
            args=args,
        )
    )

    strategy = CBDRStrategy.from_artifacts(
        base_artifact_path=query_training.export_path,
        context_artifact_path=context_training.export_path,
        answer_generator=ProviderAnswerGenerator(provider=provider),
        skip_threshold=args.skip_threshold,
        cache_dir=args.cache_dir,
        device=args.device,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
        allow_truncation=args.allow_truncation,
    )
    benchmark_output = None
    benchmark_checkpoint_output = None
    if args.run_benchmark:
        benchmark_output = run_dir / "benchmark" / "cbdr_report.json"
        benchmark_checkpoint_output = run_dir / "benchmark" / "cbdr_checkpoint.jsonl"
        _run_benchmark(
            args=args,
            output=benchmark_output,
            checkpoint_output=benchmark_checkpoint_output,
            base_artifact=query_training.export_path,
            context_artifact=context_training.export_path,
        )

    summary = {
        "run_dir": str(run_dir),
        "generation": {
            "query_answerability": _generation_summary(query_generation),
            "query_context_answerability": _generation_summary(context_generation),
        },
        "dataset_reports": {
            "query_answerability": str(
                report_dir / "query_answerability_dataset_report.json"
            ),
            "query_context_answerability": str(
                report_dir / "query_context_answerability_dataset_report.json"
            ),
        },
        "training": {
            "query_answerability": _training_summary(query_training),
            "query_context_answerability": _training_summary(context_training),
        },
        "artifacts": {
            "query_answerability": str(query_training.export_path),
            "query_context_answerability": str(context_training.export_path),
        },
        "cbdr": {
            "strategy_loaded": strategy.algorithm == "cbdr",
            "skip_threshold": args.skip_threshold,
        },
        "benchmark": {
            "executed": args.run_benchmark,
            "output": str(benchmark_output) if benchmark_output is not None else None,
            "checkpoint_output": (
                str(benchmark_checkpoint_output)
                if benchmark_checkpoint_output is not None
                else None
            ),
            "command": _benchmark_command(run_dir, args=args),
        },
    }
    _write_json(report_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full LM Studio confidence pipeline: generation, reports, "
            "training, artifact export, and CBDR loading."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--query-raw", type=Path, default=None)
    parser.add_argument("--query-context-raw", type=Path, default=None)
    parser.add_argument("--build-qa-raw", action="store_true")
    parser.add_argument("--qa-source", choices=("triviaqa",), default="triviaqa")
    parser.add_argument("--qa-dataset-name", default="mandarjoshi/trivia_qa")
    parser.add_argument("--qa-dataset-config", default="rc")
    parser.add_argument("--qa-split", default="train[:20000]")
    parser.add_argument("--qa-input-jsonl", type=Path, default=None)
    parser.add_argument("--qa-max-source-items", type=int, default=20000)
    parser.add_argument("--qa-max-query-items", type=int, default=5000)
    parser.add_argument("--qa-max-query-context-items", type=int, default=5000)
    parser.add_argument("--lmstudio-model", default=None)
    parser.add_argument("--lmstudio-base-url", default=None)
    parser.add_argument("--lmstudio-api-key", default=None)
    parser.add_argument("--lmstudio-max-tokens", type=int, default=128)
    parser.add_argument("--generation-max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--source", default=None)
    parser.add_argument("--resume-generation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--encoder-name", default="bert-base-uncased")
    parser.add_argument("--encoder-revision", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--skip-threshold", type=float, default=0.8)
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument(
        "--benchmark-dataset",
        choices=("fixture", "benchmark-cache", "beir-scifact"),
        default="benchmark-cache",
    )
    parser.add_argument("--benchmark-cache-dir", type=Path)
    parser.add_argument("--benchmark-candidates", type=Path)
    parser.add_argument("--benchmark-max-cases", type=int)
    parser.add_argument("--benchmark-top-k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.build_qa_raw and (
        args.query_raw is not None or args.query_context_raw is not None
    ):
        parser.error("--build-qa-raw cannot be combined with explicit raw paths")
    if not args.build_qa_raw and (
        args.query_raw is None or args.query_context_raw is None
    ):
        parser.error(
            "--query-raw and --query-context-raw are required unless "
            "--build-qa-raw is set"
        )
    if args.qa_max_query_items < 30:
        parser.error("--qa-max-query-items must be >= 30")
    if args.qa_max_query_context_items < 30:
        parser.error("--qa-max-query-context-items must be >= 30")
    if args.qa_max_source_items < 2:
        parser.error("--qa-max-source-items must be >= 2")
    if args.qa_max_query_context_items % 2 != 0:
        parser.error("--qa-max-query-context-items must be even")
    if args.overwrite and args.resume_generation:
        parser.error("--overwrite and --resume-generation cannot both be set")
    if args.lmstudio_max_tokens < 1:
        parser.error("--lmstudio-max-tokens must be >= 1")
    if args.generation_max_attempts < 1:
        parser.error("--generation-max-attempts must be >= 1")
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be >= 1")
    if args.max_context_chars < 1:
        parser.error("--max-context-chars must be >= 1")
    if args.skip_threshold < 0.0 or args.skip_threshold > 1.0:
        parser.error("--skip-threshold must be in [0, 1]")
    if args.run_benchmark and not args.allow_live:
        parser.error("--run-benchmark requires --allow-live")
    if args.run_benchmark and args.benchmark_dataset == "benchmark-cache":
        if args.benchmark_cache_dir is None:
            parser.error("--benchmark-cache-dir is required for benchmark-cache")
        if args.benchmark_candidates is None:
            parser.error("--benchmark-candidates is required for benchmark-cache")
    if args.benchmark_max_cases is not None and args.benchmark_max_cases < 1:
        parser.error("--benchmark-max-cases must be >= 1")
    if args.benchmark_top_k < 1:
        parser.error("--benchmark-top-k must be >= 1")
    return args


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
    )


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


def _train_confidence_scorer_isolated(
    config: ConfidenceTrainingConfig,
) -> ConfidenceTrainingResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = Path(config.export_path)

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


def _generation_summary(result: ConfidenceGenerationResult) -> Mapping[str, Any]:
    return {
        "output_path": str(result.output_path),
        "input_count": result.input_count,
        "generated_count": result.generated_count,
        "skipped_count": result.skipped_count,
        "positive_count": result.positive_count,
        "negative_count": result.negative_count,
    }


def _training_summary(result: ConfidenceTrainingResult) -> Mapping[str, str]:
    return {
        "output_dir": str(result.output_dir),
        "export_path": str(result.export_path),
        "report_path": str(result.report_path),
        "metadata_path": str(result.metadata_path),
    }


def _benchmark_command(run_dir: Path, *, args: argparse.Namespace) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "scripts/compare_reranking.py",
        "--dataset",
        args.benchmark_dataset,
        "--algorithm",
        "cbdr",
        "--cbdr-answer-provider",
        "lmstudio",
        "--cbdr-base-artifact",
        str(run_dir / "artifacts/query_answerability.joblib"),
        "--cbdr-context-artifact",
        str(run_dir / "artifacts/query_context_answerability.joblib"),
        "--lmstudio-model",
        str(args.lmstudio_model or "$LMSTUDIO_MODEL"),
        "--output",
        str(run_dir / "benchmark/cbdr_report.json"),
        "--checkpoint-output",
        str(run_dir / "benchmark/cbdr_checkpoint.jsonl"),
        "--allow-live",
    ]
    if args.benchmark_cache_dir is not None:
        parts.extend(["--cache-dir", str(args.benchmark_cache_dir)])
    if args.benchmark_candidates is not None:
        parts.extend(["--candidates", str(args.benchmark_candidates)])
    if args.benchmark_max_cases is not None:
        parts.extend(["--max-cases", str(args.benchmark_max_cases)])
    parts.extend(["--top-k", str(args.benchmark_top_k)])
    return " ".join(parts)


def _run_benchmark(
    *,
    args: argparse.Namespace,
    output: Path,
    checkpoint_output: Path,
    base_artifact: Path,
    context_artifact: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts/compare_reranking.py"),
        "--dataset",
        args.benchmark_dataset,
        "--algorithm",
        "cbdr",
        "--cbdr-answer-provider",
        "lmstudio",
        "--cbdr-base-artifact",
        str(base_artifact),
        "--cbdr-context-artifact",
        str(context_artifact),
        "--cbdr-skip-threshold",
        str(args.skip_threshold),
        "--cbdr-device",
        args.device,
        "--cbdr-max-length",
        str(args.max_length),
        "--lmstudio-max-tokens",
        str(args.lmstudio_max_tokens),
        "--top-k",
        str(args.benchmark_top_k),
        "--output",
        str(output),
        "--checkpoint-output",
        str(checkpoint_output),
        "--allow-live",
    ]
    if args.lmstudio_model is not None:
        command.extend(["--lmstudio-model", args.lmstudio_model])
    if args.lmstudio_base_url is not None:
        command.extend(["--lmstudio-base-url", args.lmstudio_base_url])
    if args.lmstudio_api_key is not None:
        command.extend(["--lmstudio-api-key", args.lmstudio_api_key])
    if args.benchmark_cache_dir is not None:
        command.extend(["--cache-dir", str(args.benchmark_cache_dir)])
    if args.benchmark_candidates is not None:
        command.extend(["--candidates", str(args.benchmark_candidates)])
    if args.benchmark_max_cases is not None:
        command.extend(["--max-cases", str(args.benchmark_max_cases)])
    if args.cache_dir is not None:
        command.extend(["--cbdr-cache-dir", args.cache_dir])
    if args.local_files_only:
        command.append("--cbdr-local-files-only")
    if args.allow_truncation:
        command.append("--cbdr-allow-truncation")
    subprocess.run(command, cwd=ROOT, check=True)


def _require_input_file(path: Path, option_name: str) -> None:
    if not path.exists():
        raise SystemExit(f"{option_name} file does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"{option_name} must be a file: {path}")


def _build_qa_raw_files(
    *,
    raw_dir: Path,
    query_raw: Path,
    query_context_raw: Path,
    args: argparse.Namespace,
) -> None:
    if args.resume_generation and query_raw.exists() and query_context_raw.exists():
        return
    command = [
        sys.executable,
        str(ROOT / "scripts/build_qa_confidence_raw_dataset.py"),
        "--source",
        args.qa_source,
        "--dataset-name",
        args.qa_dataset_name,
        "--dataset-config",
        args.qa_dataset_config,
        "--split",
        args.qa_split,
        "--output-dir",
        str(raw_dir),
        "--max-source-items",
        str(args.qa_max_source_items),
        "--max-query-items",
        str(args.qa_max_query_items),
        "--max-query-context-items",
        str(args.qa_max_query_context_items),
        "--max-context-chars",
        str(args.max_context_chars),
    ]
    if args.qa_input_jsonl is not None:
        command.extend(["--input-jsonl", str(args.qa_input_jsonl)])
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, cwd=ROOT, check=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
