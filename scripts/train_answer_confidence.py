#!/usr/bin/env python
"""Generate answer-confidence labels with a live model and train the scorer.

Turnkey path from ``scripts/build_answer_confidence_training_data.py`` output
to the ``.joblib`` artifact that ``scripts/compare_reranking.py
--algorithm answer_confidence`` consumes:

1. Generation: for every row the live model answers ``query`` from
   ``context`` (same answer prompt the reranker uses at inference), and the
   row is labeled by whether the answer matches ``gold_answer``.
2. Training: features from a frozen encoder + LightGBM + sigmoid calibration
   via ``train_confidence_scorer``; test-split metrics are printed.

For a comparable benchmark row, generate with the SAME deployment the
benchmark uses for live calls (the README table used ``gpt-5.4-nano``): the
estimator should be trained on answers from the model it will score.

Model access follows scripts/compare_reranking.py: set
``RANKSMITH_OPENAI_BASE_URL`` (+ optional ``RANKSMITH_OPENAI_MODEL`` /
``RANKSMITH_OPENAI_API_KEY``) for any OpenAI-compatible endpoint such as
LM Studio, otherwise Azure OpenAI environment variables are required.
Requires the ``confidence-train`` extra (torch, transformers, lightgbm,
scikit-learn, joblib).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Quality gate: the committed spec run saw test roc_auc 0.875 with 500 rows
# and 0.333 (worse than random) with 100 rows. Below this bound the artifact
# is not benchmark-worthy.
MIN_TEST_ROC_AUC = 0.6


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    _load_env_file(args.env_file)
    generated_path = args.workdir / "generated.jsonl"
    if args.skip_generation:
        if not generated_path.is_file():
            raise SystemExit(
                f"--skip-generation requires an existing {generated_path}."
            )
        print(f"Skipping generation, reusing {generated_path}.", file=sys.stderr)
    else:
        _run_generation(args, generated_path)
    _run_training(args, generated_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate answer-confidence labels with a live model, then train "
            "the scorer artifact."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "AnswerGenerationSample JSONL, e.g. the output of "
            "scripts/build_answer_confidence_training_data.py."
        ),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help=(
            "Working directory for generated.jsonl and training outputs "
            "(features, report, manifests)."
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Export path for the trained scorer artifact (.joblib).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Optional cap on generated rows (smoke runs).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted generation run (skips completed row ids).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing generated.jsonl instead of failing.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reuse an existing generated.jsonl and only run training.",
    )
    parser.add_argument(
        "--encoder",
        default="bert-base-uncased",
        help="Hugging Face encoder name or local path for feature extraction.",
    )
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer name or path when it differs from --encoder.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Encoder max token length.",
    )
    parser.add_argument(
        "--allow-truncation",
        action="store_true",
        help=(
            "Allow the encoder to truncate long context+answer pairs. SQuAD "
            "contexts regularly exceed 256 tokens, so this is required for "
            "the documented runbook; the benchmark loads the artifact with "
            "truncation allowed as well."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help="Path to a .env file. Existing process environment values win.",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Required because generation sends live model requests.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise SystemExit(f"--input file does not exist: {args.input}")
    if args.max_items is not None and args.max_items < 1:
        raise SystemExit("--max-items must be greater than 0.")
    if args.max_length < 1:
        raise SystemExit("--max-length must be greater than 0.")
    if args.skip_generation and (args.resume or args.overwrite):
        raise SystemExit(
            "--skip-generation cannot be combined with --resume/--overwrite."
        )
    if not args.skip_generation and not args.allow_live:
        raise SystemExit("Refusing live model calls without --allow-live.")
    if args.artifact.suffix != ".joblib":
        raise SystemExit("--artifact must end in .joblib.")


def _run_generation(args: argparse.Namespace, generated_path: Path) -> None:
    from ranksmith.confidence_generation import (
        AnswerGenerationConfig,
        generate_answer_confidence_dataset,
    )

    provider = _CallCountingProvider(_build_provider())
    args.workdir.mkdir(parents=True, exist_ok=True)
    result = generate_answer_confidence_dataset(
        AnswerGenerationConfig(
            input_path=args.input,
            output_path=generated_path,
            provider=provider,
            overwrite=args.overwrite,
            resume=args.resume,
            max_items=args.max_items,
        )
    )
    print(
        f"Generation finished: {result.generated_count} generated, "
        f"{result.skipped_count} skipped, {result.positive_count} positive, "
        f"{result.negative_count} negative "
        f"({provider.call_count} model calls).",
        file=sys.stderr,
    )
    if result.positive_count == 0 or result.negative_count == 0:
        raise SystemExit(
            "Training needs both labels but generation produced "
            f"{result.positive_count} positive / {result.negative_count} "
            "negative rows. Check the model quality or the training data mix."
        )


def _run_training(args: argparse.Namespace, generated_path: Path) -> None:
    from ranksmith.confidence_training import (
        ConfidenceTrainingConfig,
        train_confidence_scorer,
    )

    result = train_confidence_scorer(
        ConfidenceTrainingConfig(
            task_type="answer_confidence",
            dataset_path=generated_path,
            output_dir=args.workdir / "training",
            export_path=args.artifact,
            encoder_name=args.encoder,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
            allow_truncation=args.allow_truncation,
            seed=args.seed,
        )
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    test_metrics = report["test"]
    print(f"Training report: {result.report_path}", file=sys.stderr)
    print(
        "Test split (calibrated): "
        + ", ".join(
            f"{name}={test_metrics[name]:.4f}"
            for name in ("roc_auc", "average_precision", "brier_score")
            if name in test_metrics
        ),
        file=sys.stderr,
    )
    print(f"Artifact exported: {result.export_path}", file=sys.stderr)
    roc_auc = float(test_metrics["roc_auc"])
    if roc_auc < MIN_TEST_ROC_AUC:
        raise SystemExit(
            f"Artifact quality gate failed: test roc_auc {roc_auc:.4f} < "
            f"{MIN_TEST_ROC_AUC}. Do not benchmark with this artifact; add "
            "training rows (the spec run needed ~500) or inspect "
            "generated.jsonl labels."
        )


class _CallCountingProvider:
    """Counts provider calls; usage metadata is endpoint-dependent."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.call_count = 0

    def complete(self, request: Any) -> Any:
        self.call_count += 1
        return self._inner.complete(request)


def _build_provider() -> Any:
    base_url = os.getenv("RANKSMITH_OPENAI_BASE_URL")
    if base_url:
        return _openai_compatible_provider(base_url)
    from ranksmith.providers.azure import AzureAOAIProvider

    return AzureAOAIProvider(
        api_key=_required_env("AZURE_OPENAI_API_KEY"),
        azure_endpoint=_required_env("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=_required_env(
            "AZURE_OPENAI_LLM_DEPLOYMENT",
            fallback="AZURE_OPENAI_DEPLOYMENT",
        ),
        api_version=_env_value(
            "AZURE_OPENAI_LLM_API_VERSION",
            fallback="AZURE_OPENAI_API_VERSION",
            default="2024-08-01-preview",
        ),
        timeout=_env_float("AZURE_OPENAI_LLM_TIMEOUT"),
    )


def _openai_compatible_provider(base_url: str) -> Any:
    # Same escape hatch as scripts/compare_reranking.py: any OpenAI-compatible
    # chat endpoint (LM Studio, vLLM, ...) with RANKSMITH_OPENAI_MODEL /
    # RANKSMITH_OPENAI_API_KEY overrides.
    from openai import OpenAI

    from ranksmith.model import ModelRequest, ModelResponse

    client = OpenAI(
        base_url=base_url,
        api_key=os.getenv("RANKSMITH_OPENAI_API_KEY", "local"),
        timeout=180,
    )
    model = os.getenv("RANKSMITH_OPENAI_MODEL", "local-model")

    class _Provider:
        def complete(self, request: ModelRequest) -> ModelResponse:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": m.role, "content": m.content} for m in request.messages
                ],
                temperature=0,
            )
            return ModelResponse(content=resp.choices[0].message.content or "")

    return _Provider()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if separator == "":
            raise SystemExit(f"Invalid .env line without '=': {line}")
        key = key.strip()
        if key == "":
            raise SystemExit(f"Invalid .env line with empty key: {line}")
        os.environ.setdefault(key, _clean_env_value(value))


def _clean_env_value(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith(("'", '"')):
        quote = stripped[0]
        end = stripped.find(quote, 1)
        if end == -1:
            raise SystemExit("Invalid .env quoted value.")
        return stripped[1:end]
    return stripped.split("#", maxsplit=1)[0].strip()


def _required_env(name: str, *, fallback: str | None = None) -> str:
    value = _env_value(name, fallback=fallback)
    if value is None or value == "":
        names = name if fallback is None else f"{name} or {fallback}"
        raise SystemExit(f"Missing required environment variable: {names}")
    return value


def _env_value(
    name: str,
    *,
    fallback: str | None = None,
    default: str | None = None,
) -> str | None:
    value = os.environ.get(name)
    if value is not None and value != "":
        return value
    if fallback is not None:
        fallback_value = os.environ.get(fallback)
        if fallback_value is not None and fallback_value != "":
            return fallback_value
    return default


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


if __name__ == "__main__":
    main()
