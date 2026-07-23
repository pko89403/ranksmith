#!/usr/bin/env python
"""Build answer-confidence training data from SQuAD v1.1.

The ``answer_confidence`` reranker needs a scorer artifact, and the artifact
needs QA training data with gold answers. IR benchmarks such as AskUbuntu have
qrels but no gold answers, so the artifact must be trained on a QA dataset and
that domain must be reported next to any benchmark number.

Each sampled SQuAD question produces one gold-context row and
``--negatives-per-question`` BM25 hard-negative-context rows. This mirrors the
reranker's inference distribution: at inference the estimator scores answers
generated from both relevant and non-relevant candidate documents. Labels are
NOT assigned here — the generation pipeline
(``scripts/train_answer_confidence.py``) labels each row by whether the live
model's answer matches ``gold_answer``, so a hard negative that still yields a
correct answer is truthfully labeled positive.

Output rows follow the ``AnswerGenerationSample`` schema:
``{"id", "query", "context", "gold_answer", "source", "group_id", "metadata"}``.
Contexts are ``"{title}\\n\\n{text}"`` to match how
``scripts/compare_reranking.py`` joins candidate documents at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.bm25 import (  # noqa: E402
    DEFAULT_B,
    DEFAULT_K1,
    BM25Index,
    bm25_search,
    build_bm25_index,
)

SQUAD_TRAIN_URL = (
    "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/"
    "master/dataset/train-v1.1.json"
)
SQUAD_SOURCE_LABEL = "squad-v1.1-train"
# The generation pipeline rejects contexts longer than its max_context_chars
# (default 4000), and the reranking strategy rejects documents longer than its
# max_document_chars (default 4000). Keep the builder aligned with both.
DEFAULT_MAX_CONTEXT_CHARS = 4000
DEFAULT_QUESTIONS = 250
DEFAULT_NEGATIVES_PER_QUESTION = 1
DEFAULT_NEGATIVE_POOL = 10
DEFAULT_SEED = 13


@dataclass(frozen=True)
class SquadParagraph:
    id: str
    title: str
    text: str

    @property
    def context(self) -> str:
        return f"{self.title}\n\n{self.text}"


@dataclass(frozen=True)
class SquadQuestion:
    id: str
    title: str
    paragraph_id: str
    text: str
    answers: tuple[str, ...]


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    squad_path = _resolve_squad_path(args)
    paragraphs, questions = _load_squad(squad_path)
    print(
        f"Loaded {len(paragraphs)} unique paragraphs and "
        f"{len(questions)} questions from {squad_path}.",
        file=sys.stderr,
    )
    index = build_bm25_index(
        {paragraph.id: paragraph.context for paragraph in paragraphs.values()},
        k1=args.bm25_k1,
        b=args.bm25_b,
    )
    rows, stats = _build_rows(
        paragraphs=paragraphs,
        questions=questions,
        index=index,
        question_count=args.questions,
        negatives_per_question=args.negatives_per_question,
        negative_pool=args.negative_pool,
        max_context_chars=args.max_context_chars,
        seed=args.seed,
    )
    _write_jsonl(args.output, rows)
    report = {
        "source": SQUAD_SOURCE_LABEL,
        "squad_file": str(squad_path),
        "squad_sha256": _sha256(squad_path),
        "parameters": {
            "questions": args.questions,
            "negatives_per_question": args.negatives_per_question,
            "negative_pool": args.negative_pool,
            "max_context_chars": args.max_context_chars,
            "seed": args.seed,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
        },
        "statistics": stats,
    }
    report_path = _report_path(args.output)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(rows)} rows to {args.output} "
        f"({stats['gold_rows']} gold + {stats['negative_rows']} hard-negative). "
        f"Report: {report_path}",
        file=sys.stderr,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build answer-confidence training data (gold + BM25 hard-negative "
            "contexts) from SQuAD v1.1 train."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path, e.g. .benchmark-cache/answer_train.jsonl.",
    )
    parser.add_argument(
        "--squad-train",
        type=Path,
        help=(
            "Local train-v1.1.json path. Omit together with --download to "
            "fetch the official file."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            f"Download train-v1.1.json from {SQUAD_TRAIN_URL} next to --output "
            "when --squad-train is not given. Reuses an existing download."
        ),
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=DEFAULT_QUESTIONS,
        help=(
            "Number of SQuAD questions to sample. Each question emits "
            "1 + negatives-per-question rows. The training pipeline needs "
            ">= 30 rows total and both labels present after generation; "
            "the committed spec run found 500 total rows workable and "
            "100 too few."
        ),
    )
    parser.add_argument(
        "--negatives-per-question",
        type=int,
        default=DEFAULT_NEGATIVES_PER_QUESTION,
        help="BM25 hard-negative contexts per question.",
    )
    parser.add_argument(
        "--negative-pool",
        type=int,
        default=DEFAULT_NEGATIVE_POOL,
        help=(
            "How many top BM25 paragraphs (excluding the gold paragraph) to "
            "consider when picking hard negatives."
        ),
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=(
            "Skip questions whose gold or negative context exceeds this "
            "length. Must match the generation pipeline and strategy limits."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bm25-k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--bm25-b", type=float, default=DEFAULT_B)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --output and its report if they already exist.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("questions", "negatives_per_question", "negative_pool"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than 0.")
    if args.max_context_chars < 1:
        raise SystemExit("--max-context-chars must be greater than 0.")
    if args.negative_pool < args.negatives_per_question:
        raise SystemExit("--negative-pool must be at least --negatives-per-question.")
    if args.squad_train is not None and args.download:
        raise SystemExit("Pass either --squad-train or --download, not both.")
    if args.squad_train is None and not args.download:
        raise SystemExit(
            "SQuAD input is required: pass --squad-train <path> or --download."
        )
    if not args.overwrite:
        for path in (args.output, _report_path(args.output)):
            if path.exists():
                raise SystemExit(f"Refusing to overwrite {path} without --overwrite.")


def _report_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.report.json")


def _resolve_squad_path(args: argparse.Namespace) -> Path:
    if args.squad_train is not None:
        if not args.squad_train.is_file():
            raise SystemExit(f"--squad-train file does not exist: {args.squad_train}")
        return args.squad_train
    target = args.output.parent / "train-v1.1.json"
    if target.is_file():
        print(f"Reusing existing download: {target}", file=sys.stderr)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SQUAD_TRAIN_URL} -> {target}", file=sys.stderr)
    try:
        with urllib.request.urlopen(SQUAD_TRAIN_URL, timeout=120) as response:
            data = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"SQuAD download failed: {exc}") from exc
    target.write_bytes(data)
    return target


def _load_squad(
    path: Path,
) -> tuple[dict[str, SquadParagraph], list[SquadQuestion]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to read SQuAD JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        raise SystemExit(f'{path} is not SQuAD-format JSON (missing "data" list).')
    version = payload.get("version")
    if version is not None and str(version) != "1.1":
        raise SystemExit(
            f"Expected SQuAD version 1.1, found {version!r}. SQuAD 2.0 adds "
            "unanswerable questions and needs an explicit design decision."
        )

    paragraphs: dict[str, SquadParagraph] = {}
    paragraph_id_by_text: dict[str, str] = {}
    questions: list[SquadQuestion] = []
    seen_question_ids: set[str] = set()
    for article_number, article in enumerate(payload["data"], 1):
        title = _required_str(article, "title", f"data[{article_number}]")
        for paragraph_number, paragraph in enumerate(
            _required_list(article, "paragraphs", f"data[{article_number}]"), 1
        ):
            location = f"data[{article_number}].paragraphs[{paragraph_number}]"
            context = _required_str(paragraph, "context", location)
            paragraph_id = paragraph_id_by_text.get(context)
            if paragraph_id is None:
                paragraph_id = f"p{len(paragraphs):05d}"
                paragraph_id_by_text[context] = paragraph_id
                paragraphs[paragraph_id] = SquadParagraph(
                    id=paragraph_id,
                    title=title,
                    text=context,
                )
            for qa in _required_list(paragraph, "qas", location):
                question_id = _required_str(qa, "id", location)
                if question_id in seen_question_ids:
                    raise SystemExit(f"Duplicate SQuAD question id: {question_id}")
                seen_question_ids.add(question_id)
                answers = tuple(
                    dict.fromkeys(
                        answer["text"]
                        for answer in _required_list(qa, "answers", location)
                        if isinstance(answer, Mapping)
                        and isinstance(answer.get("text"), str)
                        and answer["text"].strip() != ""
                    )
                )
                questions.append(
                    SquadQuestion(
                        id=question_id,
                        title=title,
                        paragraph_id=paragraph_id,
                        text=_required_str(qa, "question", location),
                        answers=answers,
                    )
                )
    if not paragraphs or not questions:
        raise SystemExit(f"{path} contains no usable paragraphs or questions.")
    return paragraphs, questions


def _build_rows(
    *,
    paragraphs: Mapping[str, SquadParagraph],
    questions: Sequence[SquadQuestion],
    index: BM25Index,
    question_count: int,
    negatives_per_question: int,
    negative_pool: int,
    max_context_chars: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    order = list(range(len(questions)))
    random.Random(seed).shuffle(order)

    rows: list[dict[str, object]] = []
    accepted = 0
    skipped: dict[str, int] = {
        "empty_question_or_answers": 0,
        "gold_context_too_long": 0,
        "not_enough_negatives": 0,
    }
    max_context_length = 0
    for position in order:
        if accepted >= question_count:
            break
        question = questions[position]
        if question.text.strip() == "" or not question.answers:
            skipped["empty_question_or_answers"] += 1
            continue
        gold = paragraphs[question.paragraph_id]
        if len(gold.context) > max_context_chars:
            skipped["gold_context_too_long"] += 1
            continue
        negative_candidates = [
            paragraphs[document_id]
            for document_id, _ in bm25_search(
                index,
                question.text,
                top_k=negative_pool,
                exclude=(gold.id,),
            )
            if len(paragraphs[document_id].context) <= max_context_chars
        ]
        if len(negative_candidates) < negatives_per_question:
            skipped["not_enough_negatives"] += 1
            continue

        accepted += 1
        gold_answer = list(question.answers)
        rows.append(
            {
                "id": f"{question.id}::gold",
                "query": question.text,
                "context": gold.context,
                "gold_answer": gold_answer,
                "source": SQUAD_SOURCE_LABEL,
                "group_id": question.title,
                "metadata": {
                    "context_kind": "gold",
                    "paragraph_id": gold.id,
                },
            }
        )
        max_context_length = max(max_context_length, len(gold.context))
        for negative_rank, negative in enumerate(
            negative_candidates[:negatives_per_question], 1
        ):
            rows.append(
                {
                    "id": f"{question.id}::neg{negative_rank}",
                    "query": question.text,
                    "context": negative.context,
                    "gold_answer": gold_answer,
                    "source": SQUAD_SOURCE_LABEL,
                    "group_id": question.title,
                    "metadata": {
                        "context_kind": "bm25_hard_negative",
                        "paragraph_id": negative.id,
                        "bm25_rank": negative_rank,
                    },
                }
            )
            max_context_length = max(max_context_length, len(negative.context))

    if accepted < question_count:
        raise SystemExit(
            f"Only {accepted} of the requested {question_count} questions were "
            f"usable (skipped: {skipped}). Lower --questions or relax limits."
        )
    stats = {
        "questions_accepted": accepted,
        "gold_rows": accepted,
        "negative_rows": len(rows) - accepted,
        "total_rows": len(rows),
        "skipped": skipped,
        "max_context_chars_seen": max_context_length,
        "distinct_group_ids": len({str(row["group_id"]) for row in rows}),
    }
    return rows, stats


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_str(data: Mapping[str, object], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise SystemExit(f'{location}: "{key}" must be a string.')
    return value


def _required_list(data: Mapping[str, object], key: str, location: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise SystemExit(f'{location}: "{key}" must be a list.')
    return value


if __name__ == "__main__":
    main()
