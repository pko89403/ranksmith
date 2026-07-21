#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NO_ANSWER = "__NO_ANSWER__"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = _load_rows(args)
    examples = _build_examples(
        rows,
        max_query_items=args.max_query_items,
        max_query_context_items=args.max_query_context_items,
        max_context_chars=args.max_context_chars,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = output_dir / "query_answerability_raw.jsonl"
    context_path = output_dir / "query_context_answerability_raw.jsonl"
    manifest_path = output_dir / "dataset_manifest.json"
    _ensure_outputs_writable(
        (query_path, context_path, manifest_path),
        overwrite=args.overwrite,
    )
    query_rows = [example.query_row for example in examples if example.query_row]
    context_rows = [
        context_row
        for example in examples
        for context_row in (example.positive_context_row, example.negative_context_row)
        if context_row is not None
    ]
    _write_jsonl(query_path, query_rows)
    _write_jsonl(
        context_path,
        context_rows,
    )
    manifest = {
        "source": args.source,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "max_source_items": args.max_source_items,
        "max_query_items": args.max_query_items,
        "max_query_context_items": args.max_query_context_items,
        "max_context_chars": args.max_context_chars,
        "seed": args.seed,
        "query_answerability_count": len(query_rows),
        "query_context_answerability_count": len(context_rows),
        "query_answerability_raw": str(query_path),
        "query_context_answerability_raw": str(context_path),
        "label_policy": {
            "query_answerability": "gold answer aliases from QA dataset",
            "query_context_positive": "evidence context from same QA example",
            "query_context_negative": (
                "TF-IDF retrieved evidence context from a different QA example "
                "that does not contain the current answer aliases"
            ),
        },
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build confidence raw datasets from TriviaQA-style QA examples with "
            "positive evidence and retrieved negative contexts."
        )
    )
    parser.add_argument("--source", choices=("triviaqa",), default="triviaqa")
    parser.add_argument("--dataset-name", default="mandarjoshi/trivia_qa")
    parser.add_argument("--dataset-config", default="rc")
    parser.add_argument("--split", default="train[:20000]")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-source-items", type=int, default=20000)
    parser.add_argument("--max-query-items", type=int, default=5000)
    parser.add_argument("--max-query-context-items", type=int, default=5000)
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_query_items < 30:
        parser.error("--max-query-items must be >= 30")
    if args.max_query_context_items < 30:
        parser.error("--max-query-context-items must be >= 30")
    if args.max_source_items < 2:
        parser.error("--max-source-items must be >= 2")
    if args.max_query_context_items % 2 != 0:
        parser.error("--max-query-context-items must be even for balanced labels")
    if args.max_context_chars < 1:
        parser.error("--max-context-chars must be >= 1")
    return args


@dataclass(frozen=True)
class _RawExample:
    id: str
    question: str
    answers: list[str]
    positive_context: str


@dataclass(frozen=True)
class _BuiltExample:
    query_row: Mapping[str, Any]
    positive_context_row: Mapping[str, Any] | None
    negative_context_row: Mapping[str, Any] | None


def _load_rows(args: argparse.Namespace) -> list[Mapping[str, Any]]:
    if args.input_jsonl is not None:
        return _read_jsonl(args.input_jsonl, max_items=args.max_source_items)
    try:
        datasets = __import__("datasets")
    except ImportError as exc:
        raise SystemExit(
            "HuggingFace datasets is required for direct dataset loading. "
            "Run with: uv run --with datasets python "
            "scripts/build_qa_confidence_raw_dataset.py ..."
        ) from exc
    dataset = datasets.load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
    )
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(dataset):
        if index >= args.max_source_items:
            break
        rows.append(dict(row))
    return rows


def _build_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_query_items: int,
    max_query_context_items: int,
    max_context_chars: int,
    seed: int,
) -> list[_BuiltExample]:
    raw_examples: list[_RawExample] = []
    for row in rows:
        example = _parse_triviaqa_row(row)
        if example is None:
            continue
        if len(example.positive_context) > max_context_chars:
            continue
        raw_examples.append(example)
    if len(raw_examples) < 2:
        raise SystemExit("Need at least two usable QA examples.")

    rng = random.Random(seed)
    rng.shuffle(raw_examples)
    required_context_pairs = max_query_context_items // 2
    required_examples = max(max_query_items, required_context_pairs)
    selected = raw_examples[:required_examples]
    if len(selected) < required_examples:
        raise SystemExit(
            f"Only {len(selected)} usable examples found; "
            f"requested {required_examples}."
        )

    negative_by_id = _retrieve_negative_contexts(
        selected,
        max_pairs=required_context_pairs,
    )
    built: list[_BuiltExample] = []
    for index, example in enumerate(selected):
        include_query = index < max_query_items
        include_context = index < required_context_pairs
        negative = negative_by_id.get(example.id)
        built.append(
            _BuiltExample(
                query_row=(
                    {
                        "id": f"triviaqa-q-{example.id}",
                        "query": example.question,
                        "gold_answer": example.answers,
                        "source": "triviaqa",
                        "group_id": f"triviaqa-{example.id}",
                        "metadata": {"qa_id": example.id},
                    }
                    if include_query
                    else {}
                ),
                positive_context_row=(
                    {
                        "id": f"triviaqa-qc-pos-{example.id}",
                        "query": example.question,
                        "context": example.positive_context,
                        "gold_answer": example.answers,
                        "source": "triviaqa",
                        "group_id": f"triviaqa-{example.id}",
                        "metadata": {
                            "qa_id": example.id,
                            "context_label": "positive",
                        },
                    }
                    if include_context
                    else None
                ),
                negative_context_row=(
                    {
                        "id": f"triviaqa-qc-neg-{example.id}",
                        "query": example.question,
                        "context": negative.positive_context,
                        "gold_answer": NO_ANSWER,
                        "source": "triviaqa",
                        "group_id": f"triviaqa-{example.id}",
                        "metadata": {
                            "qa_id": example.id,
                            "negative_context_qa_id": negative.id,
                            "context_label": "negative_retrieved_tfidf",
                        },
                    }
                    if include_context and negative is not None
                    else None
                ),
            )
        )

    query_count = sum(1 for example in built if example.query_row)
    context_count = sum(
        1
        for example in built
        for row in (example.positive_context_row, example.negative_context_row)
        if row is not None
    )
    if query_count != max_query_items:
        raise SystemExit(
            f"Built {query_count} query rows; requested {max_query_items}."
        )
    if context_count != max_query_context_items:
        raise SystemExit(
            f"Built {context_count} query-context rows; "
            f"requested {max_query_context_items}. "
            "Increase source rows or lower the requested context count."
        )
    return built


def _retrieve_negative_contexts(
    examples: Sequence[_RawExample],
    *,
    max_pairs: int,
) -> dict[str, _RawExample]:
    try:
        sklearn_text = importlib.import_module("sklearn.feature_extraction.text")
        sklearn_neighbors = importlib.import_module("sklearn.neighbors")
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required for retrieved negatives. "
            "Install dev dependencies or run through `uv run`."
        ) from exc

    tfidf_vectorizer = sklearn_text.TfidfVectorizer
    nearest_neighbors = sklearn_neighbors.NearestNeighbors
    contexts = [example.positive_context for example in examples]
    vectorizer = tfidf_vectorizer(
        lowercase=True,
        stop_words="english",
        max_features=100_000,
        ngram_range=(1, 2),
    )
    context_matrix = vectorizer.fit_transform(contexts)
    query_matrix = vectorizer.transform([example.question for example in examples])
    neighbors = nearest_neighbors(
        algorithm="brute",
        metric="cosine",
        n_neighbors=min(len(examples), 64),
    )
    neighbors.fit(context_matrix)
    _, neighbor_indices = neighbors.kneighbors(query_matrix[:max_pairs])

    result: dict[str, _RawExample] = {}
    for index, example in enumerate(examples[:max_pairs]):
        for candidate_index in neighbor_indices[index]:
            candidate = examples[int(candidate_index)]
            if _is_valid_negative_candidate(example, candidate):
                result[example.id] = candidate
                break
        if example.id in result:
            continue
        for candidate in examples:
            if _is_valid_negative_candidate(example, candidate):
                result[example.id] = candidate
                break
    return result


def _is_valid_negative_candidate(
    example: _RawExample,
    candidate: _RawExample,
) -> bool:
    return candidate.id != example.id and not _contains_any_answer(
        candidate.positive_context,
        example.answers,
    )


def _contains_any_answer(context: str, answers: Sequence[str]) -> bool:
    context_tokens = _tokens_for_match(context)
    for answer in answers:
        answer_tokens = _tokens_for_match(answer)
        if answer_tokens and _contains_token_sequence(context_tokens, answer_tokens):
            return True
    return False


def _contains_token_sequence(
    context_tokens: Sequence[str],
    answer_tokens: Sequence[str],
) -> bool:
    if len(answer_tokens) > len(context_tokens):
        return False
    window_size = len(answer_tokens)
    for index in range(len(context_tokens) - window_size + 1):
        if list(context_tokens[index : index + window_size]) == list(answer_tokens):
            return True
    return False


def _tokens_for_match(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _parse_triviaqa_row(row: Mapping[str, Any]) -> _RawExample | None:
    question = _text(row.get("question"))
    if question is None:
        return None
    row_id = _text(row.get("question_id")) or _text(row.get("id"))
    if row_id is None:
        return None
    answers = _answer_aliases(row.get("answer"))
    if not answers:
        return None
    context = _best_context(row, answers=answers)
    if context is None:
        return None
    return _RawExample(
        id=row_id,
        question=question,
        answers=answers,
        positive_context=context,
    )


def _answer_aliases(answer: object) -> list[str]:
    if not isinstance(answer, Mapping):
        return []
    values: list[str] = []
    for key in ("value", "normalized_value"):
        value = _text(answer.get(key))
        if value is not None:
            values.append(value)
    for key in ("aliases", "normalized_aliases"):
        aliases = answer.get(key)
        if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
            for alias in aliases:
                value = _text(alias)
                if value is not None:
                    values.append(value)
    return _dedupe(values)


def _best_context(row: Mapping[str, Any], *, answers: Sequence[str]) -> str | None:
    candidates: list[str] = []
    for collection_name in ("entity_pages", "search_results"):
        collection = row.get(collection_name)
        if isinstance(collection, Mapping):
            candidates.extend(_contexts_from_columnar_collection(collection))
        if isinstance(collection, Sequence) and not isinstance(
            collection, (str, bytes)
        ):
            candidates.extend(_contexts_from_records(collection))
    for context in candidates:
        if _contains_any_answer(context, answers):
            return context
    return None


def _contexts_from_columnar_collection(collection: Mapping[str, Any]) -> list[str]:
    contexts = collection.get("wiki_context") or collection.get("search_context")
    titles = collection.get("title")
    if not isinstance(contexts, Sequence) or isinstance(contexts, (str, bytes)):
        return []
    title_values = titles if isinstance(titles, Sequence) else []
    result: list[str] = []
    for index, context in enumerate(contexts):
        text = _text(context)
        if text is None:
            continue
        title = ""
        if index < len(title_values):
            raw_title = _text(title_values[index])
            if raw_title is not None:
                title = raw_title + "\n\n"
        result.append(title + text)
    return result


def _contexts_from_records(records: Sequence[object]) -> list[str]:
    result: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        text = _text(record.get("wiki_context")) or _text(record.get("search_context"))
        if text is None:
            continue
        title = _text(record.get("title"))
        result.append(f"{title}\n\n{text}" if title is not None else text)
    return result


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _read_jsonl(path: Path, *, max_items: int) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if len(rows) >= max_items:
            break
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise SystemExit(f"line {line_number} must be a JSON object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ensure_outputs_writable(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise SystemExit(
            f"output already exists: {formatted}. Use --overwrite to replace it."
        )


if __name__ == "__main__":
    raise SystemExit(main())
