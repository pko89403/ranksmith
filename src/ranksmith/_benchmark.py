from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ranksmith._metrics import mrr_at_k, ndcg_at_k, recall_at_k

SCHEMA_VERSION = 1
CandidateStrategy = Literal["candidate_file", "oracle_plus_random"]


@dataclass(frozen=True)
class BenchmarkDocument:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class BenchmarkCase:
    fixture_id: str
    dataset: str
    source: str
    license: str
    query_id: str
    query: str
    documents: tuple[BenchmarkDocument, ...]
    qrels: Mapping[str, int]
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class EvaluationResult:
    fixture_id: str
    query_id: str
    algorithm: str
    ranked_ids: tuple[str, ...]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class AggregateResult:
    algorithm: str
    case_count: int
    metrics: Mapping[str, float]


def load_fixture_cases(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if line.strip() == "":
            continue
        data = _load_json_object(line, path=path, line_number=line_number)
        cases.append(_case_from_fixture_dict(data, path=path, line_number=line_number))
    if not cases:
        raise ValueError(f"Fixture has no cases: {path}")
    return cases


def load_beir_cases(
    cache_dir: Path,
    *,
    split: str,
    candidates_path: Path | None,
    candidate_strategy: CandidateStrategy = "candidate_file",
    candidate_count: int = 20,
    max_cases: int | None = None,
    seed: int = 13,
) -> list[BenchmarkCase]:
    _validate_positive("candidate_count", candidate_count)
    if max_cases is not None:
        _validate_positive("max_cases", max_cases)

    corpus_path = cache_dir / "corpus.jsonl"
    queries_path = cache_dir / "queries.jsonl"
    qrels_path = cache_dir / "qrels" / f"{split}.tsv"
    corpus = _load_beir_corpus(corpus_path)
    queries = _load_beir_queries(queries_path)
    qrels = _load_beir_qrels(qrels_path)

    if candidate_strategy == "candidate_file":
        if candidates_path is None:
            raise ValueError(
                "BEIR benchmark mode requires --candidates. "
                "Use --candidate-strategy oracle_plus_random only for diagnostics."
            )
        candidates = _load_candidate_file(candidates_path)
    else:
        candidates = _build_oracle_plus_random_candidates(
            corpus=corpus,
            qrels=qrels,
            candidate_count=candidate_count,
            seed=seed,
        )

    query_ids = (
        sorted(candidates) if candidate_strategy == "candidate_file" else sorted(qrels)
    )
    cases: list[BenchmarkCase] = []
    for query_id in query_ids:
        if query_id not in queries:
            raise ValueError(f"qrels query_id not found in queries.jsonl: {query_id}")
        if query_id not in qrels:
            raise ValueError(f"candidate query_id not found in qrels: {query_id}")
        document_ids = candidates.get(query_id)
        if document_ids is None:
            continue
        if not document_ids:
            raise ValueError(f"No candidate documents for query_id={query_id}")
        documents = tuple(
            _document_for_id(corpus, document_id) for document_id in document_ids
        )
        cases.append(
            BenchmarkCase(
                fixture_id=f"beir-scifact-{split}-{query_id}",
                dataset=f"BEIR/SciFact {split}",
                source=f"cache:{cache_dir}",
                license="See upstream SciFact/BEIR license metadata.",
                query_id=query_id,
                query=queries[query_id],
                documents=documents,
                qrels=qrels[query_id],
            )
        )
        if max_cases is not None and len(cases) >= max_cases:
            break

    if not cases:
        raise ValueError(
            f"No benchmark cases were built from {cache_dir} "
            f"with candidate_strategy={candidate_strategy}."
        )
    return cases


def evaluate_ranked_ids(
    *,
    case: BenchmarkCase,
    algorithm: str,
    ranked_ids: Sequence[str],
    top_k: int,
) -> EvaluationResult:
    _validate_positive("top_k", top_k)
    return EvaluationResult(
        fixture_id=case.fixture_id,
        query_id=case.query_id,
        algorithm=algorithm,
        ranked_ids=tuple(ranked_ids),
        metrics={
            f"ndcg@{top_k}": ndcg_at_k(ranked_ids, case.qrels, top_k),
            f"mrr@{top_k}": mrr_at_k(ranked_ids, case.qrels, top_k),
            f"recall@{top_k}": recall_at_k(ranked_ids, case.qrels, top_k),
        },
    )


def aggregate_evaluations(
    evaluations: Sequence[EvaluationResult],
) -> list[AggregateResult]:
    grouped: dict[str, list[EvaluationResult]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[evaluation.algorithm].append(evaluation)

    aggregates: list[AggregateResult] = []
    for algorithm, algorithm_evaluations in sorted(grouped.items()):
        metric_names = sorted(algorithm_evaluations[0].metrics)
        aggregates.append(
            AggregateResult(
                algorithm=algorithm,
                case_count=len(algorithm_evaluations),
                metrics={
                    name: sum(result.metrics[name] for result in algorithm_evaluations)
                    / len(algorithm_evaluations)
                    for name in metric_names
                },
            )
        )
    return aggregates


def evaluation_to_dict(evaluation: EvaluationResult) -> dict[str, object]:
    return {
        "fixture_id": evaluation.fixture_id,
        "query_id": evaluation.query_id,
        "algorithm": evaluation.algorithm,
        "ranked_ids": list(evaluation.ranked_ids),
        "metrics": dict(evaluation.metrics),
    }


def aggregate_to_dict(aggregate: AggregateResult) -> dict[str, object]:
    return {
        "algorithm": aggregate.algorithm,
        "case_count": aggregate.case_count,
        "metrics": dict(aggregate.metrics),
    }


def _case_from_fixture_dict(
    data: Mapping[str, object],
    *,
    path: Path,
    line_number: int,
) -> BenchmarkCase:
    try:
        schema_version = _required_int(data, "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={schema_version}")
        raw_documents = _required_list(data, "documents")
        documents = tuple(_fixture_document(item) for item in raw_documents)
        qrels = {
            str(document_id): _object_to_int(score, key=f"qrels[{document_id}]")
            for document_id, score in _required_mapping(data, "qrels").items()
        }
        return BenchmarkCase(
            schema_version=schema_version,
            fixture_id=_required_str(data, "fixture_id"),
            dataset=_required_str(data, "dataset"),
            source=_required_str(data, "source"),
            license=_required_str(data, "license"),
            query_id=_required_str(data, "query_id"),
            query=_required_str(data, "query"),
            documents=documents,
            qrels=qrels,
        )
    except (TypeError, ValueError, KeyError) as exc:
        message = f"Invalid fixture case at {path}:{line_number}: {exc}"
        raise ValueError(message) from exc


def _fixture_document(item: object) -> BenchmarkDocument:
    if not isinstance(item, dict):
        raise TypeError("document must be an object")
    data = cast(Mapping[str, object], item)
    return BenchmarkDocument(
        id=_required_str(data, "id"),
        title=_required_str(data, "title"),
        text=_required_str(data, "text"),
    )


def _load_beir_corpus(path: Path) -> dict[str, BenchmarkDocument]:
    corpus: dict[str, BenchmarkDocument] = {}
    for line_number, data in _read_jsonl(path):
        document_id = _required_str(data, "_id")
        if document_id in corpus:
            raise ValueError(f"Duplicate corpus document id at {path}:{line_number}")
        corpus[document_id] = BenchmarkDocument(
            id=document_id,
            title=str(data.get("title", "")),
            text=_required_str(data, "text"),
        )
    if not corpus:
        raise ValueError(f"BEIR corpus has no documents: {path}")
    return corpus


def _load_beir_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for line_number, data in _read_jsonl(path):
        query_id = _required_str(data, "_id")
        if query_id in queries:
            raise ValueError(f"Duplicate query id at {path}:{line_number}")
        queries[query_id] = _required_str(data, "text")
    if not queries:
        raise ValueError(f"BEIR queries file has no queries: {path}")
    return queries


def _load_beir_qrels(path: Path) -> dict[str, dict[str, int]]:
    _require_file(path)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if line.strip() == "":
            continue
        columns = line.split()
        if line_number == 1 and _looks_like_header(columns):
            continue
        if len(columns) < 3:
            raise ValueError(f"Invalid qrels row at {path}:{line_number}")
        query_id, document_id, score_text = columns[0], columns[1], columns[2]
        score = _parse_int(score_text, path=path, line_number=line_number)
        if score > 0:
            qrels[query_id][document_id] = score
    if not qrels:
        raise ValueError(f"BEIR qrels file has no positive qrels: {path}")
    return dict(qrels)


def _load_candidate_file(path: Path) -> dict[str, tuple[str, ...]]:
    _require_file(path)
    candidates: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if line.strip() == "":
            continue
        columns = line.split()
        if line_number == 1 and _looks_like_header(columns):
            continue
        if len(columns) < 2:
            raise ValueError(f"Invalid candidates row at {path}:{line_number}")
        query_id, document_id = columns[0], columns[1]
        if document_id in seen[query_id]:
            raise ValueError(
                f"Duplicate candidate document {document_id} "
                f"for query_id={query_id} at {path}:{line_number}"
            )
        seen[query_id].add(document_id)
        candidates[query_id].append(document_id)
    if not candidates:
        raise ValueError(f"Candidate file has no candidates: {path}")
    return {
        query_id: tuple(document_ids) for query_id, document_ids in candidates.items()
    }


def _build_oracle_plus_random_candidates(
    *,
    corpus: Mapping[str, BenchmarkDocument],
    qrels: Mapping[str, Mapping[str, int]],
    candidate_count: int,
    seed: int,
) -> dict[str, tuple[str, ...]]:
    rng = random.Random(seed)
    corpus_ids = sorted(corpus)
    candidates: dict[str, tuple[str, ...]] = {}
    for query_id, query_qrels in qrels.items():
        relevant_ids = sorted(
            document_id for document_id, score in query_qrels.items() if score > 0
        )
        if len(relevant_ids) > candidate_count:
            raise ValueError(
                f"candidate_count={candidate_count} is smaller than the number "
                f"of relevant documents for query_id={query_id}."
            )
        non_relevant_ids = [
            document_id for document_id in corpus_ids if document_id not in query_qrels
        ]
        rng.shuffle(non_relevant_ids)
        candidates[query_id] = tuple(
            relevant_ids + non_relevant_ids[: candidate_count - len(relevant_ids)]
        )
    return candidates


def _document_for_id(
    corpus: Mapping[str, BenchmarkDocument],
    document_id: str,
) -> BenchmarkDocument:
    try:
        return corpus[document_id]
    except KeyError as exc:
        message = f"Candidate document id not found in corpus: {document_id}"
        raise ValueError(message) from exc


def _read_jsonl(path: Path) -> list[tuple[int, Mapping[str, object]]]:
    _require_file(path)
    rows: list[tuple[int, Mapping[str, object]]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if line.strip() == "":
            continue
        rows.append(
            (line_number, _load_json_object(line, path=path, line_number=line_number))
        )
    return rows


def _load_json_object(
    line: str,
    *,
    path: Path,
    line_number: int,
) -> Mapping[str, object]:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}:{line_number}")
    return cast(Mapping[str, object], data)


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(data: Mapping[str, object], key: str) -> int:
    value = data[key]
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_list(data: Mapping[str, object], key: str) -> list[object]:
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _required_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = data[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return cast(Mapping[str, object], value)


def _parse_int(value: str, *, path: Path, line_number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Expected integer score at {path}:{line_number}") from exc


def _object_to_int(value: object, *, key: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _looks_like_header(columns: Sequence[str]) -> bool:
    header_names = {"query-id", "query_id", "corpus-id", "score"}
    return any(column.lower() in header_names for column in columns)


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be greater than 0")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Required benchmark file does not exist: {path}")
