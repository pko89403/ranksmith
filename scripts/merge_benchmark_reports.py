#!/usr/bin/env python
"""Merge per-method compare_reranking reports into one comparison report.

The documented benchmark convention runs optional methods (for example
``answer_confidence``) as separate ``scripts/compare_reranking.py``
invocations against the same cache and candidate file, then merges the
reports (see ``benchmark-results/live/*.merged.json``). This script replaces
the previously manual merge and refuses to merge runs that were not measured
on identical cases:

- benchmark identity fields must match across inputs
  (dataset, candidate file name, candidate count, top_k, case count, ...);
- every algorithm in every input must cover exactly the same query ids;
- the same algorithm may not appear in two inputs.

Fields that are neither validated nor merged (for example ``timeout``) are
taken from the first input; any differing values are recorded per input under
``merged_from`` so the provenance stays visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MUST_MATCH_FIELDS = (
    "schema_version",
    "benchmark_type",
    "dataset",
    "dataset_name",
    "candidate_strategy",
    "candidate_count",
    "case_count",
    "top_k",
)
MERGED_FIELDS = (
    "algorithm",
    "aggregate",
    "per_query",
    "call_estimates",
    "method_settings",
)


def main() -> None:
    args = _parse_args()
    if not args.overwrite and args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output} without --overwrite.")
    reports = [_load_report(path) for path in args.inputs]
    merged = _merge_reports(args.inputs, reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Merged {len(reports)} reports covering algorithms "
        f"{merged['algorithm']} into {args.output}.",
        file=sys.stderr,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge compare_reranking reports measured on identical cases "
            "into one comparison report."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Two or more compare_reranking report JSON files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("Provide at least two input reports to merge.")
    return args


def _load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to read report {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise SystemExit(f"Report {path} is not a JSON object.")
    missing = [
        key
        for key in (*MUST_MATCH_FIELDS, *MERGED_FIELDS, "candidates", "seed")
        if key not in report
    ]
    if missing:
        raise SystemExit(f"Report {path} is missing required keys: {missing}")
    return report


def _merge_reports(
    paths: Sequence[Path],
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first_path, first = paths[0], reports[0]
    for path, report in zip(paths[1:], reports[1:], strict=True):
        for field in MUST_MATCH_FIELDS:
            if report[field] != first[field]:
                raise SystemExit(
                    f"{path} and {first_path} disagree on {field!r}: "
                    f"{report[field]!r} != {first[field]!r}."
                )
        if _candidate_name(report) != _candidate_name(first):
            raise SystemExit(
                f"{path} and {first_path} used different candidate files: "
                f"{report['candidates']!r} != {first['candidates']!r}."
            )
        if (
            first["candidate_strategy"] == "oracle_plus_random"
            and report["seed"] != first["seed"]
        ):
            raise SystemExit(
                f"{path} and {first_path} used different seeds with "
                "oracle_plus_random candidates; the cases differ."
            )

    algorithms: list[str] = []
    aggregate: list[Any] = []
    per_query: list[Any] = []
    call_estimates: dict[str, Any] = {}
    method_settings: dict[str, Any] = {}
    query_ids_by_algorithm: dict[str, frozenset[str]] = {}
    for path, report in zip(paths, reports, strict=True):
        for algorithm in report["algorithm"]:
            if algorithm in algorithms:
                raise SystemExit(
                    f"Algorithm {algorithm!r} appears in more than one input "
                    f"(second occurrence in {path}); refusing to merge "
                    "duplicate measurements."
                )
            algorithms.append(algorithm)
        aggregate.extend(report["aggregate"])
        per_query.extend(report["per_query"])
        call_estimates.update(report["call_estimates"])
        method_settings.update(report["method_settings"])
        for row in report["per_query"]:
            algorithm = str(row["algorithm"])
            query_ids_by_algorithm.setdefault(algorithm, frozenset())
            query_ids_by_algorithm[algorithm] |= {str(row["query_id"])}

    reference_algorithm = algorithms[0]
    reference_ids = query_ids_by_algorithm[reference_algorithm]
    for algorithm, query_ids in query_ids_by_algorithm.items():
        if query_ids != reference_ids:
            missing = sorted(reference_ids - query_ids)[:5]
            extra = sorted(query_ids - reference_ids)[:5]
            raise SystemExit(
                f"Algorithm {algorithm!r} covers different query ids than "
                f"{reference_algorithm!r} (missing e.g. {missing}, extra "
                f"e.g. {extra}); the runs are not an identical-case "
                "comparison."
            )

    merged = dict(first)
    merged["algorithm"] = algorithms
    merged["aggregate"] = aggregate
    merged["per_query"] = per_query
    merged["call_estimates"] = call_estimates
    merged["method_settings"] = method_settings
    merged["checkpoint_output"] = None
    merged["merged_from"] = [
        {
            "file": path.name,
            "algorithm": list(report["algorithm"]),
            "divergent_fields": {
                key: report[key]
                for key in sorted(report)
                if key not in MUST_MATCH_FIELDS
                and key not in MERGED_FIELDS
                and key not in {"checkpoint_output", "merged_from"}
                and report[key] != first.get(key)
            },
        }
        for path, report in zip(paths, reports, strict=True)
    ]
    return merged


def _candidate_name(report: Mapping[str, Any]) -> str | None:
    candidates = report["candidates"]
    if candidates is None:
        return None
    return Path(str(candidates)).name


if __name__ == "__main__":
    main()
