from __future__ import annotations

import inspect
import re
from pathlib import Path

import ranksmith

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "ranksmith-advisor"

_EXAMPLE_RE = re.compile(r"examples/[\w./-]+\.py")
_IMPORT_RE = re.compile(r"from ranksmith import\s+(?:\(([^)]*)\)|([^\n]+))")


def _skill_markdown_files() -> list[Path]:
    files = sorted(SKILL_DIR.glob("*.md"))
    assert files, f"no skill markdown found in {SKILL_DIR}"
    return files


def _imported_names(body: str) -> list[str]:
    # Strip inline comments per line before splitting on commas, so a symbol
    # after an inline comment in a multi-line import is still validated.
    no_comments = "\n".join(line.split("#")[0] for line in body.splitlines())
    names: list[str] = []
    for raw in no_comments.split(","):
        name = raw.strip().split(" as ")[0].strip()
        if name:
            names.append(name)
    return names


def test_referenced_examples_exist() -> None:
    missing: list[str] = []
    for path in _skill_markdown_files():
        text = path.read_text(encoding="utf-8")
        for rel in sorted(set(_EXAMPLE_RE.findall(text))):
            if not (ROOT / rel).is_file():
                missing.append(f"{path.name}: {rel}")
    assert not missing, f"referenced example files do not exist: {missing}"


def test_snippets_use_public_ranksmith_symbols() -> None:
    public = set(ranksmith.__all__)
    unknown: list[str] = []
    for path in _skill_markdown_files():
        text = path.read_text(encoding="utf-8")
        for paren_body, line_body in _IMPORT_RE.findall(text):
            for name in _imported_names(paren_body or line_body):
                if name not in public:
                    unknown.append(f"{path.name}: {name}")
    assert not unknown, f"snippets import non-public ranksmith symbols: {unknown}"


# Defaults the advisor docs (method-guide.md) tell the model to rely on. If a
# default changes in src/ranksmith, this guard fails so the advisor docs are
# updated in lock-step instead of silently drifting.
_DOCUMENTED_DEFAULTS: dict[str, dict[str, object]] = {
    "ListwiseStrategy": {
        "window_size": 20,
        "stride": 10,
        "max_document_chars": 4000,
    },
    "PairwiseStrategy": {
        "passes": 10,
        "max_document_chars": 4000,
    },
    "SetwiseStrategy": {
        "set_size": 3,
        "max_document_chars": 4000,
    },
    "TourRankStrategy": {
        "rounds": 2,
        "shuffle_seed": 13,
        "max_document_chars": 4000,
    },
    "AcuRankStrategy": {
        "target_rank": 10,
        "window_size": 20,
        "tolerance": 0.01,
        "uncertain_threshold": 10,
        "initial_pass": True,
        "max_adaptive_reranker_calls": None,
    },
}


def test_documented_strategy_defaults_match_source() -> None:
    mismatches: list[str] = []
    for cls_name, expected in _DOCUMENTED_DEFAULTS.items():
        cls = getattr(ranksmith, cls_name)
        params = inspect.signature(cls.__init__).parameters
        for param, want in expected.items():
            if param not in params:
                mismatches.append(f"{cls_name}.{param}: param missing from source")
                continue
            got = params[param].default
            if type(got) is not type(want) or got != want:
                mismatches.append(
                    f"{cls_name}.{param}: source default {got!r} != documented {want!r}"
                )
    assert not mismatches, (
        "method-guide.md documents strategy defaults that no longer match source; "
        f"update the advisor docs and this guard together: {mismatches}"
    )
