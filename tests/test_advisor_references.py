from __future__ import annotations

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
    names: list[str] = []
    for raw in body.split("#")[0].split(","):
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
