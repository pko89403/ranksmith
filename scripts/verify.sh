#!/usr/bin/env bash
set -euo pipefail

uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build
