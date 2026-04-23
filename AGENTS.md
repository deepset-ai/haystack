# Haystack Guidelines for AI Agents

## Environment

Haystack uses **uv** for environment and dependency management in CI and for local development.

Do not run `python` or `pip` directly.

Before running code on this project, you must be able to run `uv --version` and get a correct output.

If not, install uv by following https://docs.astral.sh/uv/getting-started/installation/.

### Sync dependencies and run a script

uv run python SCRIPT.py

### Sync test dependencies and run a script

uv sync --group test
uv run python SCRIPT.py

### Open a shell with test dependencies

uv sync --group test
source .venv/bin/activate

### Install temporary dependencies (for experiments only)

uv pip install PACKAGE

### Delete the environment

rm -rf .venv

## Tests

Tests run via uv and support pytest arguments.

Prefer running tests on a specific module or using `-k`, since the full suite is large.

### Run unit tests

uv sync --group test
uv run pytest --cov-report xml:coverage.xml --cov="haystack" -m "not integration" test

### Run integration tests

uv sync --group test
uv run pytest --maxfail=5 -m "integration" test

## Quality Checks

### Type checking with mypy
uv sync --group test
uv run mypy --install-types --non-interactive --cache-dir=.mypy_cache/ haystack test/core/ test/marshal/ test/testing/ test/tracing/

To fix type issues, avoid `type: ignore`, casts, or assertions when possible. If they are necessary, explain why.

### Format and lint
uv sync --group dev
uv run ruff check --fix && uv run ruff format

## Release Notes

Every user-facing PR (not docs, not CI) must include a release note:

uv sync --group dev
uv run reno new SHORT_DESCRIPTION

Edit the generated file in `releasenotes/notes/`.
