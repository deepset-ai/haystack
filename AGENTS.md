# Haystack Guidelines for AI Agents

## Environment

Haystack uses **Hatch** for environment and dependency management.

Do not run `python` or `pip` directly.

Before running code on this project, you must be able to run `hatch --version` and get a correct output.

If not, ask the user where Hatch is or if they want to install it. For installation instructions, refer to https://hatch.pypa.io/latest/install/#installation.

### Run scripts with test dependencies

hatch -e test run python SCRIPT.py

### Open a shell with test dependencies

hatch -e test shell

### Install temporary dependencies (for experiments only)

uv pip install PACKAGE

### Delete the environment

hatch env prune

## Tests

Tests run via Hatch and support pytest arguments.

Prefer running tests on a specific module or using `-k`, since the full suite is large.

### Run unit tests

hatch run test:unit

### Run integration tests

hatch run test:integration

## Quality Checks

### Type checking with mypy
hatch run test:types

To fix type issues, avoid `type: ignore`, casts, or assertions when possible. If they are necessary, explain why.

### Format and lint
hatch run fmt

## Release Notes

Every user-facing PR (not docs, not CI) must include a release note:

hatch run release-note SHORT_DESCRIPTION

Edit the generated file in `releasenotes/notes/`.
