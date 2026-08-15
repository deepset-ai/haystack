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

Edit the generated file in `releasenotes/notes/`. Release notes use reStructuredText formatting; see the [release notes section in CONTRIBUTING.md](CONTRIBUTING.md#release-notes) for details.

The rules below were mined from 2,746 PR review comments written by the deepset
team between 2025-07-01 and 2026-08-14, then filtered against the current source tree so that
guidance referring to APIs removed or moved in Haystack 3.0 does not survive.

They describe what reviewers actually enforce. Follow them the way you would follow a
reviewer's note: they encode reasons, not ceremony, so when a rule genuinely does not
fit the change at hand, say why rather than contorting the code to satisfy it.

Also see directory-specific guidelines:

- [docs-website/AGENTS.md](docs-website/AGENTS.md)
- [docs-website/docs/AGENTS.md](docs-website/docs/AGENTS.md)
- [haystack/hooks/compaction/AGENTS.md](haystack/hooks/compaction/AGENTS.md)
- [releasenotes/notes/AGENTS.md](releasenotes/notes/AGENTS.md)
- [test/AGENTS.md](test/AGENTS.md)

## API Design

- Let `Agent` own tool calls; don’t add pipeline invokers — `ToolInvoker` was removed in `3.0`
- Use `Pipeline.run()` and `Pipeline.run_async()`; don’t add `AsyncPipeline` — merged in `3.0`
- Use ChatGenerators like `OpenAIChatGenerator` — non-chat generators were removed in `3.0`
- Use Haystack serialization helpers (`component_to_dict`, `default_from_dict`) — preserves wire compatibility
- Make optional params keyword-only with `*` — preserves API compatibility
- Build API clients and load models in `warm_up()`, not `__init__` — preserves constructibility
- Treat `run_async` as optional in `haystack/components/` — fallback to `asyncio.to_thread(component.run, ...)` and log it
- Implement `warm_up()` only for real setup; make it idempotent and run it before tool access — This avoids fake lifecycle APIs, duplicate guards, and premature tool access before required setup.
- Keep `run_async`/`warm_up_async`/`close_async` paths async-only — use separate async state/hooks and only intentional non-blocking fallbacks like `asyncio.to_thread`
- Pass explicit `required_variables` for optional prompt vars — avoids requiring optional inputs
- Expose explicit `Agent` tool output contracts — use `raw=True` by default; document `last_message`, text-only output, and `exit_reason` when narrowing payloads
- Subclass `Toolset` only when inherited collection APIs match — prefer composition or raise `NotImplementedError`
- Read live Haystack `State` resources via `state.data.get(...)` or `state.data[...]` — avoids deep-copy bugs
- Keep `run(messages, *, streaming_callback, generation_kwargs, tools)` order in `haystack/components/generators/chat/` — preserves API consistency
- Append optional params in public signatures, including `__init__` — avoid breaking positional callers
- Prefer existing `haystack/core/pipeline` APIs or inline logic — avoid duplicate or one-off public APIs
- In `from_dict()`, pass only present fields to `__init__()` — keeps defaults centralized
- Shallow-copy `from_dict` payloads; rebuild parsed nested fields — avoids costly `deepcopy()`
- Spell out mirrored constructor params; avoid `*args`/`**kwargs` — clarifies supported config
- Reuse splitter chunk metadata in `haystack/components/preprocessors/` — preserve traceability fields like `page_number`, `source_id`, `header`, `parent_headers`, and split positions; add new keys only for existing downstream contracts
- Serialize tools/toolsets under `data` without flattening — preserves round-trip fidelity

## Documentation

- Match markup to surface: use reStructuredText in `releasenotes/notes/`, Haystack docstring style in code — prevents broken docs rendering
- Keep inline comments concise and non-obvious — remove restatements, but preserve durable caveats
- Write concise, user-visible `releasenotes/notes/*.yaml` entries — clarify API, workflow, error, output, and semantic changes users rely on
- Update docstrings with signature/type changes — remove obsolete behavior and reflect aliases like `ToolsType`
- Document private helpers only when needed — explain non-obvious behavior, constraints, or rationale
- Use only `releasenotes/config.yaml` sections in `releasenotes/notes/*.yaml` — keeps notes accurate and consistent
- Document public API params and `:returns:` contracts — clarify usage, types, and behavior
- Update `PromptBuilder`/`ChatPromptBuilder` docs with behavior changes — keep examples accurate, commented, and runnable
- Keep `:param` docs concise and current — document meaning, defaults, and exceptions only
- Use default constructors in docs examples; note required env vars like `OPENAI_API_KEY` nearby — Keeps docs focused on the demonstrated feature while still making required auth setup clear to readers.
- Document raised exceptions with existing `:raises ...:` style — use `ValueError if filters have invalid syntax` for invalid filters
- Keep doc examples minimal and local — include only used imports/code inside each snippet
- Show example output as comments (`# ...`, `# >> ...`) — keeps snippets valid Python
- Omit explicit defaults in docs examples — specify models only for model-specific behavior
- Align class/API docstrings with existing style, especially in `haystack/components/` — keeps API docs readable
- Remove `experimental` wording when promoting features — update docstrings, tutorials, `pydoc` IDs, and generated markdown filenames
- Sync `docs-website/versioned_docs/version-*/concepts/data-classes.mdx` with dataclass/message fields — prevents stale API docs
- Sync `haystack/components/` docs with actual behavior — prevent stale API, endpoint, integration, and example guidance
- Document user-visible changes in `MIGRATION.md` — explain what changed, why, and required action

## Code Style

- Use keyword args for multi-parameter calls — improves readability and prevents mix-ups
- Keep lint suppressions current and exact — use honored codes like `# noqa: PLR0915` only while needed
- Share sync/async logic in private helpers under `haystack/components/` — prevents drift and `# noqa: PLR0915`
- Scope PR diffs to the stated goal — avoid unrelated refactors, formatting, serialization, or cleanup
- Remove temporary `print()` debugging before merging — keeps tests and runtime output clean
- Avoid filename-only headers in source files — keeps files clean and style-consistent
- Omit args that only restate callee defaults — reduces noise and avoids stale examples
- Remove redundant branches and unused private helpers — document non-obvious compatibility or typing needs inline
- Use `{placeholder}` logger templates with kwargs — preserves structured logs and avoids eager formatting
- Keep private constants local and prefix module constants with `_` — clarifies API boundaries

## Type System

- Use coded `# type: ignore[...]` with a nearby why-safe comment — avoids hiding unrelated typing bugs
- Resolve callable annotations with `typing.get_type_hints()` — avoids bugs from raw or string `inspect.signature()` annotations
- Keep union aliases non-redundant and concept-named — include base types only when subclasses inherit
- Use `T | None`, not `Optional[T]`, in annotations and casts — keeps typing concise
- Annotate class-or-PEP-604 inputs as `type | types.UnionType` — matches runtime values
- Annotate helper params broadly and accurately; use `Any` for arbitrary typing objects — Accurate helper signatures improve static analysis without rejecting valid typing constructs at runtime.
- Use `@overload` only for real API variants; fix bad types at source, not with broad `type: ignore` — Real overloads preserve API semantics, while source fixes and narrow ignores prevent hiding type bugs.

## Imports

- Import via public APIs and keep only used or compatibility re-export imports — preserves API stability
- Use top-level absolute `haystack...` imports in `haystack/components/` and `haystack/core/component/*.py` — keeps component dependencies traceable
- Use `LazyImport` only for optional third-party deps — otherwise import directly
- Use the narrowest clear import form, but keep module imports like `import httpx` when namespaces matter — This prevents namespace, runtime, typing, and lazy-import conflicts while keeping imports readable and minimal.

## Config

- Pass `allowed_modules=` or set `HAYSTACK_DESERIALIZATION_ALLOWLIST` for YAML loads — never widen `haystack/core/serialization_security.py` allowlists in library code
- Remove dead config in `.github/workflows/` — keeps CI behavior accurate and maintainable

## Naming

- Prefix internal helpers with `_`, including `haystack/utils/` helpers — clarifies public API boundaries
- Avoid naming locals after imported decorators/functions/utilities — prevents shadowing ambiguity

## Testing

- Centralize `DocumentStore` tests in `haystack/testing/document_store.py` mixins — use `DocumentStoreBaseTests` as the minimum suite and compose capability mixins into `DocumentStoreBaseExtendedTests`; add explicit integration skips only for unsupported backend features.
- Test `haystack/core/pipeline/` behavior changes in matching pipeline test modules — cover end-to-end edge cases like shorthands, early returns, errors, task completion, empty inputs, and outputs

## General

- Define intentional package exports in `__init__.py` via `__all__`; avoid implementation-module `__all__` — Keeping package `__init__.py` exports intentional preserves stable public APIs and avoids accidental top-level imports.
- Update `pyproject.toml` for new package APIs — declare deps and minimum versions used
- State concrete deprecation timelines and status — helps users plan migrations

## File-Specific Rules

### `README.md`

- Keep `README.md` feature lists complete and concrete — clarify broad terms with examples
