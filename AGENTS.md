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

Type checking targets Python 3.12, above the 3.10 floor in `requires-python`, because numpy's stubs use PEP 695 `type` statements that mypy rejects below 3.12. So mypy will not catch code that breaks on 3.10; the unit test matrix runs on 3.10 and covers that.

To fix type issues, avoid `type: ignore`, casts, or assertions when possible. If they are necessary, explain why.

### Format and lint
hatch run fmt

## Release Notes

Every user-facing PR (not docs, not CI) must include a release note:

hatch run release-note SHORT_DESCRIPTION

Edit the generated file in `releasenotes/notes/`. Release notes use reStructuredText formatting; see the [release notes section in CONTRIBUTING.md](CONTRIBUTING.md#release-notes) for details.

In addition, there are rules inferred from previous code reviews. Follow them like a
reviewer's note: they encode reasons, not ceremony, so when a rule genuinely does not
fit the change at hand, say why rather than contorting the code to satisfy it.

Also see directory-specific guidelines:

- [docs-website/AGENTS.md](docs-website/AGENTS.md)
- [docs-website/docs/AGENTS.md](docs-website/docs/AGENTS.md)
- [haystack/hooks/compaction/AGENTS.md](haystack/hooks/compaction/AGENTS.md)
- [releasenotes/notes/AGENTS.md](releasenotes/notes/AGENTS.md)
- [test/AGENTS.md](test/AGENTS.md)

## API Design

- Target the 3.0 API: `ToolInvoker`, `AsyncPipeline`, and non-chat generators were removed — let `Agent` own tool calls, use `Pipeline.run()`/`Pipeline.run_async()`, and use ChatGenerators like `OpenAIChatGenerator`
- Serialize with the Haystack helpers (`component_to_dict`, `default_from_dict`): in `from_dict()` pass only the fields present to `__init__()` so defaults stay centralized, shallow-copy the payload instead of `deepcopy()`, and keep tools/toolsets nested under `data` — preserves wire compatibility and round-trip fidelity
- Keep public signatures explicit and compatible: optional params keyword-only after `*` and appended rather than inserted, mirrored constructor params spelled out instead of `*args`/`**kwargs`, and chat generators keeping the `run(messages, *, streaming_callback, generation_kwargs, tools)` order — avoids breaking positional callers
- Create API clients and load models in `warm_up()`, not `__init__`; implement `warm_up()` only for real setup and make it idempotent — keeps components constructible and serializable without credentials or network
- Implement `run_async` only when there is a real async execution path — `Pipeline.run_async()` already falls back to `asyncio.to_thread(component.run, ...)` when it is missing; where it exists, keep `run_async`/`warm_up_async`/`close_async` genuinely async with separate async state and hooks, and share sync/async logic through private helpers — prevents event-loop blocking and sync/async drift
- Pass explicit `required_variables` for optional prompt vars — avoids requiring optional inputs
- Subclass `Toolset` only when inherited collection APIs match — prefer composition or raise `NotImplementedError`
- Read live Haystack `State` resources via `state.data.get(...)` or `state.data[...]` — avoids deep-copy bugs
- Prefer existing `haystack/core/pipeline` APIs or inline logic — avoid duplicate or one-off public APIs
- Reuse splitter chunk metadata in `haystack/components/preprocessors/` — keep traceability fields like `page_number`, `source_id`, `header`, `parent_headers`, and split positions; add new keys only for existing downstream contracts

## Documentation

- Keep inline comments and private-helper docs to what is non-obvious — remove restatements, keep durable caveats and rationale
- Keep docstrings current with signatures and behavior, in the existing Haystack style: each public `:param` by meaning, default, and constraints; `:returns:` contracts; exceptions in the existing `:raises ValueError:` style; aliases like `ToolsType` reflected — stale docs mislead users and assistants
- Keep doc examples minimal, runnable, and local: default constructors with required env vars like `OPENAI_API_KEY` noted nearby, only the imports the snippet uses, no restated defaults (name a model only for model-specific behavior), expected output as comments, generally `# >> ...`
- When behavior, fields, or names change, update every surface in the same PR: `haystack/components/` docstrings and examples, `docs-website/docs/` plus the current `versioned_docs/version-*/` page (e.g. `concepts/data-classes.mdx`), and `experimental` wording, `pydoc` IDs, and generated markdown filenames when promoting features

## Code Style

- Use keyword args for multi-parameter calls — improves readability and prevents mix-ups
- Keep lint suppressions exact and current — a `# noqa: PLR0915` only while the code needs it
- Scope diffs to the stated goal and leave them clean: no unrelated refactors or formatting, no stray `print()`, no filename-only header comments, no unused private helpers or redundant branches, no args that only restate callee defaults
- Use `{placeholder}` logger templates with kwargs — preserves structured logs and avoids eager formatting
- Prefix internal helpers and module-level constants with `_` and keep private constants local; never name locals after imported functions, decorators, or utilities — clarifies API boundaries and avoids shadowing

## Type System

- Fix types at the source; where a suppression is unavoidable use a coded `# type: ignore[...]` with a nearby why-safe comment, and add `@overload` only for real API variants
- Resolve callable annotations with `typing.get_type_hints()` — avoids bugs from raw or string `inspect.signature()` annotations
- Keep union aliases non-redundant and concept-named — include base types only when subclasses inherit
- Use `T | None`, not `Optional[T]`, in annotations and casts — keeps typing concise
- Annotate helper params broadly and accurately — `type | types.UnionType` for class-or-PEP-604 inputs, `Any` for arbitrary typing objects

## Imports

- Import via public APIs in the narrowest clear form (keep module imports like `import httpx` when namespaces matter); keep only used imports plus deliberate compatibility re-exports
- Use top-level absolute `haystack...` imports in `haystack/components/` and `haystack/core/component/*.py` — keeps component dependencies traceable
- Use `LazyImport` only for optional third-party deps — otherwise import directly

## Config

- Pass `allowed_modules=` or set `HAYSTACK_DESERIALIZATION_ALLOWLIST` for YAML loads — never widen `haystack/core/serialization_security.py` allowlists in library code

## Testing

- Centralize `DocumentStore` tests in `haystack/testing/document_store.py` mixins — use `DocumentStoreBaseTests` as the minimum suite and compose capability mixins into `DocumentStoreBaseExtendedTests`; add explicit integration skips only for unsupported backend features.

## General

- Define package exports in `__init__.py` via `__all__`; avoid `__all__` in implementation modules — keeps public APIs intentional
- Update `pyproject.toml` for new package APIs — declare deps and minimum versions used
