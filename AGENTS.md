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

<!-- braindump:begin -->

The rules below were mined from 2,746 PR review comments written by the deepset
team between 2025-07-01 and 2026-08-14, then filtered against the current source tree so that
guidance referring to APIs removed or moved in Haystack 3.0 does not survive. Each
`<!-- rule:N -->` marker traces back to the review comments it came from.

They describe what reviewers actually enforce. Follow them the way you would follow a
reviewer's note: they encode reasons, not ceremony, so when a rule genuinely does not
fit the change at hand, say why rather than contorting the code to satisfy it.

## API Design

<!-- rule:-4 -->
- Let `Agent` own tool calls; don’t add pipeline invokers — `ToolInvoker` was removed in `3.0`
<!-- rule:-5 -->
- Use `Pipeline.run()` and `Pipeline.run_async()`; don’t add `AsyncPipeline` — merged in `3.0`
<!-- rule:-1 -->
- Use ChatGenerators like `OpenAIChatGenerator` — non-chat generators were removed in `3.0`
<!-- rule:98 -->
- Use Haystack serialization helpers (`component_to_dict`, `default_from_dict`) — preserves wire compatibility
<!-- rule:26 -->
- Make optional params keyword-only with `*` — preserves API compatibility
<!-- rule:316 -->
- Build API clients and load models in `warm_up()`, not `__init__` — preserves constructibility
<!-- rule:207 -->
- Treat `run_async` as optional in `haystack/components/` — fallback to `asyncio.to_thread(component.run, ...)` and log it
<!-- rule:35 -->
- Implement `warm_up()` only for real setup; make it idempotent and run it before tool access — This avoids fake lifecycle APIs, duplicate guards, and premature tool access before required setup.
<!-- rule:362 -->
- Keep `run_async`/`warm_up_async`/`close_async` paths async-only — use separate async state/hooks and only intentional non-blocking fallbacks like `asyncio.to_thread`
<!-- rule:176 -->
- Pass explicit `required_variables` for optional prompt vars — avoids requiring optional inputs
<!-- rule:0 -->
- Expose explicit `Agent` tool output contracts — use `raw=True` by default; document `last_message`, text-only output, and `exit_reason` when narrowing payloads
<!-- rule:100 -->
- Subclass `Toolset` only when inherited collection APIs match — prefer composition or raise `NotImplementedError`
<!-- rule:389 -->
- Read live Haystack `State` resources via `state.data.get(...)` or `state.data[...]` — avoids deep-copy bugs
<!-- rule:365 -->
- Keep `run(messages, *, streaming_callback, generation_kwargs, tools)` order in `haystack/components/generators/chat/` — preserves API consistency
<!-- rule:21 -->
- Append optional params in public signatures, including `__init__` — avoid breaking positional callers
<!-- rule:205 -->
- Prefer existing `haystack/core/pipeline` APIs or inline logic — avoid duplicate or one-off public APIs
<!-- rule:97 -->
- In `from_dict()`, pass only present fields to `__init__()` — keeps defaults centralized
<!-- rule:393 -->
- Shallow-copy `from_dict` payloads; rebuild parsed nested fields — avoids costly `deepcopy()`
<!-- rule:190 -->
- Spell out mirrored constructor params; avoid `*args`/`**kwargs` — clarifies supported config
<!-- rule:274 -->
- Reuse splitter chunk metadata in `haystack/components/preprocessors/` — preserve traceability fields like `page_number`, `source_id`, `header`, `parent_headers`, and split positions; add new keys only for existing downstream contracts
<!-- rule:324 -->
- Serialize tools/toolsets under `data` without flattening — preserves round-trip fidelity

## Documentation

<!-- rule:51 -->
- Match markup to surface: use reStructuredText in `releasenotes/notes/`, Haystack docstring style in code — prevents broken docs rendering
<!-- rule:108 -->
- Keep inline comments concise and non-obvious — remove restatements, but preserve durable caveats
<!-- rule:9 -->
- Write concise, user-visible `releasenotes/notes/*.yaml` entries — clarify API, workflow, error, output, and semantic changes users rely on
<!-- rule:162 -->
- Update docstrings with signature/type changes — remove obsolete behavior and reflect aliases like `ToolsType`
<!-- rule:255 -->
- Document private helpers only when needed — explain non-obvious behavior, constraints, or rationale
<!-- rule:53 -->
- Use only `releasenotes/config.yaml` sections in `releasenotes/notes/*.yaml` — keeps notes accurate and consistent
<!-- rule:49 -->
- Document public API params and `:returns:` contracts — clarify usage, types, and behavior
<!-- rule:28 -->
- Update `PromptBuilder`/`ChatPromptBuilder` docs with behavior changes — keep examples accurate, commented, and runnable
<!-- rule:381 -->
- Keep `:param` docs concise and current — document meaning, defaults, and exceptions only
<!-- rule:44 -->
- Use default constructors in docs examples; note required env vars like `OPENAI_API_KEY` nearby — Keeps docs focused on the demonstrated feature while still making required auth setup clear to readers.
<!-- rule:128 -->
- Document raised exceptions with existing `:raises ...:` style — use `ValueError if filters have invalid syntax` for invalid filters
<!-- rule:166 -->
- Keep doc examples minimal and local — include only used imports/code inside each snippet
<!-- rule:59 -->
- Show example output as comments (`# ...`, `# >> ...`) — keeps snippets valid Python
<!-- rule:242 -->
- Omit explicit defaults in docs examples — specify models only for model-specific behavior
<!-- rule:350 -->
- Align class/API docstrings with existing style, especially in `haystack/components/` — keeps API docs readable
<!-- rule:47 -->
- Remove `experimental` wording when promoting features — update docstrings, tutorials, `pydoc` IDs, and generated markdown filenames
<!-- rule:45 -->
- Sync `docs-website/versioned_docs/version-*/concepts/data-classes.mdx` with dataclass/message fields — prevents stale API docs
<!-- rule:247 -->
- Sync `haystack/components/` docs with actual behavior — prevent stale API, endpoint, integration, and example guidance
<!-- rule:276 -->
- Document user-visible changes in `MIGRATION.md` — explain what changed, why, and required action

## Code Style

<!-- rule:374 -->
- Use keyword args for multi-parameter calls — improves readability and prevents mix-ups
<!-- rule:408 -->
- Delete lines explicitly marked to remove — resolves review feedback exactly and avoids dead code
<!-- rule:29 -->
- Compute values once per function and reuse them — prevents drift and duplicate work
<!-- rule:173 -->
- Keep lint suppressions current and exact — use honored codes like `# noqa: PLR0915` only while needed
<!-- rule:70 -->
- Share sync/async logic in private helpers under `haystack/components/` — prevents drift and `# noqa: PLR0915`
<!-- rule:175 -->
- Scope PR diffs to the stated goal — avoid unrelated refactors, formatting, serialization, or cleanup
<!-- rule:254 -->
- Remove temporary `print()` debugging before merging — keeps tests and runtime output clean
<!-- rule:353 -->
- Avoid filename-only headers in source files — keeps files clean and style-consistent
<!-- rule:57 -->
- Omit args that only restate callee defaults — reduces noise and avoids stale examples
<!-- rule:291 -->
- Revert review-rejected changes exactly — restore affected files or lines to the accepted state
<!-- rule:495 -->
- Remove redundant branches and unused private helpers — document non-obvious compatibility or typing needs inline
<!-- rule:476 -->
- Use `{placeholder}` logger templates with kwargs — preserves structured logs and avoids eager formatting
<!-- rule:312 -->
- Keep private constants local and prefix module constants with `_` — clarifies API boundaries

## Type System

<!-- rule:120 -->
- Use coded `# type: ignore[...]` with a nearby why-safe comment — avoids hiding unrelated typing bugs
<!-- rule:382 -->
- Resolve callable annotations with `typing.get_type_hints()` — avoids bugs from raw or string `inspect.signature()` annotations
<!-- rule:290 -->
- Keep union aliases non-redundant and concept-named — include base types only when subclasses inherit
<!-- rule:73 -->
- Use `T | None`, not `Optional[T]`, in annotations and casts — keeps typing concise
<!-- rule:88 -->
- Annotate class-or-PEP-604 inputs as `type | types.UnionType` — matches runtime values
<!-- rule:542 -->
- Use `typing.Union[...]` instead of `|` unions until Python `3.10+` is required — Using `typing.Union[...]` keeps annotations compatible with Python versions before `3.10`.
<!-- rule:136 -->
- Annotate helper params broadly and accurately; use `Any` for arbitrary typing objects — Accurate helper signatures improve static analysis without rejecting valid typing constructs at runtime.
<!-- rule:373 -->
- Use `@overload` only for real API variants; fix bad types at source, not with broad `type: ignore` — Real overloads preserve API semantics, while source fixes and narrow ignores prevent hiding type bugs.

## Imports

<!-- rule:300 -->
- Import via public APIs and keep only used or compatibility re-export imports — preserves API stability
<!-- rule:177 -->
- Use top-level absolute `haystack...` imports in `haystack/components/` and `haystack/core/component/*.py` — keeps component dependencies traceable
<!-- rule:241 -->
- Use `LazyImport` only for optional third-party deps — otherwise import directly
<!-- rule:339 -->
- Use the narrowest clear import form, but keep module imports like `import httpx` when namespaces matter — This prevents namespace, runtime, typing, and lazy-import conflicts while keeping imports readable and minimal.

## Config

<!-- rule:-3 -->
- Pass `allowed_modules=` or set `HAYSTACK_DESERIALIZATION_ALLOWLIST` for YAML loads — never widen `haystack/core/serialization_security.py` allowlists in library code
<!-- rule:223 -->
- Remove dead config in `.github/workflows/` — keeps CI behavior accurate and maintainable

## Naming

<!-- rule:169 -->
- Prefix internal helpers with `_`, including `haystack/utils/` helpers — clarifies public API boundaries
<!-- rule:106 -->
- Avoid naming locals after imported decorators/functions/utilities — prevents shadowing ambiguity

## Testing

<!-- rule:125 -->
- Centralize `DocumentStore` tests in `haystack/testing/document_store.py` mixins — use `DocumentStoreBaseTests` as the minimum suite and compose capability mixins into `DocumentStoreBaseExtendedTests`; add explicit integration skips only for unsupported backend features.
<!-- rule:110 -->
- Test `haystack/core/pipeline/` behavior changes in matching pipeline test modules — cover end-to-end edge cases like shorthands, early returns, errors, task completion, empty inputs, and outputs

## General

<!-- rule:311 -->
- Define intentional package exports in `__init__.py` via `__all__`; avoid implementation-module `__all__` — Keeping package `__init__.py` exports intentional preserves stable public APIs and avoids accidental top-level imports.
<!-- rule:422 -->
- Update `pyproject.toml` for new package APIs — declare deps and minimum versions used
<!-- rule:117 -->
- State concrete deprecation timelines and status — helps users plan migrations

## File-Specific Rules

### `README.md`
<!-- rule:163 -->
- Keep `README.md` feature lists complete and concrete — clarify broad terms with examples

## Directory-specific conventions

### `docs-website/`

<!-- rule:20 -->
- Sync `docs-website` API names and import paths with public exports — avoids stale docs
<!-- rule:36 -->
- Omit explicit `.warm_up()` in docs unless required — lazy/idempotent warm-up handles it
<!-- rule:17 -->
- Keep setup/usage for maintained APIs and integrations in `docs-website/docs/` — it is the authoritative, navigable source; add pages to `docs-website/sidebars.js` when needed
<!-- rule:134 -->
- Keep `docs-website/docs/concepts/` current-facing — put history and upgrades in migration docs
<!-- rule:295 -->
- Link data-class symbols to anchored API docs — improves discoverability and precision
<!-- rule:212 -->
- Verify all `docs-website` MDX links — keeps external URLs current and internal routes valid

### `docs-website/docs/`

<!-- rule:30 -->
- Keep `docs-website/docs/pipeline-components/generators/` examples concise and `Agent`-level — improves copyability and keeps docs focused
<!-- rule:388 -->
- Edit `docs-website/docs/pipeline-components/agents-1/` only for outdated, incorrect, or materially useful user-facing API/behavior guidance — Keeps the agents docs focused and avoids churn while ensuring users get accurate, useful guidance.
<!-- rule:263 -->
- Add `## Overview` near the top of `docs-website/docs/pipeline-components/**` pages — explains what the component does and why to use it before details
<!-- rule:286 -->
- Label `agents-1` docs sections clearly — mark examples/variants and customization paths
<!-- rule:245 -->
- Document component outputs and link producers/API refs — keeps docs ecosystem-connected
<!-- rule:218 -->
- Use `ChatPromptBuilder`, `ChatMessage`, and chat generators in new docs LLM pipelines — match wiring, edge names like `prompt`, and declared variables to real chat interfaces.
<!-- rule:81 -->
- Polish `docs-website/docs/pipeline-components/**/*.mdx` prose before merge — keeps docs clear and consistent
<!-- rule:251 -->
- Sync `docs-website/docs/pipeline-components` YAML examples with current defaults — stale model names cause config errors
<!-- rule:130 -->
- Mark joiners/adapters optional when smart pipeline connections make them optional — Prevents docs from implying extra pipeline components are mandatory when smart connections already handle the composition.
<!-- rule:262 -->
- Document extractor side effects and exact `doc.meta` keys — clarifies pipeline data flow
<!-- rule:129 -->
- Prefer `result["last_message"]` for Haystack agent final responses — highlights the intended API
<!-- rule:10 -->
- Link partial config/API summaries to authoritative references — helps users find full details

### `haystack/hooks/compaction/`

<!-- rule:405 -->
- Use provider-compatible roles in `haystack/hooks/compaction/` — prefer `user` for synthetic markers
<!-- rule:411 -->
- Name compaction retention by semantic unit (`turns`/`steps`), not `messages` — matches what is actually preserved
<!-- rule:412 -->
- Document `haystack/hooks/compaction/` APIs by real semantics — prevents compaction misuse
<!-- rule:419 -->
- Keep `haystack/hooks/compaction/` compactors narrative — move shared indexing, grouping, token counting, and helpers into focused utils

### `releasenotes/notes/`

<!-- rule:92 -->
- Add `upgrade` notes for breaking/user-visible changes in `releasenotes/notes/` — explain affected users, old/new behavior, and migration steps
<!-- rule:121 -->
- Add `releasenotes/notes/` entries only for in-scope user-facing PR changes — keeps release notes accurate and low-noise; leave unrelated note files untouched.
<!-- rule:211 -->
- Name affected APIs/configs in `releasenotes/notes/` — clarifies scope and impact for users
<!-- rule:258 -->
- Write bug/security notes around public impact, not private helpers — clarifies user risk
<!-- rule:109 -->
- Highlight APIs in `releasenotes/notes/` only with examples or clear use cases — shows practical value
<!-- rule:151 -->
- Keep `releasenotes/notes/` reno notes synced with shipped behavior — prevents misleading release docs
<!-- rule:303 -->
- Use one `releasenotes/notes/` file per PR — group related change notes together
<!-- rule:32 -->
- Proofread `releasenotes/notes/` entries — catches API typos and formatting issues

### `test/`

<!-- rule:4 -->
- Name tests after verified behavior — keep names/docs current to avoid misleading coverage
<!-- rule:145 -->
- Share `pytest` fixtures/helpers/constants only for truly common test setup — prevents brittle coupling and duplicate inputs
<!-- rule:58 -->
- Keep tests minimal and non-redundant; prefer one smoke/contract test for brittle live paths like `test/components/generators/chat/`, `test/components/preprocessors/`, and `test/components/agents/test_agent.py` — reduces flaky, expensive, low-value coverage
<!-- rule:116 -->
- Group tests by behavior in existing files/classes — use layouts like `test/components/test_{component}.py`, `test/components/agents/test_agent.py`, `test/core/pipeline/test_pipeline_base.py` (`TestPipelineBaseFromDict`), and `test/core/pipeline/` for breakpoints; split only when files get hard to navigate
<!-- rule:257 -->
- Test serialization via public round trips — use `component_to_dict`/`component_from_dict`, `to_dict()`/`from_dict()`, or constructors; avoid hardcoded deep dicts
<!-- rule:259 -->
- Use real multi-chunk fixtures in `test/components/preprocessors/`; assert exact ordered content, metadata, and per-source `split_id`s. — Exact, ordered assertions catch splitter regressions in chunking, metadata propagation, and per-source `split_id` behavior that broad checks miss.
<!-- rule:77 -->
- Assert exception messages with `pytest.raises(..., match=...)`; fully match one related invalid case — catches user-visible error regressions while keeping tests readable
<!-- rule:114 -->
- Add regression tests for pipeline socket metadata changes — protects `Variadic`/`GreedyVariadic` edge cases
<!-- rule:203 -->
- Pair sync `run` integration tests with `run_async` tests — keep async behavior covered
<!-- rule:250 -->
- Assert `"".join(doc.content for doc in split_docs) == text` in `DocumentSplitter` tests — catches content loss
<!-- rule:279 -->
- Skip OpenAI integration tests without `OPENAI_API_KEY`; use dummy keys for non-live tests — Keeps tests reliable in local and CI runs without requiring real OpenAI credentials unless explicitly testing the live integration.
<!-- rule:369 -->
- Assert `RecursiveDocumentSplitter` overlap chunks and offsets — verify `split_idx_start`/`_split_overlap` as original-text character ranges, with parallel `split_unit` coverage for `word`/`token`.
<!-- rule:76 -->
- Test PEP 604 unions (`X | Y`, `X | None`) with `typing.Union`/`Optional` — catches annotation-compat bugs
<!-- rule:275 -->
- Assert full dict shapes in `test/components/generators/chat/` — catches schema regressions
<!-- rule:240 -->
- Assert `Pipeline` fan-in order from runtime semantics — use joiners for custom order
<!-- rule:238 -->
- Avoid `# type: ignore` in tests; narrow with `hasattr(...)` or `assert isinstance(...)` first — Explicit narrowing keeps tests type-safe and exposes real API mismatches instead of hiding bugs from `mypy`.
<!-- rule:299 -->
- Keep test imports at module scope; remove redundant local imports — improves visibility and consistency

<!-- braindump:end -->
