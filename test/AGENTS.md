<!-- Mined from deepset PR reviews; see the repo-root AGENTS.md. -->

# test/ Guidelines

## Testing

- Name tests after verified behavior — keep names/docs current to avoid misleading coverage
- Share `pytest` fixtures/helpers/constants only for truly common test setup — prevents brittle coupling and duplicate inputs
- Keep tests minimal and non-redundant; prefer one smoke/contract test for brittle live paths like `test/components/generators/chat/`, `test/components/preprocessors/`, and `test/components/agents/test_agent.py` — reduces flaky, expensive, low-value coverage
- Group tests by behavior in existing files and classes, mirroring the source layout (`test/components/test_{component}.py`) — split a file only when it becomes hard to navigate
- Test serialization via public round trips — use `component_to_dict`/`component_from_dict`, `to_dict()`/`from_dict()`, or constructors; avoid hardcoded deep dicts
- Use real multi-chunk fixtures in `test/components/preprocessors/`; assert exact ordered content, metadata, and per-source `split_id`s. — Exact, ordered assertions catch splitter regressions in chunking, metadata propagation, and per-source `split_id` behavior that broad checks miss.
- Assert exception messages with `pytest.raises(..., match=...)`; fully match one related invalid case — catches user-visible error regressions while keeping tests readable
- Add regression tests for pipeline socket metadata changes — protects `Variadic`/`GreedyVariadic` edge cases
- Pair sync `run` integration tests with `run_async` tests — keep async behavior covered
- In splitter tests, assert the joined split content equals the input text — catches dropped, duplicated, or reordered content
- Skip OpenAI integration tests without `OPENAI_API_KEY`; use dummy keys for non-live tests — Keeps tests reliable in local and CI runs without requiring real OpenAI credentials unless explicitly testing the live integration.
- Assert splitter overlap offsets explicitly — verify `split_idx_start` and `_split_overlap` as character ranges into the original text, with coverage for each `split_unit`
- Test PEP 604 unions (`X | Y`, `X | None`) with `typing.Union`/`Optional` — catches annotation-compat bugs
- Assert full dict shapes in `test/components/generators/chat/` — catches schema regressions
- Assert `Pipeline` fan-in order from runtime semantics — use joiners for custom order

## General

- Avoid `# type: ignore` in tests; narrow with `hasattr(...)` or `assert isinstance(...)` first — Explicit narrowing keeps tests type-safe and exposes real API mismatches instead of hiding bugs from `mypy`.
- Keep test imports at module scope; remove redundant local imports — improves visibility and consistency
