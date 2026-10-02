# test/ Guidelines

## Testing

- Name tests after the behavior they verify and group them by behavior in the existing file mirroring the source layout (`test/components/<group>/test_{component}.py`); keep the suite minimal — one smoke/contract test for brittle live paths such as chat generators, preprocessors, and `test_agent.py` — and split a file only when it becomes hard to navigate
- Keep test imports at module scope and share `pytest` fixtures/helpers/constants only for truly common setup — prevents brittle coupling
- Test serialization via public round trips — use `component_to_dict`/`component_from_dict`, `to_dict()`/`from_dict()`, or constructors; avoid hardcoded deep dicts
- In splitter tests use real multi-chunk fixtures and assert exact ordered content, metadata, and per-source `split_id`s; assert the joined split content equals the input text; check `split_idx_start`/`_split_overlap` as character ranges into the original text for each `split_unit`
- Assert exception messages with `pytest.raises(..., match=...)`; fully match one related invalid case — catches user-visible error regressions while keeping tests readable
- Cover `haystack/core/pipeline/` changes in the matching pipeline test module with end-to-end edge cases (shorthands, early returns, errors, empty inputs, outputs); add regression tests for socket metadata around `Variadic`/`GreedyVariadic`; assert fan-in order from runtime semantics and use joiners when a custom order matters
- Pair sync `run` integration tests with `run_async` tests — keep async behavior covered
- Skip OpenAI integration tests without `OPENAI_API_KEY`; use dummy keys for non-live tests — keeps local and CI runs credential-free unless testing live
- Test PEP 604 unions (`X | Y`, `X | None`) with `typing.Union`/`Optional` — catches annotation-compat bugs
- Assert full dict shapes in `test/components/generators/chat/` — catches schema regressions

## General

- Avoid `# type: ignore` in tests; narrow with `hasattr(...)` or `assert isinstance(...)` first — exposes real API mismatches
