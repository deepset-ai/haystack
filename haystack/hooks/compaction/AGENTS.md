# haystack/hooks/compaction/ Guidelines

## API Design

- Use provider-compatible roles in `haystack/hooks/compaction/` — prefer `user` for synthetic markers
- Name and document compaction settings by their real semantics — retention in `turns`/`steps`, not `messages`; state when compaction runs, what is retained, and which tool-call context is preserved

## General

- Keep `haystack/hooks/compaction/` compactors narrative — move shared indexing, grouping, token counting, and helpers into focused utils
