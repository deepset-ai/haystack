<!-- Mined from deepset PR reviews; see the repo-root AGENTS.md. -->

# haystack/hooks/compaction/ Guidelines

## API Design

- Use provider-compatible roles in `haystack/hooks/compaction/` — prefer `user` for synthetic markers
- Name compaction retention by semantic unit (`turns`/`steps`), not `messages` — matches what is actually preserved

## General

- Document `haystack/hooks/compaction/` APIs by real semantics — prevents compaction misuse
- Keep `haystack/hooks/compaction/` compactors narrative — move shared indexing, grouping, token counting, and helpers into focused utils
