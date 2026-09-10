# docs-website/docs/ Guidelines

## Documentation

- Edit `docs-website/docs/pipeline-components/` pages only for outdated, incorrect, or materially useful guidance; keep examples concise and `Agent`-level and polish prose before merge — avoids churn while keeping docs copyable
- Add `## Overview` near the top of `docs-website/docs/pipeline-components/**` pages — explains what the component does and why to use it before details
- Document component outputs, extractor side effects, and exact `doc.meta` keys; link producers, API references, and the authoritative reference for any partial config summary
- Use current chat APIs in new docs pipelines — `ChatPromptBuilder`, `ChatMessage`, chat generators, and `result["last_message"]` for agent output — matching wiring, edge names like `prompt`, and declared variables; keep YAML examples on current default model names
- Mark joiners/adapters optional where smart pipeline connections already handle the composition
