<!-- Mined from deepset PR reviews; see the repo-root AGENTS.md. -->

# docs-website/docs/ Guidelines

## Documentation

- Keep `docs-website/docs/pipeline-components/generators/` examples concise and `Agent`-level — improves copyability and keeps docs focused
- Edit `docs-website/docs/pipeline-components/agents-1/` only for outdated, incorrect, or materially useful user-facing API/behavior guidance — Keeps the agents docs focused and avoids churn while ensuring users get accurate, useful guidance.
- Add `## Overview` near the top of `docs-website/docs/pipeline-components/**` pages — explains what the component does and why to use it before details
- Label `agents-1` docs sections clearly — mark examples/variants and customization paths
- Document component outputs and link producers/API refs — keeps docs ecosystem-connected
- Use `ChatPromptBuilder`, `ChatMessage`, and chat generators in new docs LLM pipelines — match wiring, edge names like `prompt`, and declared variables to real chat interfaces.
- Polish `docs-website/docs/pipeline-components/**/*.mdx` prose before merge — keeps docs clear and consistent
- Sync `docs-website/docs/pipeline-components` YAML examples with current defaults — stale model names cause config errors
- Mark joiners/adapters optional when smart pipeline connections make them optional — Prevents docs from implying extra pipeline components are mandatory when smart connections already handle the composition.
- Document extractor side effects and exact `doc.meta` keys — clarifies pipeline data flow
- Prefer `result["last_message"]` for Haystack agent final responses — highlights the intended API
- Link partial config/API summaries to authoritative references — helps users find full details
