<!-- Mined from deepset PR reviews; see the repo-root AGENTS.md. -->

# docs-website/ Guidelines

## Documentation

- Sync `docs-website` API names and import paths with public exports — avoids stale docs
- Omit explicit `.warm_up()` in docs unless required — lazy/idempotent warm-up handles it
- Keep setup/usage for maintained APIs and integrations in `docs-website/docs/` — it is the authoritative, navigable source; add pages to `docs-website/sidebars.js` when needed
- Keep `docs-website/docs/concepts/` current-facing — put history and upgrades in migration docs
- Link data-class symbols to anchored API docs — improves discoverability and precision
- Verify all `docs-website` MDX links — keeps external URLs current and internal routes valid
