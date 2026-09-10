# docs-website/ Guidelines

## Documentation

- Keep `docs-website` API names, import paths, and links current with public exports — link data-class symbols to anchored API docs and verify every MDX link (internal routes and external URLs) resolves
- Omit explicit `.warm_up()` in docs unless required — lazy/idempotent warm-up handles it
- Keep setup/usage for maintained APIs and integrations in `docs-website/docs/` — it is the authoritative, navigable source; add pages to `docs-website/sidebars.js` when needed
