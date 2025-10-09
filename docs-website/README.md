# Haystack Docs Website

This folder contains the Docusaurus-powered Haystack documentation website. It was migrated from the standalone `haystack-docs` repository into the monorepo under `docs-website/`.

## Prerequisites

- Node.js >= 18
- npm (preferred, repo includes `package-lock.json`) or Yarn
- Run all commands from `haystack/docs-website`

## Quick start

Install dependencies:

```bash
npm install
# or
yarn
```

Start the local dev server:

```bash
npm run start
# or
yarn start
```

This opens a local server with live-reload. Edits to docs reflect automatically.

> [!NOTE]
> The legacy Python docs under `haystack/docs/` are not part of this site. They remain unchanged and will be integrated later.

## Build and preview

Build static assets:

```bash
npm run build
# or
yarn build
```

Preview the production build locally:

```bash
npm run serve
# or
yarn serve
```

## Testing Python Snippets

Python code snippets in the docs are automatically tested. To test locally:

```bash
# Setup dependencies for a specific Haystack version
./scripts/setup-dev.sh 2.16.1

# Run tests
python scripts/test_python_snippets.py --verbose
```

See `scripts/` directory for more details.

## Versioning

Use the Docusaurus CLI to create and manage versions. See the official docs for details: [Versioning](https://docusaurus.io/docs/versioning).

Create a new version from the current docs (default plugin):

```bash
npm run docusaurus -- docs:version 2.1.0
# or
yarn docusaurus docs:version 2.1.0
```

If you also need to version the API Reference docs (separate docs plugin with id `reference`), run:

```bash
npm run docusaurus -- docs:version 2.1.0 -- --id reference
# or
yarn docusaurus docs:version 2.1.0 --id reference
```

These commands will:
- Create `versioned_docs/` and `versioned_sidebars/` entries for the new version
- Update `versions.json`

Optional configuration:
- You can customize the label shown for the “current” docs via `docs.versions.current.label` in `docusaurus.config.js`.
- To control which versions appear and the order in the dropdown, adjust `lastVersion` and `onlyIncludeVersions` in the docs plugin config.

## Authoring templates (hidden, not rendered)

- Templates live under `docs/_templates/` and are excluded from the site build.
- Duplicate a template and move the copy to the appropriate place under `docs/`:
  - `docs/_templates/component-template.mdx` → for new component docs
  - `docs/_templates/document-store-template.mdx` → for new document store docs
- After copying, update the frontmatter (`title`, `id`, `description`, `slug`) and fill in the sections.
- Do not commit new docs under `_templates/`; place them in their final location under `docs/`.

## Troubleshooting

### Blank page

If you see a blank page when running `npm start`:

```bash
# Clear Docusaurus cache and restart
npm run clear
npm start
```

If the issue persists, you may need to build once to generate route metadata:

```bash
npm run build
npm start
```

This happens because Docusaurus needs to generate internal routing metadata for versioned docs on first run.

### General issues

Clear cached data if something looks off:

```bash
npm run clear
# or
yarn clear
```
