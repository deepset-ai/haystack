# Contributing to Haystack Documentation

Thank you for your interest in contributing to the Haystack documentation! This guide provides everything you need to write, review, and maintain high-quality documentation for the Haystack project.

This guide focuses specifically on documentation contributions. For code contributions, tests, or integrations in the main Haystack codebase, please see the [main Haystack contribution guide](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md).

## TL;DR — Your first docs PR in 10 minutes

1. Clone and enter the docs site:

```bash
git clone https://github.com/YOUR_USERNAME/haystack.git
cd haystack/docs-website
```

2. Install and start:

```bash
npm install
npm start
```

3. Edit under `docs/`. If you add a new page, include its ID in `sidebars.js`.

4. Optional production check:

```bash
npm run build && npm run serve
```

5. Commit and push:

```bash
git checkout -b docs/your-branch
git add .
git commit -m "docs: fix <desc>"
git push -u origin HEAD
```

6. Open a PR and use the checklist below.

Optional:
- Prose lint: `vale --config .vale.ini "docs/**/*.{md,mdx}"`
- Snippet tests: `./scripts/setup-dev.sh main && python scripts/test_python_snippets.py --verbose`

**Table of Contents**

- [TL;DR — Your first docs PR in 10 minutes](#tldr--your-first-docs-pr-in-10-minutes)
- [Authoring New or Updated Pages](#authoring-new-or-updated-pages)
  - [Where should I edit?](#where-should-i-edit)
  - [Page Frontmatter](#page-frontmatter)
  - [Updating Navigation](#updating-navigation)
  - [Linking and Anchors](#linking-and-anchors)
  - [Admonitions (Callouts)](#admonitions-callouts)
- [Working with Templates](#working-with-templates)
- [Testing](#testing)
  - [Build Testing](#build-testing)
  - [Python Snippet Testing](#python-snippet-testing)
  - [Prose and Style Linting with Vale](#prose-and-style-linting-with-vale)
- [API Reference Contributions](#api-reference-contributions)
- [Versioning](#versioning)
  - [Creating a New Version](#creating-a-new-version)
  - [Version Management](#version-management)
  - [Understanding Documentation Versions and Where to Make Changes](#understanding-documentation-versions-and-where-to-make-changes)
- [CI/CD Workflows](#cicd-workflows)
  - [Vale Linting](#vale-linting)
  - [Python Snippet Testing](#python-snippet-testing-1)
  - [API Reference Sync](#api-reference-sync)
  - [Version Promotion](#version-promotion)
- [Preview Deployments](#preview-deployments)
- [Troubleshooting](#troubleshooting)
  - [Blank Page on npm start](#blank-page-on-npm-start)
  - [Cache Issues](#cache-issues)
  - [Build Errors](#build-errors)
  - [Vale StylesPath Error](#vale-stylespath-error)
- [Moving or Removing Pages](#moving-or-removing-pages)
- [Images and Assets](#images-and-assets)
- [Pull Request Process](#pull-request-process)
  - [Pull Request Checklist](#pull-request-checklist)
- [Review Process](#review-process)
- [Accessibility and Inclusivity](#accessibility-and-inclusivity)
- [Getting Help](#getting-help)

## Authoring New or Updated Pages

### Where should I edit?

| Your change | Edit here | Also edit here |
|---|---|---|
| New feature on Haystack `main` | `docs/` | — |
| Fix in current stable docs | `docs/` | `versioned_docs/version-<current>/` |
| API reference content | Edit Python docstrings in main repo | — |

### Page Frontmatter

Every documentation page requires frontmatter at the top:

```md
---
title: "Page Title"
id: "page-id"
description: "One to two sentences describing the page content for SEO and previews"
slug: "/target-url"
---
```

**Frontmatter fields:**

- `title`: Displayed page title (title case)
- `id`: Unique identifier for the page
- `description`: SEO description (1-2 sentences)
- `slug`: URL path for the page (optional, defaults to file path)

### Updating Navigation

After creating or moving a page, update the sidebar:

**For narrative docs (`docs/`):**

Edit `sidebars.js` and add your page to the appropriate category:

```javascript
{
  type: 'category',
  label: 'Concepts',
  items: [
    'concepts/pipelines',
    'concepts/your-new-page',  // Add here
  ],
}
```

**For API reference (`reference/`):**

Edit `reference-sidebars.js` if needed (some sections are auto-generated).

### Linking and Anchors

**Internal links:**

Use relative paths:

```md
See the [Pipeline Guide](../concepts/pipelines.mdx)
```

**Explicit anchors:**

For stable cross-links, use explicit heading IDs:

```markdown
## Installation {#install-guide}
```

Link to it: `[Install](./page.mdx#install-guide)`

### Admonitions (Callouts)

Use Docusaurus admonitions sparingly for supporting information:

```mdx
:::note
General notes or important information to highlight.
:::

:::tip
Short tip that helps the reader succeed.
:::

:::info
Useful but non-blocking background information.
:::

:::warning
Risky settings or potential pitfalls.
:::

:::danger
Data loss or security-impacting issues.
:::
```

## Working with Templates

Starter templates are available in `docs/_templates/`:

- `component-template.mdx` - For new component documentation
- `document-store-template.mdx` - For new document store guides

**How to use templates:**

1. Copy the appropriate template from `docs/_templates/`
2. Move the copy to its final location under `docs/`
3. Update the frontmatter (title, id, description, slug)
4. Fill in all sections marked with placeholders
5. Update the sidebar to include your new page

**Do not:**
- Commit new documentation under `_templates/`
- Leave template placeholder text in production docs

## Testing

### Build Testing

Before opening a PR, ensure the site builds cleanly:

```bash
npm run build
```

This command:
- Builds production-ready static files
- Validates all links and anchors
- Reports broken links, duplicate routes, and errors

**Fix all warnings before submitting your PR.**

### Python Snippet Testing

Python code snippets in the docs are automatically tested in CI. Test them locally:

**Step 1: Set up the development environment**

```bash
# From docs-website directory
./scripts/setup-dev.sh 2.19.0  # or 'main' for latest
```

This script:
- Installs base dependencies (requests, toml)
- Generates `requirements.txt` for the specified Haystack version
- Installs all Haystack dependencies

**Step 2: Run snippet tests**

```bash
python scripts/test_python_snippets.py --verbose
```

**Optional flags:**
- `--paths docs versioned_docs` - Test specific directories
- `--timeout-seconds 30` - Set execution timeout
- `--allow-unsafe` - Allow potentially unsafe patterns

**Snippet markers:**

Control how snippets are tested using HTML comments:

````markdown
<!-- test-ignore -->
```python
# This snippet will be skipped
```

<!-- test-run -->
```python
# Force run even without imports
print("hello")
```

<!-- test-concept -->
```python
# Skip this conceptual example
@dataclass
class Example:
    ...
```

<!-- test-require-files: assets/data.json -->
```python
# Skip if required file is missing
with open("assets/data.json") as f:
    data = json.load(f)
```
````

**How it works:**
- Extracts all Python code blocks from `.md` and `.mdx` files
- Runs each snippet in isolation via subprocess
- Heuristically skips conceptual snippets (no imports)
- Reports failures with file and line information

See `scripts/test_python_snippets.py` docstring for detailed usage.

### Prose and Style Linting with Vale

Vale runs automatically in CI on pull requests. Run it locally to catch issues early:

```bash
# From repository root
vale --config .vale.ini "**/*.{md,mdx}"

# Or from docs-website directory
vale --config .vale.ini "docs/**/*.{md,mdx}"
```

**Vale configuration:**

- Config file: `.vale.ini`
- Styles: `.vale/styles/` (Google + custom Haystack rules)
- Minimum alert level: suggestion

**CI behavior:**

- Runs on all PRs and pushes to `main`
- Creates GitHub PR review comments on issues
- Does not fail the build (set to `fail_on_error: false`)
- Shows errors, warnings, and suggestions as annotations

**Common Vale rules:**

- Google.FirstPerson (avoid "I", "we")
- Google.Passive (prefer active voice)
- Google.WordList (use recommended terminology)
- MyStyle.Branding (capitalize product names correctly)
- MyStyle.WeakWords (avoid "just", "simply", etc.)

If you encounter a StylesPath error, ensure the path in `.vale.ini` matches the repository layout.

## API Reference Contributions

The API reference documentation is automatically generated from Python docstrings in the main Haystack codebase.

**To update API documentation:**

1. Edit docstrings in the [Haystack repository](https://github.com/deepset-ai/haystack)
2. Open a PR in the main Haystack repo
3. After merge, the API reference will be automatically synced via CI

**Do not:**
- Manually edit files in `reference/` or `reference_versioned_docs/`
- Commit changes to auto-generated API documentation
- Any manual changes will be overwritten by the next sync

## Versioning

Documentation versions correspond to Haystack releases. Use the Docusaurus CLI to manage versions.

### Creating a New Version

**Step 1: Create a version for main docs**

```bash
npm run docusaurus -- docs:version 2.20.0
```

This command:
- Creates `versioned_docs/version-2.20.0/`
- Creates `versioned_sidebars/version-2.20.0-sidebars.json`
- Updates `versions.json`

**Step 2: Create a version for API reference**

```bash
npm run docusaurus -- docs:version 2.20.0 -- --id reference
```

This command:
- Creates `reference_versioned_docs/version-2.20.0/`
- Creates `reference_versioned_sidebars/version-2.20.0-sidebars.json`
- Updates `reference_versions.json`

**Note:** The actual versioning workflow is typically handled by maintainers through automated CI/CD scripts during release.

### Version Management

Version configuration in `docusaurus.config.js`:

```javascript
versions: {
  current: {
    label: '2.20-unstable',  // Label for the "next" version
    path: 'next',             // URL path
    banner: 'unreleased',     // Shows banner
  },
},
lastVersion: '2.19',          // Default version shown
```

**Available versions:**
- `current` (or `next`): Development version from `docs/` and `reference/`
- Stable versions: Listed in `versions.json` and `reference_versions.json`

See the [Docusaurus versioning documentation](https://docusaurus.io/docs/versioning) for more details.

### Understanding Documentation Versions and Where to Make Changes

The documentation structure supports multiple Haystack versions. Understanding where to make your changes is crucial:

**Documentation directories:**
- `docs/` - Unstable/next version (corresponds to Haystack's `main` branch)
- `versioned_docs/version-2.19/` - Current stable release documentation
- `versioned_docs/version-2.18/` - Previous release documentation
- And so on for older versions

**When to edit which version:**

**Scenario 1: New feature or change in Haystack main branch**

If you're documenting a new feature or change that exists in Haystack's `main` branch (next release):

✅ Edit files in `docs/` (the unstable version)

Example: A new component was added to Haystack main → document it in `docs/pipeline-components/`

**Scenario 2: Bug fix or correction for current release**

If you're fixing an error in the current release documentation (for example, incorrect information, broken link, typo):

✅ Edit files in BOTH locations:
1. `docs/` (so the fix persists in future versions)
2. `versioned_docs/version-2.19/` (or whichever version is current)

Example: A code example has a bug in the Pipelines guide → fix it in both `docs/concepts/pipelines.mdx` AND `versioned_docs/version-2.19/concepts/pipelines.mdx`

**How to check the current version:**

Look at `versions.json` - the first version listed is the current stable release:

```json
[
  "2.19",  // Current release
  "2.18"   // Previous release
]
```

**Pro tip:** When fixing bugs in current release docs, make the change in `docs/` first, then copy it to the versioned directory to ensure consistency.

## CI/CD Workflows

The documentation site includes several GitHub Actions workflows (located in `.github/workflows/` at the repository root).

### Vale Linting

**Workflow:** `docs-website-vale.yml`

**Triggers:**
- Pull requests that modify `docs-website/**`
- Pushes to `main` branch

**Actions:**
- Checks out repository
- Sets up Node.js 20
- Installs `mdx2vast` for MDX support
- Runs Vale on `docs/` and `versioned_docs/`
- Posts review comments on the PR
- Does not fail on errors (set to `continue-on-error: true`)

### Python Snippet Testing

**Workflow:** `docs-website-test-docs-snippets.yml`

**Triggers:**
- Daily scheduled run (03:17 UTC)
- Manual workflow dispatch with version input

**Actions:**
- Sets up Python 3.11
- Installs base dependencies (requests, toml)
- Generates `requirements.txt` for specified Haystack version
- Installs all dependencies
- Runs `test_python_snippets.py` with specified paths
- Reports failures as GitHub annotations

**Note:** Currently configured to test a limited set of files. Full testing will be enabled after migration.

### API Reference Sync

**Workflow:** `docusaurus_sync.yml`

**Triggers:**
- Workflow dispatch (manual)
- Pushes to `main` that modify Python code or docstring configs

**Actions:**
- Checks out Haystack repository
- Sets up Python and Hatch
- Generates API reference from docstrings
- Syncs to `docs-website/reference/haystack-api`
- Creates a pull request with changes

### Version Promotion

**Workflows:** `promote_unstable_docs.yml`, `minor_version_release.yml`

**Actions:**
- Automated during Haystack releases
- Creates new version directories
- Updates version configuration
- Creates pull requests with version changes

These workflows are typically triggered by maintainers during the release process.

## Preview Deployments

Pull requests that modify documentation may automatically generate preview deployments. Check your PR for a preview link, which allows reviewers to see your changes in a live environment before merging.

Preview deployments include:
- Full site build with your changes
- All versions and navigation
- Search functionality
- Identical to production except for the URL

## Troubleshooting

### Blank Page on npm start

If you see a blank page when running `npm start`:

```bash
# Clear Docusaurus cache
npm run clear
npm start
```

If the issue persists, build once to generate route metadata:

```bash
npm run build
npm start
```

This is necessary because Docusaurus needs to generate internal routing metadata for versioned docs on first run.

### Cache Issues

Clear cached data if something looks off:

```bash
npm run clear
```

This removes:
- `.docusaurus/` directory
- Build cache
- Generated metadata

### Build Errors

**Broken links:**
- Check that all internal links use correct relative paths
- Verify file names and paths match exactly (case-sensitive)
- Ensure linked pages have proper frontmatter with `id` field

**Duplicate routes:**
- Check for duplicate `slug` values in frontmatter
- Ensure no two pages map to the same URL path

**Missing images:**
- Verify image paths are correct
- Check that images exist in `static/img/` or local `assets/` directories
- Use relative paths from the markdown file location

### Vale StylesPath Error

If Vale reports a StylesPath error:

1. Ensure `.vale/styles/` directory exists
2. Check the path in `.vale.ini` matches your directory structure
3. Verify Vale style packages are present in `.vale/styles/`

## Moving or Removing Pages

**Moving a page:**

1. Keep the existing URL stable by retaining the `slug` in frontmatter
2. Update `sidebars.js` or `reference-sidebars.js` to reflect new file location
3. Update any internal links that reference the moved page

**Removing a page:**

1. Remove the file from `docs/`
2. Remove references from `sidebars.js`
3. Check for and update any links pointing to the removed page
4. Coordinate with maintainers for redirect setup if the URL was public

**If a URL must change:**
- Coordinate with maintainers to set up redirect rules
- Avoid breaking inbound links from external sites

## Images and Assets

**Placement:**
- Shared images: `static/img/`
- Per-section assets: `docs/<section>/assets/`

**Best practices:**
- Use descriptive filenames (for example, `pipeline-architecture.png`)
- Always include alt text: `![Pipeline architecture diagram](./assets/pipeline.png)`
- Optimize images before committing (use tools like ImageOptim, TinyPNG)
- Prefer modern formats (WebP, optimized PNG/JPEG)
- Docusaurus uses `@docusaurus/plugin-ideal-image` for responsive optimization

**Responsive images:**

Use the Image component for automatic optimization:

```jsx
import Image from '@theme/IdealImage';
import thumbnail from './assets/thumbnail.png';

<Image img={thumbnail} />
```

## Pull Request Process

### Pull Request Checklist

Before submitting your PR, verify:

- [ ] Content follows writing and style guidelines
- [ ] Navigation updated (`sidebars.js` or `reference-sidebars.js`)
- [ ] Internal links verified (no broken anchors)
- [ ] Code samples tested and include language tags
- [ ] Images optimized and include alt text
- [ ] Local build passes (`npm run build`)
- [ ] Vale checks pass or issues are addressed
- [ ] Python snippets are tested (if applicable)
- [ ] Conventional commit message format used in PR title
- [ ] PR description includes context and related issues

**PR title format:**

Use conventional commits in the PR title:

```
docs: add troubleshooting guide for pipelines
docs: fix typo in installation instructions
docs: update API reference links
```

**PR description:**

Include:
- Summary of changes
- Screenshots (if UI changes are visible)
- Related issues (for example, "Fixes #123")
- Testing performed
- Notes for reviewers

## Review Process

1. Open a PR from your branch to `main`
2. Automated checks will run (Vale linting, build validation)
3. Maintainers will review your changes
4. Address any requested changes
5. Once approved and checks pass, a maintainer will merge
6. Your changes will be deployed automatically

**What reviewers check:**
- Technical accuracy
- Writing style and clarity
- Completeness
- Link validity
- Code snippet correctness
- Adherence to guidelines

## Accessibility and Inclusivity

Ensure your documentation is accessible to all users:

- **Alt text:** Provide descriptive alt text for all images
- **Link text:** Use descriptive link text (not "click here")
- **Language:** Use clear, concise sentences; avoid jargon where possible
- **Examples:** Use inclusive language and diverse examples
- **Headings:** Use proper heading hierarchy (don't skip levels)
- **Code blocks:** Include language tags for proper syntax highlighting

## Getting Help

**Questions about contributing:**
- Review this guide and the [README](./README.md)
- Check existing [issues](https://github.com/deepset-ai/haystack/issues) and [discussions](https://github.com/deepset-ai/haystack/discussions)
- Ask in the [Discord community](https://discord.com/invite/haystack)

**Technical issues:**
- Search existing issues first
- Open a new issue with the `documentation` label
- Provide reproduction steps and environment details

**Style or writing questions:**
- Refer to the [Google Developer Documentation Style Guide](https://developers.google.com/style)
- Check Vale output for specific style issues
- Ask maintainers for clarification in your PR

Thank you for contributing to Haystack documentation! Your efforts help make Haystack more accessible and easier to use for everyone.
