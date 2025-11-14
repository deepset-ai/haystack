# Contributing to Haystack Documentation

Thank you for your interest in contributing to the Haystack documentation! This guide provides everything you need to write, review, and maintain high-quality documentation for the Haystack project.

This guide focuses specifically on documentation contributions. For code contributions, tests, or integrations in the main Haystack codebase, see the [main Haystack contribution guide](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md).

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

**Note:** All subsequent commands in this guide should be run from the `haystack/docs-website` directory unless otherwise specified.

3. Edit under `docs/` for the unstable version, and under `versioned_docs/version-<highest>/` for the latest stable release. If you add a new page, include its ID in `sidebars.js` or the appropriate versioned sidebar.

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

6. Open a PR and review the [Pull Request Checklist](#pull-request-checklist).

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
  - [Prose and Style Linting with Vale](#prose-and-style-linting-with-vale)
- [API Reference Contributions](#api-reference-contributions)
- [Understanding Documentation Versions and Where to Make Changes](#understanding-documentation-versions-and-where-to-make-changes)
- [CI/CD Workflows](#cicd-workflows)
  - [Vale Linting](#vale-linting)
  - [API Reference Sync](#api-reference-sync)
- [Preview Deployments](#preview-deployments)
- [Troubleshooting](#troubleshooting)
  - [Blank Page on npm start](#blank-page-on-npm-start)
  - [Cache Issues](#cache-issues)
  - [Build Errors](#build-errors)
  - [Vale Errors](#vale-errors)
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
| Fix in current stable docs | `docs/` | `versioned_docs/version-<highest>/` (for example, `version-2.20`) |
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

Edit `reference-sidebars.js` if needed (however, most sections are auto-generated).

### Linking and Anchors

**Links within `docs/`:**

Use relative paths for links within the same documentation section:

```md
See the [Pipeline Guide](../concepts/pipelines.mdx)
See the [Components Overview](./components-overview.mdx)
```

**Links between `docs/` and `reference/`:**

Because `docs/` and `reference/` are separate Docusaurus plugin instances, you must use absolute paths when linking between them:

```md
<!-- From docs/ to reference/ -->
See the [Pipeline API Reference](/reference/haystack-api/pipelines/pipeline)

<!-- From reference/ to docs/ -->
See the [Pipeline Concepts Guide](/docs/concepts/pipelines)
```

**Note:** Always use `/docs/` or `/reference/` as the path prefix when linking across sections, not relative paths like `../../reference/`.

**Explicit anchors:**

For stable cross-links, use explicit heading IDs:

```markdown
## Installation {#install-guide}
```

Link to it: `[Install](./page.mdx#install-guide)` or `[Install](/docs/overview/quick-start#install-guide)` from `reference/`

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

### Prose and Style Linting with Vale

Vale is a prose linter that checks documentation for style consistency. It runs automatically in CI on all pull requests - you don't need to run it locally.

**CI behavior:**

- Runs automatically on all PRs and pushes to `main`
- Creates GitHub PR review comments on issues
- Does not fail the build
- Shows errors, warnings, and suggestions as annotations

**Common Vale rules:**

- Google.FirstPerson (avoid "I", "we")
- Google.Passive (prefer active voice)
- Google.WordList (use recommended terminology)
- MyStyle.Branding (capitalize product names correctly)
- MyStyle.WeakWords (avoid "just", "simply", and similar words)

**Running Vale locally (optional):**

If you want to check your prose before pushing, you can run Vale locally:

1. Install Vale CLI: https://vale.sh/docs/vale-cli/installation/
2. Navigate to the `docs-website/` directory
3. Download Vale styles: `vale sync`
4. Run Vale: `vale --config .vale.ini "docs/**/*.{md,mdx}"`

The GitHub Action will provide the final validation on your PR, so running Vale locally is completely optional.

## API Reference Contributions

The API reference documentation is automatically generated from Python docstrings in the main Haystack codebase.

**To update API documentation:**

1. Edit docstrings in the [Haystack repository](https://github.com/deepset-ai/haystack)
2. Open a PR in the main Haystack repo
3. After merge, the API reference will be automatically synced through CI

**Do not:**
- Manually edit files in `reference/` or `reference_versioned_docs/`
- Commit changes to auto-generated API documentation
- Any manual changes will be overwritten by the next sync

## Understanding Documentation Versions and Where to Make Changes

The documentation structure supports multiple Haystack versions. Haystack releases new versions monthly, and documentation versioning is handled automatically through GitHub workflows during the release process.

**Documentation directories:**
- `docs/` - Unstable/next version (corresponds to Haystack's `main` branch)
- `versioned_docs/version-X.Y/` - Stable release documentation for version X.Y

**Note:** The highest version number in `versioned_docs/` represents the current stable release. For example, if you see `version-2.20`, `version-2.19`, and `version-2.18`, then version 2.20 is the current stable release, and older versions are for reference.

**When to edit which version:**

**Scenario 1: New feature or change in Haystack main branch**

If you're documenting a new feature or change that exists in Haystack's `main` branch (next release):

✅ Edit files in `docs/` (the unstable version)

Example: A new component was added to Haystack main → document it in `docs/pipeline-components/`

**Scenario 2: Bug fix or correction for current release**

If you're fixing an error in the current release documentation (for example, incorrect information, broken link, typo):

✅ Edit files in BOTH locations:
1. `docs/` (so the fix persists in future versions)
2. `versioned_docs/version-<highest>/` (the highest-numbered version directory)

Example: A code example has a bug in the Pipelines guide → fix it in both `docs/concepts/pipelines.mdx` AND `versioned_docs/version-2.20/concepts/pipelines.mdx` (if 2.20 is the current stable release)

**Pro tip:** When fixing bugs in current release docs, make the change in `docs/` first, then copy it to the highest-numbered versioned directory to ensure consistency.

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

## Preview Deployments

Pull requests that modify documentation may automatically generate preview deployments. Check your PR for a preview link, which allows reviewers to see your changes in a live environment before merging.

Preview deployments include:
- Full site build with your changes
- All versions and navigation
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

### Vale Errors

If you see Vale-related errors when running locally, ensure you've run `vale sync` from the `docs-website/` directory to download the required style packages. The Vale GitHub Action in CI automatically handles this, so local setup issues won't affect your PR validation.

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

Shared images are stored in `static/img/`.

**Best practices:**
- Use descriptive filenames (for example, `pipeline-architecture.png`)
- Optimize images before committing (use tools like ImageOptim, TinyPNG)
- Prefer modern formats (WebP, optimized PNG/JPEG)
- Always include alt text for accessibility

**Adding images:**

Use the `ClickableImage` component for all images. Import it at the top of your MDX file:

```mdx
import ClickableImage from "@site/src/components/ClickableImage";

<ClickableImage
  src="/img/pipeline-architecture.png"
  alt="Pipeline architecture diagram"
/>
```

**For zoomable images** (diagrams, screenshots that users may want to see in detail), use `size="large"`:

```mdx
<ClickableImage
  src="/img/detailed-architecture.png"
  alt="Detailed architecture diagram"
  size="large"
/>
```

**Images with transparent backgrounds:**

For transparent PNGs that need better visibility in dark mode, add a background class:

```mdx
<!-- White background in dark mode -->
<div className="img-white-bg">
  <ClickableImage src="/img/logo.png" alt="Logo" />
</div>

<!-- Light grey background (softer) -->
<div className="img-light-bg">
  <ClickableImage src="/img/diagram.png" alt="Diagram" />
</div>
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
- [ ] Vercel preview deployment succeeds (fix any deployment errors)
- [ ] Vale checks pass or issues are addressed
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
- Code correctness
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
