# Haystack Documentation Website

This directory contains the Docusaurus-powered documentation website for [Haystack](https://github.com/deepset-ai/haystack), an open-source framework for building production-ready applications with Large Language Models (LLMs).

- **Website URL:** https://docs.haystack.deepset.ai

**Table of Contents**

- [About](#about)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Common tasks](#common-tasks)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Available Scripts](#available-scripts)
- [Contributing](#contributing)
- [CI/CD and Automation](#cicd-and-automation)
  - [Versioning](#versioning)
- [Deployment](#deployment)
- [llms.txt for AI tools](#llms.txt-for-ai-tools)

## About

This documentation site is built with Docusaurus 3 and provides comprehensive guides, tutorials, API references, and best practices for using Haystack. The site supports multiple versions and automated API reference generation.

## Prerequisites

- **Node.js** 18 or higher
- **npm** (included with Node.js) or Yarn

## Quick Start

> [!NOTE]
> All commands must be run from the `haystack/docs-website` directory.

```bash
# Clone the repository and navigate to docs-website
git clone https://github.com/deepset-ai/haystack.git
cd haystack/docs-website

# Install dependencies
npm install

# Start the development server
npm start

# The site opens at http://localhost:3000 with live reload
```

## Common tasks

- Edit a page: update files under `docs/` or `versioned_docs/` and preview at http://localhost:3000
- Add to sidebar: update `sidebars.js` with your doc ID
- Production check: `npm run build && npm run serve`
- Prose lint (optional): `vale --config .vale.ini "docs/**/*.{md,mdx}"`
- Full guidance: see `CONTRIBUTING.md`

## Project Structure

```
docs-website/
├── docs/                          # Main documentation (guides, tutorials, concepts)
│   ├── _templates/               # Authoring templates (excluded from build)
│   ├── concepts/                 # Core Haystack concepts
│   ├── pipeline-components/      # Component documentation
│   └── ...
├── reference/                     # API reference (auto-generated, do not edit manually)
├── versioned_docs/               # Versioned copies of docs/
├── reference_versioned_docs/     # Versioned copies of reference/
├── src/                          # React components and custom code
│   ├── components/              # Custom React components
│   ├── css/                     # Global styles
│   ├── pages/                   # Custom pages
│   ├── remark/                  # Remark plugins
│   └── theme/                   # Docusaurus theme customizations
├── static/                       # Static assets (images, files)
├── scripts/                      # Build and test scripts
│   ├── generate_requirements.py # Generates Python dependencies
│   ├── setup-dev.sh             # Development environment setup
│   └── test_python_snippets.py  # Tests Python code in docs
├── sidebars.js                   # Navigation for docs/
├── reference-sidebars.js         # Navigation for reference/
├── docusaurus.config.js          # Main Docusaurus configuration
├── versions.json                 # Available docs versions
├── reference_versions.json       # Available API reference versions
└── package.json                  # Node.js dependencies and scripts
```

## Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| [Docusaurus](https://docusaurus.io/) | 3.8.1 | Static site generator |
| [React](https://react.dev/) | 19.0.0 | UI framework |
| [MDX](https://mdxjs.com/) | 3.0.0 | Markdown with JSX |
| [Node.js](https://nodejs.org/) | ≥18.0 | Runtime environment |
| [Vale](https://vale.sh/) | Latest | Prose linting |

**Key Docusaurus Plugins:**
- `@docusaurus/plugin-content-docs` (dual instances for docs and API reference)
- Custom remark plugins for versioned reference links

## Available Scripts

**Important:** Run these commands from the `haystack/docs-website` directory:

| Command | Description |
|---------|-------------|
| `npm install` | Install all dependencies |
| `npm start` | Start development server with live reload (http://localhost:3000) |
| `npm run build` | Build production-ready static files to `build/` |
| `npm run serve` | Preview production build locally |
| `npm run clear` | Clear Docusaurus cache (use if encountering build issues) |
| `npm run docusaurus` | Run Docusaurus CLI commands directly |
| `npm run swizzle` | Eject and customize Docusaurus theme components |

## Contributing

We welcome contributions to improve the documentation! See [CONTRIBUTING.md](./CONTRIBUTING.md) for:

- Writing and style guidelines
- How to author new documentation pages
- Setting up your development environment
- Testing requirements
- Pull request process

For code contributions to Haystack itself, see the [main repository's contribution guide](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md).

## CI/CD and Automation

This site uses automated workflows for prose linting, API reference sync, and preview deployments. See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

### Versioning

Documentation versions are released alongside Haystack releases and are fully automated through GitHub workflows. Contributors do not need to manually create or manage versions.

**Automated Workflows:**
- `promote_unstable_docs.yml` - Automatically triggered during Haystack releases
- `minor_version_release.yml` - Creates new version directories and updates version configuration

These workflows automatically create versioned documentation snapshots and pull requests during the release process.

## Deployment

The documentation site is automatically deployed to **https://docs.haystack.deepset.ai** when changes are merged to the `main` branch.

## llms.txt for AI tools

This docs site exposes a concatenated view of the documentation for AI tools with an `llms.txt` file, generated by the [`docusaurus-plugin-generate-llms-txt`](https://github.com/din0s/docusaurus-plugin-llms-txt) plugin.

- **What it is**: A single, generated text file that concatenates the docs content to make it easier for LLMs and other tools to consume.
- **Where to find it (deployed)**: At the site root `https://docs.haystack.deepset.ai/llms.txt`.
- **How it’s generated**:
  - Automatically when you run:
    - `npm run start`
    - `npm run build`
  - Manually with:

    ```bash
    npm run generate-llms-txt
    ```

- **Configuration**:
  - The plugin is wired in `docusaurus.config.js` under the `plugins` array as `'docusaurus-plugin-generate-llms-txt'` with `outputFile: 'llms.txt'`.
  - A local plugin (`plugins/txtLoaderPlugin.js`) configures Webpack to treat `.txt` files (including `llms.txt`) as text assets so they don’t cause build-time parse errors.*
