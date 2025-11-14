# Haystack Documentation Website

This directory contains the Docusaurus-powered documentation website for [Haystack](https://github.com/deepset-ai/haystack), an open-source framework for building production-ready applications with Large Language Models (LLMs).

- **Vercel production deployment:** https://haystack-docs.vercel.app/docs/intro
- **Live site:** https://docs.haystack.deepset.ai

**Table of Contents**

- [About](#about)
- [Key Features](#key-features)
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
- [Related Links](#related-links)
- [License](#license)
- [Support](#support)

## About

This documentation site is built with Docusaurus 3 and provides comprehensive guides, tutorials, API references, and best practices for using Haystack. The site supports multiple versions and automated API reference generation.

## Key Features

- **Versioned Documentation**: Multiple Haystack versions with dropdown navigation
- **Dual Documentation Plugins**: Separate sections for narrative docs (`docs/`) and API reference (`reference/`)
- **Automated API Generation**: Python docstrings automatically synced from the main Haystack codebase
- **Live Reload**: Vercel development server with instant preview of changes
- **Optimized Images**: Responsive image processing for faster page loads

## Prerequisites

- **Node.js** 18 or higher
- **npm** (included with Node.js) or Yarn
- Optional: [Vale](https://vale.sh/) (for local prose linting)

## Quick Start

**Important:** All commands must be run from the `haystack/docs-website` directory.

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

The documentation site is automatically deployed to **https://haystack-docs.vercel.app/docs/intro** (in future https://docs.haystack.deepset.ai) when changes are merged to the `main` branch.

## Related Links

- **Main Haystack Repository**: https://github.com/deepset-ai/haystack
- **Haystack Documentation**: https://docs.haystack.deepset.ai
- **Discord Community**: https://discord.com/invite/haystack
- **Twitter**: https://twitter.com/haystack_ai
- **Tutorials**: https://haystack.deepset.ai/tutorials
- **Code of Conduct**: https://github.com/deepset-ai/haystack/blob/main/code_of_conduct.txt

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/deepset-ai/haystack/blob/main/LICENSE) file in the main Haystack repository for details.

© 2025 deepset GmbH. All rights reserved.

## Support

- **Questions**: Join our [Discord community](https://discord.com/invite/haystack)
- **Bug Reports**: [Open an issue](https://github.com/deepset-ai/haystack/issues/new?template=bug_report.md)
- **Feature Requests**: [Start a discussion](https://github.com/deepset-ai/haystack/discussions)
- **Documentation Issues**: [Open an issue](https://github.com/deepset-ai/haystack/issues/new) with the `documentation` label
