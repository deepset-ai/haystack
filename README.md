<div align="center">
  <a href="https://haystack.deepset.ai/"><img src="https://raw.githubusercontent.com/deepset-ai/haystack/main/images/banner.png" alt="Blue banner with the Haystack logo and the text ‘haystack by deepset – The Open Source AI Framework for Production Ready RAG & Agents’ surrounded by abstract icons representing search, documents, agents, pipelines, and cloud systems."></a>

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CI/CD   | [![Tests](https://github.com/deepset-ai/haystack/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/tests.yml) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![Coverage Status](https://coveralls.io/repos/github/deepset-ai/haystack/badge.svg?branch=main)](https://coveralls.io/github/deepset-ai/haystack?branch=main) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Docs    | [![Website](https://img.shields.io/website?label=documentation&up_message=online&url=https%3A%2F%2Fdocs.haystack.deepset.ai)](https://docs.haystack.deepset.ai)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Package | [![PyPI](https://img.shields.io/pypi/v/haystack-ai)](https://pypi.org/project/haystack-ai/) ![PyPI - Downloads](https://img.shields.io/pypi/dm/haystack-ai?color=blue&logo=pypi&logoColor=gold) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/haystack-ai?logo=python&logoColor=gold) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/haystack-ai.svg)](https://anaconda.org/conda-forge/haystack-ai) [![GitHub](https://img.shields.io/github/license/deepset-ai/haystack?color=blue)](LICENSE) [![License Compliance](https://github.com/deepset-ai/haystack/actions/workflows/license_compliance.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/license_compliance.yml) |
| Meta    | [![Discord](https://img.shields.io/discord/993534733298450452?logo=discord)](https://discord.com/invite/xYvH6drSmA) [![Twitter Follow](https://img.shields.io/twitter/follow/haystack_ai)](https://twitter.com/haystack_ai)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
</div>

[Haystack](https://haystack.deepset.ai/) is an open-source AI orchestration framework for building production-ready LLM applications in Python.

Design modular pipelines and agent workflows with explicit control over retrieval, routing, memory, and generation. Build scalable RAG systems, multimodal applications, semantic search, question answering, and autonomous agents, all in a transparent architecture that lets you experiment, customize deeply, and deploy with confidence.

## Table of Contents

- [Installation](#installation)
- [Documentation](#documentation)
- [Features](#features)
- [Haystack Enterprise: Support & Platform](#haystack-enterprise-support--platform)
- [Telemetry](#telemetry)
- [🖖 Community](#-community)
- [Contributing to Haystack](#contributing-to-haystack)
- [Organizations using Haystack](#organizations-using-haystack)


## Installation

The simplest way to get Haystack is via pip:

```sh
pip install haystack-ai
```

Install from the `main` branch to try the newest features:
```sh
pip install git+https://github.com/deepset-ai/haystack.git@main
```

Haystack supports multiple installation methods, including Docker images. For a comprehensive guide, please refer
to the [documentation](https://docs.haystack.deepset.ai/docs/installation).

## Documentation

If you're new to the project, check out ["What is Haystack?"](https://haystack.deepset.ai/overview/intro) then go
through the ["Get Started Guide"](https://haystack.deepset.ai/overview/quick-start) and build your first LLM application
in a matter of minutes. Keep learning with the [tutorials](https://haystack.deepset.ai/tutorials). For more advanced
use cases, or just to get some inspiration, you can browse our Haystack recipes in the
[Cookbook](https://haystack.deepset.ai/cookbook).

At any given point, hit the [documentation](https://docs.haystack.deepset.ai/docs/intro) to learn more about Haystack, what it can do for you, and the technology behind.

## Features

**Built for context engineering**  
Design flexible systems with explicit control over how information is retrieved, structured, routed, and evaluated before it reaches the model. Define pipelines and agent workflows where retrieval, memory, tools, and generation are transparent and traceable.

**Model- and vendor-agnostic**  
Integrate with OpenAI, Mistral, Anthropic, Cohere, Hugging Face, Azure OpenAI, AWS Bedrock, local models, and many others. Swap models or infrastructure components without rewriting your system.

**Modular and customizable**  
Use built-in components for retrieval, indexing, tool calling, memory, evaluation, and deployment, or create your own. Add loops, branches, and conditional logic to precisely control how context moves through your pipelines and agent workflows.

**Extensible ecosystem**  
Build and share custom components through a consistent interface that makes it easy for the community and third parties to extend Haystack and contribute to an open ecosystem.

> [!TIP]
> 
> Would you like to deploy and serve Haystack pipelines as **REST APIs** or **MCP servers**? [Hayhooks](https://github.com/deepset-ai/hayhooks) provides a simple way for you to wrap pipelines and agents with custom logic and expose them through HTTP endpoints or MCP. It also supports OpenAI-compatible chat completion endpoints and works with chat UIs like [open-webui](https://openwebui.com/).

## Haystack Enterprise: Support & Platform

Get expert support from the Haystack team, build faster with enterprise-grade templates, and scale securely with deployment guides for cloud and on-prem environments with **Haystack Enterprise Starter**. Read more about it in the [announcement post](https://haystack.deepset.ai/blog/announcing-haystack-enterprise).

👉 [Get Haystack Enterprise Starter](https://www.deepset.ai/products-and-services/haystack-enterprise-starter?utm_source=github.com&utm_medium=referral&utm_campaign=haystack_enterprise)

Need a managed production setup for Haystack? The **Haystack Enterprise Platform** helps you deploy and operate Haystack pipelines with built-in observability, governance, and access controls. It’s available as a managed cloud service or as a self-hosted solution.

👉 Learn more about [Haystack Enterprise Platform](https://www.deepset.ai/products-and-services/haystack-enterprise-platform?utm_campaign=developer-relations&utm_source=haystack&utm_medium=readme) or [try it free](https://www.deepset.ai/haystack-enterprise-platform-trial?utm_campaign=developer-relations&utm_source=haystack&utm_medium=readme)

## Telemetry

Haystack collects **anonymous** usage statistics of pipeline components. We receive an event every time these components are initialized. This way, we know which components are most relevant to our community.

Read more about telemetry in Haystack or how you can opt out in [Haystack docs](https://docs.haystack.deepset.ai/docs/telemetry).

## 🖖 Community

If you have a feature request or a bug report, feel free to open an [issue in GitHub](https://github.com/deepset-ai/haystack/issues). We regularly check these, so you can expect a quick response. If you'd like to discuss a topic or get more general advice on how to make Haystack work for your project, you can start a thread in [Github Discussions](https://github.com/deepset-ai/haystack/discussions) or our [Discord channel](https://discord.com/invite/VBpFzsgRVF). We also check [𝕏 (Twitter)](https://twitter.com/haystack_ai) and [Stack Overflow](https://stackoverflow.com/questions/tagged/haystack).

## Contributing to Haystack

We are very open to the community's contributions - be it a quick fix of a typo, or a completely new feature! You don't need to be a Haystack expert to provide meaningful improvements. To learn how to get started, check out our [Contributor Guidelines](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md) first.

There are several ways you can contribute to Haystack:
- Contribute to the main Haystack project
- Contribute an integration on [haystack-core-integrations](https://github.com/deepset-ai/haystack-core-integrations)
- Contribute to the documentation in [haystack/docs-website](https://github.com/deepset-ai/haystack/tree/main/docs-website)

> [!TIP]
>👉 **[Check out the full list of issues that are open to contributions](https://github.com/orgs/deepset-ai/projects/14)**

## Organizations using Haystack

Haystack is used by thousands of teams building production AI systems across industries, including:

- **Technology & AI Infrastructure**: [Apple](https://www.apple.com/), [Meta](https://www.meta.com/about), [Databricks](https://www.databricks.com/), [NVIDIA](https://developer.nvidia.com/blog/reducing-development-time-for-intelligent-virtual-assistants-in-contact-centers/), [Intel](https://github.com/intel/open-domain-question-and-answer#readme)
- **Public Sector AI Initiatives**: [European Commission](https://commission.europa.eu/index_en), [German Federal Ministry of Research, Technology, and Space (BMFTR)](https://www.deepset.ai/case-studies/german-federal-ministry-research-technology-space-bmftr), [PD, Baden-Württemberg State](https://www.pd-g.de/)
- **Enterprise & Industrial AI Applications**: [Airbus](https://www.deepset.ai/case-studies/airbus), [Lufthansa Industry Solutions](https://haystack.deepset.ai/blog/lufthansa-user-story), [Infineon](https://www.infineon.com/), [LEGO](https://github.com/larsbaunwall/bricky#readme), [Comcast](https://arxiv.org/html/2405.00801v2), [Accenture](https://www.accenture.com/), [TELUS Agriculture & Consumer Goods](https://www.telus.com/agcg/en)
- **Knowledge & Content Platforms**: [Netflix](https://netflix.com), [ZEIT Online](https://www.deepset.ai/case-studies/zeit-online), [Rakuten](https://www.rakuten.com/), [Oxford University Press](https://corp.oup.com/), [Manz](https://www.deepset.ai/case-studies/manz), [YPulse](https://www.deepset.ai/case-studies/ypulse)


Are you also using Haystack? Open a PR or [tell us your story](https://forms.gle/Mm3G1aEST3GAH2rn8)
