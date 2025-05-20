<div align="center">
  <a href="https://haystack.deepset.ai/"><img src="https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/banner.png" alt="Green logo of a stylized white 'H' with the text 'Haystack, by deepset.' Abstract green and yellow diagrams in the background."></a>

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CI/CD   | [![Tests](https://github.com/deepset-ai/haystack/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/tests.yml) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![Coverage Status](https://coveralls.io/repos/github/deepset-ai/haystack/badge.svg?branch=main)](https://coveralls.io/github/deepset-ai/haystack?branch=main) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Docs    | [![Website](https://img.shields.io/website?label=documentation&up_message=online&url=https%3A%2F%2Fdocs.haystack.deepset.ai)](https://docs.haystack.deepset.ai)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Package | [![PyPI](https://img.shields.io/pypi/v/haystack-ai)](https://pypi.org/project/haystack-ai/) ![PyPI - Downloads](https://img.shields.io/pypi/dm/haystack-ai?color=blue&logo=pypi&logoColor=gold) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/haystack-ai?logo=python&logoColor=gold) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/haystack-ai.svg)](https://anaconda.org/conda-forge/haystack-ai) [![GitHub](https://img.shields.io/github/license/deepset-ai/haystack?color=blue)](LICENSE) [![License Compliance](https://github.com/deepset-ai/haystack/actions/workflows/license_compliance.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/license_compliance.yml) |
| Meta    | [![Discord](https://img.shields.io/discord/993534733298450452?logo=discord)](https://discord.com/invite/xYvH6drSmA) [![Twitter Follow](https://img.shields.io/twitter/follow/haystack_ai)](https://twitter.com/haystack_ai)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
</div>

[Haystack](https://haystack.deepset.ai/) is an end-to-end LLM framework that allows you to build applications powered by
LLMs, Transformer models, vector search and more. Whether you want to perform retrieval-augmented generation (RAG),
document search, question answering or answer generation, Haystack can orchestrate state-of-the-art embedding models
and LLMs into pipelines to build end-to-end NLP applications and solve your use case.

## Installation

The simplest way to get Haystack is via pip:

```sh
pip install haystack-ai
```

Install from the `main` branch to try the newest features:
```sh
pip install git+https://github.com/deepset-ai/haystack.git@main
```

Haystack supports multiple installation methods including Docker images. For a comprehensive guide please refer
to the [documentation](https://docs.haystack.deepset.ai/docs/installation).

## Documentation

If you're new to the project, check out ["What is Haystack?"](https://haystack.deepset.ai/overview/intro) then go
through the ["Get Started Guide"](https://haystack.deepset.ai/overview/quick-start) and build your first LLM application
in a matter of minutes. Keep learning with the [tutorials](https://haystack.deepset.ai/tutorials). For more advanced
use cases, or just to get some inspiration, you can browse our Haystack recipes in the
[Cookbook](https://haystack.deepset.ai/cookbook).

At any given point, hit the [documentation](https://docs.haystack.deepset.ai/docs/intro) to learn more about Haystack, what can it do for you and the technology behind.

## Features

- **Technology agnostic:** Allow users the flexibility to decide what vendor or technology they want and make it easy to switch out any component for another. Haystack allows you to use and compare models available from OpenAI, Cohere and Hugging Face, as well as your own local models or models hosted on Azure, Bedrock and SageMaker.
- **Explicit:** Make it transparent how different moving parts can “talk” to each other so it's easier to fit your tech stack and use case.
- **Flexible:** Haystack provides all tooling in one place: database access, file conversion, cleaning, splitting, training, eval, inference, and more. And whenever custom behavior is desirable, it's easy to create custom components.
- **Extensible:** Provide a uniform and easy way for the community and third parties to build their own components and foster an open ecosystem around Haystack.

Some examples of what you can do with Haystack:

-   Build **retrieval augmented generation (RAG)** by making use of one of the available vector databases and customizing your LLM interaction, the sky is the limit 🚀
-   Perform Question Answering **in natural language** to find granular answers in your documents.
-   Perform **semantic search** and retrieve documents according to meaning.
-   Build applications that can make complex decisions making to answer complex queries: such as systems that can resolve complex customer queries, do knowledge search on many disconnected resources and so on.
-   Scale to millions of docs using retrievers and production-scale components.
-   Use **off-the-shelf models** or **fine-tune** them to your data.
-   Use **user feedback** to evaluate, benchmark, and continuously improve your models.

> [!TIP]
><img src="https://github.com/deepset-ai/haystack/raw/main/docs/img/deepset-platform-logo-alternative.jpeg"  width=30% height=30%>
>
> Are you looking for a managed solution that benefits from Haystack? [deepset AI Platform](https://www.deepset.ai/products-and-services/deepset-ai-platform?utm_campaign=developer-relations&utm_source=haystack&utm_medium=readme) is our fully managed, end-to-end platform to integrate LLMs with your data, which uses Haystack for the LLM pipelines architecture.

> [!TIP]
>
> Would you like to deploy and serve Haystack pipelines as REST APIs yourself? [Hayhooks](https://github.com/deepset-ai/hayhooks) provides a simple way to wrap your pipelines with custom logic and expose them via HTTP endpoints, including OpenAI-compatible chat completion endpoints and compatibility with fully-featured chat interfaces like [open-webui](https://openwebui.com/).

## 🆕 deepset Studio: Your Development Environment for Haystack

Use **deepset Studio** to visually create, deploy, and test your Haystack pipelines. Learn more about it in [our announcement post](https://haystack.deepset.ai/blog/announcing-studio).

![studio](https://github.com/user-attachments/assets/e4f09746-20b5-433e-8261-eca224ac23b3)


👉 [Sign up](https://landing.deepset.ai/deepset-studio-signup)!

## Telemetry

Haystack collects **anonymous** usage statistics of pipeline components. We receive an event every time these components are initialized. This way, we know which components are most relevant to our community.

Read more about telemetry in Haystack or how you can opt out in [Haystack docs](https://docs.haystack.deepset.ai/docs/telemetry).

## 🖖 Community

If you have a feature request or a bug report, feel free to open an [issue in Github](https://github.com/deepset-ai/haystack/issues). We regularly check these and you can expect a quick response. If you'd like to discuss a topic, or get more general advice on how to make Haystack work for your project, you can start a thread in [Github Discussions](https://github.com/deepset-ai/haystack/discussions) or our [Discord channel](https://discord.com/invite/VBpFzsgRVF). We also check [𝕏 (Twitter)](https://twitter.com/haystack_ai) and [Stack Overflow](https://stackoverflow.com/questions/tagged/haystack).

## Contributing to Haystack

We are very open to the community's contributions - be it a quick fix of a typo, or a completely new feature! You don't need to be a Haystack expert to provide meaningful improvements. To learn how to get started, check out our [Contributor Guidelines](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md) first.

There are several ways you can contribute to Haystack:
- Contribute to the main Haystack project
- Contribute an integration on [haystack-core-integrations](https://github.com/deepset-ai/haystack-core-integrations)

> [!TIP]
>👉 **[Check out the full list of issues that are open to contributions](https://github.com/orgs/deepset-ai/projects/14)**

## Who Uses Haystack

Here's a list of projects and companies using Haystack. Want to add yours? Open a PR, add it to the list and let the
world know that you use Haystack!

-   [Airbus](https://www.airbus.com/en)
-   [Alcatel-Lucent](https://www.al-enterprise.com/)
-   [Apple](https://www.apple.com/)
-   [BetterUp](https://www.betterup.com/)
-   [Databricks](https://www.databricks.com/)
-   [Deepset](https://deepset.ai/)
-   [Etalab](https://www.deepset.ai/blog/improving-on-site-search-for-government-agencies-etalab)
-   [Infineon](https://www.infineon.com/)
-   [Intel](https://github.com/intel/open-domain-question-and-answer#readme)
-   [Intelijus](https://www.intelijus.ai/)
-   [Intel Labs](https://github.com/IntelLabs/fastRAG#readme)
-   [LEGO](https://github.com/larsbaunwall/bricky#readme)
-   [Netflix](https://netflix.com)
-   [NOS Portugal](https://www.nos.pt/en/welcome)
-   [Nvidia](https://developer.nvidia.com/blog/reducing-development-time-for-intelligent-virtual-assistants-in-contact-centers/)
-   [PostHog](https://github.com/PostHog/max-ai#readme)
-   [Rakuten](https://www.rakuten.com/)
-   [Sooth.ai](https://www.deepset.ai/blog/advanced-neural-search-with-sooth-ai)
