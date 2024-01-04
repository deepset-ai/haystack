<div align="center">
  <a href="https://www.deepset.ai/haystack/"><img src="https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/haystack_logo_colored.png" alt="Haystack"></a>

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![Tests](https://github.com/deepset-ai/haystack/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/tests.yml) [![code style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![Coverage Status](https://coveralls.io/repos/github/deepset-ai/haystack/badge.svg?branch=main)](https://coveralls.io/github/deepset-ai/haystack?branch=main) |
| Docs    |  [![Website](https://img.shields.io/website?label=documentation&up_message=online&url=https%3A%2F%2Fdocs.haystack.deepset.ai)](https://docs.haystack.deepset.ai)                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Package | [![PyPI](https://img.shields.io/pypi/v/haystack-ai)](https://pypi.org/project/haystack-ai/) ![PyPI - Downloads](https://img.shields.io/pypi/dm/haystack-ai?color=blue&logo=pypi&logoColor=gold) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/farm-haystack?logo=python&logoColor=gold) [![GitHub](https://img.shields.io/github/license/deepset-ai/haystack?color=blue)](LICENSE) [![License Compliance](https://github.com/deepset-ai/haystack/actions/workflows/license_compliance.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/license_compliance.yml)                                                                                                                                                                                            |
| Meta    | [![Discord](https://img.shields.io/discord/993534733298450452?logo=discord)](https://discord.gg/haystack) [![Twitter Follow](https://img.shields.io/twitter/follow/haystack_ai)](https://twitter.com/haystack_ai)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
</div>

> [!WARNING]
> You are currently looking at the readme of Haystack 2.0-Beta, an unstable version of what will eventually become Haystack 2.0. We are still maintaining Haystack 1.x which is the version of Haystack you should use in production. [Switch to Haystack 1.x, currently on 1.23 here](https://github.com/deepset-ai/haystack/tree/v1.x).

[Haystack](https://haystack.deepset.ai/) is an end-to-end LLM framework that enables you to build applications powered by LLMs, Transformer models, vector search and more. Whether you want to perform retrieval-augmented generation (RAG), documentation search, question answering or answer generation, you can use state-of-the-art embedding models and LLMs with Haystack to build end-to-end NLP applications to solve your use case.

## Quickstart

Haystack is built around the concept of pipelines. A pipeline is a powerful structure that performs an NLP task. It's made up of components connected together. For example, you can connect a [retriever](https://docs.haystack.deepset.ai/v2.0/docs/retrievers) and a [generator](https://docs.haystack.deepset.ai/v2.0/docs/generators) to build a Generative Question Answering pipeline that uses your own data.

First, run the minimal Haystack installation:

```sh
pip install haystack-ai
```
👉 To build a minimal RAG pipeline that uses GPT-4 on your own data, use the [RAG Pipeline Recipe](https://docs.haystack.deepset.ai/v2.0/docs/creating-pipelines#example)

## Core Concepts

⚛️ **[Components](https://docs.haystack.deepset.ai/v2.0/docs/components):** Each Component achieves one thing. Such as preprocessing documents, retrieving documents, using specific language models to answer questions, and so on. Components can `.connect()` to each other to form a complete pipeline.

🏃‍♀️ **[Pipelines](https://docs.haystack.deepset.ai/v2.0/docs/pipelines):** This is the standard Haystack structure that builds on top of your data to perform various NLP tasks such as retrieval augmented generation, question answering and more. Pipelines in Haystack are Directed Multigraphs composed of components. Components can receive inputs from other components and produce outputs that can be forwarded to other components. 

🗂️ **[DocumentStores](https://docs.haystack.deepset.ai/docs/document_store):** A DocumentStore is database where you store your text data for Haystack to access. Haystack DocumentStores are available with ElasticSearch, Opensearch, Weaviate, Pinecone, FAISS and more. For a full list of available DocumentStores, check out our [documentation](https://docs.haystack.deepset.ai/docs/document_store).

## What to Build with Haystack

-   Build **retrieval augmented generation (RAG)** by making use of one of the available vector databases and customizing your LLM interaction, the sky is the limit 🚀
-   Perform Question Answering **in natural language** to find granular answers in your documents.
-   Perform **semantic search** and retrieve documents according to meaning.
-   Build applications that can make complex decisions making to answer complex queries: such as systems that can resolve complex customer queries, do knowledge search on many disconnected resources and so on.
-   Use **off-the-shelf models** or **fine-tune** them to your data.
-   Use **user feedback** to evaluate, benchmark, and continuously improve your models.

## Features

-   **Latest models**: Haystack allows you to use and compare models available from OpenAI, Cohere and Hugging Face, as well as your own local models or models hosted on SageMaker. Use the latest LLMs or Transformer-based models (for example: BERT, RoBERTa, MiniLM).
-   **Modular**: Multiple choices to fit your tech stack and use case. A wide choice of DocumentStores to store your data, file conversion tools and more
-   **Open**: Integrated with Hugging Face's model hub, OpenAI, Cohere and various Azure services.
-   **Scalable**: Scale to millions of docs using retrievers and production-scale components like Elasticsearch and a fastAPI REST API.
-   **End-to-End**: All tooling in one place: file conversion, cleaning, splitting, training, eval, inference, labeling, and more.
-   **Customizable**: Fine-tune models to your domain or implement your custom Nodes.
-   **Continuous Learning**: Collect new training data from user feedback in production & improve your models continuously.

## Resources
|                                                                        |                                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📒 [Docs](https://docs.haystack.deepset.ai/v2.0/docs)                   | Components, Pipeline Nodes, Guides, API Reference                                                                                                                                                                                                 |
| 🎓 [Tutorials](https://haystack.deepset.ai/tutorials)                   | See what Haystack can do with our Notebooks & Scripts                                                                                                                                                                                             |
| 🎉 [Integrations](https://haystack.deepset.ai/integrations)     | The index of additional Haystack packages and components that can be installed separately.                                                                                                                                                  |
| 🔰 [Demos](https://github.com/deepset-ai/haystack-demos)                | A repository containing Haystack demo applications with Docker Compose and a REST API                                                                                                                                                             |
| 🖖 [Community](https://github.com/deepset-ai/haystack#-community)       | [Discord](https://discord.gg/haystack), [𝕏 (Twitter)](https://twitter.com/haystack_ai), [Stack Overflow](https://stackoverflow.com/questions/tagged/haystack), [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) |
| 💙 [Contributing](https://github.com/deepset-ai/haystack#-contributing) | We welcome all contributions!                                                                                                                                                                                                                     |
| 🔭 [Roadmap](https://haystack.deepset.ai/overview/roadmap)              | Public roadmap of Haystack                                                                                                                                                                                                                        |
| 📰 [Blog](https://haystack.deepset.ai/blog)                             | Learn about the latest with Haystack and NLP                                                                                                                                                                                                      |
| ☎️ [Jobs](https://www.deepset.ai/jobs)                                  | We're hiring! Have a look at our open positions                                                                                                                                                                                                   |


## 💾 Installation

For a detailed installation guide see [the official documentation](https://docs.haystack.deepset.ai/v2.0/docs/installation). There you’ll find instructions for custom installations handling Windows and Apple Silicon.

**Basic Installation**

Use [pip](https://github.com/pypa/pip) to install a basic version of Haystack's latest release:

```sh
pip install haystack-ai
```

This command installs everything needed for basic Pipelines that use an in-memory DocumentStore and external LLM provider (e.g. OpenAI).

If you want to try out the newest features that are not in an official release yet, you can install the unstable version from the main branch with the following command:

```sh
pip install git+https://github.com/deepset-ai/haystack.git@main#egg=haystack-ai
```

To be able to make changes to Haystack code, first of all clone this repo:

```sh
git clone https://github.com/deepset-ai/haystack.git
```

Then move into the cloned folder and install the project with `pip`, including the development dependencies:

```console
cd haystack && pip install -e '.[dev]'
```

If you want to contribute to the Haystack repo, check our [Contributor Guidelines](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md) first.

## 🔰Demos

You can find some of our hosted demos with instructions to run them locally too on our [haystack-demos](https://github.com/deepset-ai/haystack-demos) repository

:dizzy: **[Reduce Hallucinations with Retrieval Augmentation](https://huggingface.co/spaces/deepset/retrieval-augmentation-svb) - Generative QA with LLMs**

🐥 **[Should I follow?](https://huggingface.co/spaces/deepset/should-i-follow) - Summarizing tweets with LLMs**

🌎 **[Explore The World](https://haystack-demo.deepset.ai/) - Extractive Question Answering**

### 🖖 Community

If you have a feature request or a bug report, feel free to open an [issue in Github](https://github.com/deepset-ai/haystack/issues). We regularly check these and you can expect a quick response. If you'd like to discuss a topic, or get more general advice on how to make Haystack work for your project, you can start a thread in [Github Discussions](https://github.com/deepset-ai/haystack/discussions) or our [Discord channel](https://discord.gg/haystack). We also check [𝕏 (Twitter)](https://twitter.com/haystack_ai) and [Stack Overflow](https://stackoverflow.com/questions/tagged/haystack).

### 💙 Contributing

We are very open to the community's contributions - be it a quick fix of a typo, or a completely new feature! You don't need to be a Haystack expert to provide meaningful improvements. To learn how to get started, check out our [Contributor Guidelines](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md) first.


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
-   [Nvidia](https://developer.nvidia.com/blog/reducing-development-time-for-intelligent-virtual-assistants-in-contact-centers/)
-   [PostHog](https://github.com/PostHog/max-ai#readme)
-   [Rakuten](https://www.rakuten.com/)
-   [Sooth.ai](https://www.deepset.ai/blog/advanced-neural-search-with-sooth-ai)
