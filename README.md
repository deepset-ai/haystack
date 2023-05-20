<div align="center">
  <a href="https://www.deepset.ai/haystack/"><img src="https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/haystack_logo_colored.png" alt="Haystack"></a>

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![Tests](https://github.com/deepset-ai/haystack/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/tests.yml) [![Docker image release](https://github.com/deepset-ai/haystack/actions/workflows/docker_release.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/docker_release.yml) [![Schemas](https://github.com/deepset-ai/haystack/actions/workflows/schemas.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/schemas.yml) [![code style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) |
| Docs    | [![Sync docs with Readme](https://github.com/deepset-ai/haystack/actions/workflows/readme_sync.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/readme_sync.yml) ![Website](https://img.shields.io/website?label=documentation&up_message=online&url=https%3A%2F%2Fdocs.haystack.deepset.ai)                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Package | ![PyPI](https://img.shields.io/pypi/v/farm-haystack) ![PyPI - Downloads](https://img.shields.io/pypi/dm/farm-haystack?color=blue&logo=pypi&logoColor=gold) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/farm-haystack?logo=python&logoColor=gold) ![GitHub](https://img.shields.io/github/license/deepset-ai/haystack?color=blue) [![License Compliance](https://github.com/deepset-ai/haystack/actions/workflows/license_compliance.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/license_compliance.yml)                                                                                                                                                                                            |
| Meta    | ![Discord](https://img.shields.io/discord/993534733298450452?logo=discord) ![Twitter Follow](https://img.shields.io/twitter/follow/deepset_ai)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
</div>

[Haystack](https://haystack.deepset.ai/) is an end-to-end NLP framework that enables you to build NLP applications powered by LLMs, Transformer models, vector search and more. Whether you want to perform question answering, answer generation, semantic document search, or build tools that are capable of complex decision making and query resolution, you can use the state-of-the-art NLP models with Haystack to build end-to-end NLP applications solving your use case.

## Core Concepts

🏃‍♀️ **[Pipelines](https://docs.haystack.deepset.ai/docs/pipelines):** This is the standard Haystack structure that can connect to your data and perform on it NLP tasks that you define. The data in a Pipeline flows from one Node to the next. You define how Nodes interact with each other, and how one Node pushes data to the next.

An example pipeline would consist of one `Retriever` Node and one `Reader` Node. When the pipeline runs with a query, the Retriever first retrieves the documents relevant to the query and then the Reader extracts the final answer.

⚛️ **[Nodes](https://docs.haystack.deepset.ai/docs/nodes_overview):** Each Node achieves one thing. Such as preprocessing documents, retrieving documents, using language models to answer questions and so on.

🕵️ **[Agent](https://docs.haystack.deepset.ai/docs/agent):** (since 1.15) An Agent is a component that is powered by an LLM, such as GPT-3. It can decide on the next best course of action so as to get to the result of a query. It uses the Tools available to it to achieve this. While a pipeline has a clear start and end, an Agent is able to decide whether the query has resolved or not. It may also make use of a Pipeline as a Tool.

🛠️ **[Tools](https://docs.haystack.deepset.ai/docs/agent#tools):** You can think of a Tool as an expert, that is able to do something really well. Such as a calculator, good at mathematics. Or a [WebRetriever](https://docs.haystack.deepset.ai/docs/agent#web-tools), good at retrieving pages from the internet. A Node or pipeline in Haystack can also be used as a Tool. A Tool is a component that is used by an Agent, to resolve complex queries.

🗂️ **[DocumentStores](https://docs.haystack.deepset.ai/docs/document_store):** A DocumentStore is database where you store your text data for Haystack to access. Haystack DocumentStores are available with ElasticSearch, Opensearch, Weaviate, Pinecone, FAISS and more. For a full list of available DocumentStores, check out our [documentation](https://docs.haystack.deepset.ai/docs/document_store).

## What to Build with Haystack

-   Perform Question Answering **in natural language** to find granular answers in your documents.
-   **Generate answers or content** with the use of LLM such as articles, tweets, product descriptions and more, the sky is the limit 🚀
-   Perform **semantic search** and retrieve documents according to meaning.
-   Build applications that can do complex decisions making to answer complex queries: such as systems that can resolve complex customer queries, do knowledge search on many disconnected resources and so on.
-   Use **off-the-shelf models** or **fine-tune** them to your data.
-   Use **user feedback** to evaluate, benchmark, and continuously improve your models.

## Features

-   **Latest models**: Haystack allows you to use and compare models available from OpenAI, Cohere and Hugging Face, as well as your own local models. Use the latest LLMs or Transformer-based models (for example: BERT, RoBERTa, MiniLM).
-   **Modular**: Multiple choices to fit your tech stack and use case. A wide choice of DocumentStores to store your data, file conversion tools and more
-   **Open**: Integrated with Hugging Face's model hub, OpenAI, Cohere and various Azure services.
-   **Scalable**: Scale to millions of docs using retrievers and production-scale components like Elasticsearch and a fastAPI REST API.
-   **End-to-End**: All tooling in one place: file conversion, cleaning, splitting, training, eval, inference, labeling, and more.
-   **Customizable**: Fine-tune models to your domain or implement your custom Nodes.
-   **Continuous Learning**: Collect new training data from user feedback in production & improve your models continuously.

## Resources
|                                                                        |                                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📒 [Docs](https://docs.haystack.deepset.ai)                             | Components, Pipeline Nodes, Guides, API Reference                                                                                                                                                                                                 |
| 💾 [Installation](https://github.com/deepset-ai/haystack#-installation) | How to install Haystack                                                                                                                                                                                                                           |
| 🎓 [Tutorials](https://haystack.deepset.ai/tutorials)                   | See what Haystack can do with our Notebooks & Scripts                                                                                                                                                                                             |
| 🎉 [Haystack Extras](https://github.com/deepset-ai/haystack-extras)     | A repository that lists extra Haystack packages and components that can be installed separately.                                                                                                                                                  |
| 🔰 [Demos](https://github.com/deepset-ai/haystack-demos)                | A repository containing Haystack demo applications with Docker Compose and a REST API                                                                                                                                                             |
| 🖖 [Community](https://github.com/deepset-ai/haystack#-community)       | [Discord](https://haystack.deepset.ai/community/join), [Twitter](https://twitter.com/deepset_ai), [Stack Overflow](https://stackoverflow.com/questions/tagged/haystack), [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) |
| 💙 [Contributing](https://github.com/deepset-ai/haystack#-contributing) | We welcome all contributions!                                                                                                                                                                                                                     |
| 📊 [Benchmarks](https://haystack.deepset.ai/benchmarks/)                | Speed & Accuracy of Retriever, Readers and DocumentStores                                                                                                                                                                                         |
| 🔭 [Roadmap](https://haystack.deepset.ai/overview/roadmap)              | Public roadmap of Haystack                                                                                                                                                                                                                        |
| 📰 [Blog](https://haystack.deepset.ai/blog)                             | Learn about the latest with Haystack and NLP                                                                                                                                                                                                      |
| ☎️ [Jobs](https://www.deepset.ai/jobs)                                  | We're hiring! Have a look at our open positions                                                                                                                                                                                                   |


## 💾 Installation

For a detailed installation guide see [the official documentation](https://docs.haystack.deepset.ai/docs/installation). There you’ll find instructions for custom installations handling Windows and Apple Silicon.

**Basic Installation**

Use [pip](https://github.com/pypa/pip) to install a basic version of Haystack's latest release:

```sh
pip install farm-haystack
```

This command installs everything needed for basic Pipelines that use an in-memory DocumentStore.

**Full Installation**

To use more advanced features, like certain DocumentStores, FileConverters, OCR, or Ray,
you need to install further dependencies. The following command installs the [latest release](https://github.com/deepset-ai/haystack/releases) of Haystack and all its dependencies:

```sh
pip install 'farm-haystack[all]' ## or 'all-gpu' for the GPU-enabled dependencies
```

If you want to try out the newest features that are not in an official release yet, you can install the unstable version from the main branch with the following command:

```sh
pip install git+https://github.com/deepset-ai/haystack.git@main#egg=farm-haystack
```

To be able to make changes to Haystack code, first of all clone this repo:

```sh
git clone https://github.com/deepset-ai/haystack.git
```

Then move into the cloned folder and install the project with `pip`, including the development dependencies:

```console
cd haystack && pip install -e '.[dev]'
```

If you want to contribute to the Haystack repo, check our [Contributor Guidelines](#💙-contributing) first.

See the list of [dependencies](https://github.com/deepset-ai/haystack/blob/main/pyproject.toml) to check which ones you want to install (for example, `[all]`, `[dev]`, or other).

**Installing the REST API**

Haystack comes packaged with a REST API so that you can deploy it as a service. Run the following command from the root directory of the Haystack repo to install REST_API:

```
pip install rest_api/
```

You can find out more about our PyPi package on our [PyPi page](https://pypi.org/project/farm-haystack/).

## 🔰Demos

You can find some of our hosted demos with instructions to run them locally too on our [haystack-demos](https://github.com/deepset-ai/haystack-demos) repository

:dizzy: **[Reduce Hallucinations with Retrieval Augmentation](https://huggingface.co/spaces/deepset/retrieval-augmentation-svb) - Generative QA with LLMs**

🐥 **[Should I follow?](https://huggingface.co/spaces/deepset/should-i-follow) - Summarizing tweets with LLMs**

🌎 **[Explore The World](https://haystack-demo.deepset.ai/) - Extractive Question Answering**

### 🖖 Community

If you have a feature request or a bug report, feel free to open an [issue in Github](https://github.com/deepset-ai/haystack/issues). We regularly check these and you can expect a quick response. If you'd like to discuss a topic, or get more general advice on how to make Haystack work for your project, you can start a thread in [Github Discussions](https://github.com/deepset-ai/haystack/discussions) or our [Discord channel](https://haystack.deepset.ai/community). We also check [Twitter](https://twitter.com/deepset_ai) and [Stack Overflow](https://stackoverflow.com/questions/tagged/haystack).

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
