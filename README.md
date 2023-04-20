<p align="center">
  <a href="https://www.deepset.ai/haystack/"><img src="https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/haystack_logo_colored.png" alt="Haystack"></a>
</p>

<p>
    <a href="https://github.com/deepset-ai/haystack/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/deepset-ai/haystack/workflows/Tests/badge.svg?branch=main">
    </a>
    <a href="https://github.com/deepset-ai/haystack-json-schema/actions/workflows/schemas.yml">
        <img alt="Schemas" src="https://github.com/deepset-ai/haystack-json-schema/actions/workflows/schemas.yml/badge.svg">
    </a>
    <a href="https://docs.haystack.deepset.ai">
        <img alt="Documentation" src="https://img.shields.io/website?label=documentation&up_message=online&url=https%3A%2F%2Fdocs.haystack.deepset.ai">
    </a>
    <a href="https://app.fossa.com/projects/custom%2B24445%2Fgithub.com%2Fdeepset-ai%2Fhaystack?ref=badge_shield">
        <img alt="FOSSA Status" src="https://app.fossa.com/api/projects/custom%2B24445%2Fgithub.com%2Fdeepset-ai%2Fhaystack.svg?type=shield"/>
    </a>
    <a href="https://github.com/deepset-ai/haystack/releases">
        <img alt="Release" src="https://img.shields.io/github/release/deepset-ai/haystack">
    </a>
    <a href="https://github.com/deepset-ai/haystack/commits/main">
        <img alt="Last commit" src="https://img.shields.io/github/last-commit/deepset-ai/haystack">
    </a>
    <a href="https://pepy.tech/project/farm-haystack">
        <img alt="Downloads" src="https://pepy.tech/badge/farm-haystack/month">
    </a>
    <a href="https://www.deepset.ai/jobs">
        <img alt="Jobs" src="https://img.shields.io/badge/Jobs-We're%20hiring-blue">
    </a>
        <a href="https://twitter.com/intent/follow?screen_name=deepset_ai">
        <img alt="Twitter" src="https://img.shields.io/badge/follow-%40deepset_ai-1DA1F2?logo=twitter">
    </a>
    <a href="https://discord.com/invite/qZxjM4bAHU">
        <img alt="chat on Discord" src="https://img.shields.io/discord/993534733298450452?logo=discord">
    </a>
</p>

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
|                                                                                               |                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📒 [Docs](https://docs.haystack.deepset.ai)                                             | Components, Pipeline Nodes, Guides, API Reference                                                                                                                                                                                                 |
| 💾 [Installation](https://github.com/deepset-ai/haystack#-installation) | How to install Haystack                                                                                                                                                                                                                           |
| 🎓 [Tutorials](https://haystack.deepset.ai/tutorials)     | See what Haystack can do with our Notebooks & Scripts                                                                                                                                                                                             |
| 🎉 [Haystack Extras](https://github.com/deepset-ai/haystack-extras)               | A repository that lists extra Haystack packages and components that can be installed separately.                                                                                                                                                                                             |
| 🔰 [Demos](https://github.com/deepset-ai/haystack-demos)           | A repository containing Haystack demo applications with Docker Compose and a REST API                                                                                                                                                                                  |
| 🖖 [Community](https://github.com/deepset-ai/haystack#-community)   | [Discord](https://haystack.deepset.ai/community/join), [Twitter](https://twitter.com/deepset_ai), [Stack Overflow](https://stackoverflow.com/questions/tagged/haystack), [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) |
| 💙 [Contributing](https://github.com/deepset-ai/haystack#-contributing)             | We welcome all contributions!                                                                                                                                                                                                                     |
| 📊 [Benchmarks](https://haystack.deepset.ai/benchmarks/)                             | Speed & Accuracy of Retriever, Readers and DocumentStores                                                                                                                                                                                         |
| 🔭 [Roadmap](https://haystack.deepset.ai/overview/roadmap)                           | Public roadmap of Haystack                                                                                                                                                                                                                        |
| 📰 [Blog](https://haystack.deepset.ai/blog)                                             | Learn about the latest with Haystack and NLP                                                                                                                                                                   |
| ☎️ [Jobs](https://www.deepset.ai/jobs)                                                   | We're hiring! Have a look at our open positions                                                                                                                                                                                                   |


## 💾 Installation

For a detailed installation guide see [the official documentation](https://docs.haystack.deepset.ai/docs/installation). There you’ll find instructions for custom installations handling Windows and Apple Silicon.

**Basic Installation**

Use [pip](https://github.com/pypa/pip) to install a basic version of Haystack's latest release:

```
    pip install farm-haystack
```

This command installs everything needed for basic Pipelines that use an Elasticsearch DocumentStore.

**Full Installation**

To use more advanced features, like certain DocumentStores, FileConverters, OCR, or Ray, install further dependencies. The following command installs the latest version of Haystack and all its dependencies from the main branch:

```
pip install --upgrade pip
pip install 'farm-haystack[all]' ## or 'all-gpu' for the GPU-enabled dependencies
```

**Installing the REST API** Haystack comes packaged with a REST API so that you can deploy it as a service. Run the following command from the root directory of the Haystack repo to install REST_API:

```
pip install rest_api/
```

You can find out more about our PyPi package on our [PyPi page](https://pypi.org/project/farm-haystack/).

## 🔰Demos

You can find some of our hosted demos with instructions to run them locally too on our [haystack-demos](https://github.com/deepset-ai/haystack-demos) repository

🐥 **[Should I follow?](https://huggingface.co/spaces/deepset/should-i-follow) - Twitter demo**

🌎 **[Explore The World](https://haystack-demo.deepset.ai/) demo**

### 🖖 Community

If you have a feature request or a bug report, feel free to open an [issue in Github](https://github.com/deepset-ai/haystack/issues). We regularly check these and you can expect a quick response. If you'd like to discuss a topic, or get more general advice on how to make Haystack work for your project, you can start a thread in [Github Discussions](https://github.com/deepset-ai/haystack/discussions) or our [Discord channel](https://haystack.deepset.ai/community). We also check [Twitter](https://twitter.com/deepset_ai) and [Stack Overflow](https://stackoverflow.com/questions/tagged/haystack).

### 💙 Contributing

We are very open to the community's contributions - be it a quick fix of a typo, or a completely new feature! You don't need to be a Haystack expert to provide meaningful improvements. To learn how to get started, check out our [Contributor Guidelines](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md) first.

You can also find instructions to run the tests locally there.

Thanks so much to all those who have contributed to our project!

<a href="[](https://github.com/deepset-ai/haystack/graphs/contributors)[https://github.com/deepset-ai/haystack/graphs/contributors](https://github.com/deepset-ai/haystack/graphs/contributors)"> <img src="[](https://contrib.rocks/image?repo=deepset-ai/haystack)[https://contrib.rocks/image?repo=deepset-ai/haystack](https://contrib.rocks/image?repo=deepset-ai/haystack)" /> </a>

## Who Uses Haystack

Here's a list of organizations that we know about from our community. Don't hesitate to send a PR to let the world know that you use Haystack. Join our growing community!

-   [Airbus](https://www.airbus.com/en)
-   [Alcatel-Lucent](https://www.al-enterprise.com/)
-   [Apple](https://www.apple.com/)
-   [BetterUp](https://www.betterup.com/)
-   [Databricks](https://www.databricks.com/)
-   [Deepset](https://deepset.ai/)
-   [Etalab](https://www.etalab.gouv.fr/)
-   [Infineon](https://www.infineon.com/)
-   [Intelijus](https://www.intelijus.ai/)
-   [LEGO](https://www.lego.com/)
-   [Netflix](https://netflix.com)
-   [Nvidia](https://www.nvidia.com/en-us/)
-   [PostHog](https://posthog.com/)
-   [Rakuten](https://www.rakuten.com/)
-   [Sooth.ai](https://sooth.ai/)
