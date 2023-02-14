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

[Haystack](https://haystack.deepset.ai) is an end-to-end framework that enables you to build powerful and production-ready pipelines for different search use cases.
Whether you want to perform question answering (QA) or semantic document search, you can use the state-of-the-art NLP models in Haystack to provide unique search experiences and allow your users to query in natural language.
Haystack is built in a modular fashion so that you can combine the best technology from other open source projects, like Hugging Face's transformers, Elasticsearch, or Milvus.

<p align="center"><img src="https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/main_example.gif"></p>

## What to Build with Haystack

- **Ask questions in natural language** and find granular answers in your documents.
- Perform **semantic search** and retrieve documents according to meaning, not keywords.
- Use **off-the-shelf models** or **fine-tune** them to your domain.
- Use **user feedback** to evaluate, benchmark, and continuously improve your live models.
- Leverage existing **knowledge bases** and better handle the long tail of queries that **chatbots** receive.
- **Automate processes** by automatically applying a list of questions to new documents and using the extracted answers.

## Core Features

- **Latest models**: Utilize all latest transformer-based models (for example, BERT, RoBERTa, MiniLM) for extractive QA, generative QA, and document retrieval.
- **Modular**: Multiple choices to fit your tech stack and use case. Pick your favorite database, file converter, or modeling framework.
- **Pipelines**: Use the Node and Pipeline design of Haystack to route queries to only the relevant components.
- **Open**: 100% compatible with Hugging Face's model hub. Tight interfaces to other frameworks (for example, transformers, FARM, sentence-transformers).
- **Scalable**: Scale to millions of docs using retrievers, production-ready backends like Elasticsearch / FAISS, and a fastAPI REST API.
- **End-to-End**: All tooling in one place: file conversion, cleaning, splitting, training, eval, inference, labeling, and more.
- **Developer friendly**: Easy to debug, extend, and modify.
- **Customizable**: Fine-tune models to your domain or implement your custom DocumentStore.
- **Continuous Learning**: Collect new training data from user feedback in production & improve your models continuously.

|                                                                                               |                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| :ledger: [Docs](https://docs.haystack.deepset.ai)                                             | Components, Pipeline Nodes, Guides, API Reference                                                                                                                                                                                                 |
| :floppy_disk: [Installation](https://github.com/deepset-ai/haystack#floppy_disk-installation) | How to install Haystack                                                                                                                                                                                                                           |
| :mortar_board: [Tutorials](https://github.com/deepset-ai/haystack#mortar_board-tutorials)     | See what Haystack can do with our Notebooks & Scripts                                                                                                                                                                                             |
| :beginner: [Quick Demo](https://github.com/deepset-ai/haystack#beginner-quick-demo)           | Deploy a Haystack application with Docker Compose and a REST API                                                                                                                                                                                  |
| :vulcan_salute: [Community](https://github.com/deepset-ai/haystack#vulcan_salute-community)   | [Discord](https://haystack.deepset.ai/community/join), [Twitter](https://twitter.com/deepset_ai), [Stack Overflow](https://stackoverflow.com/questions/tagged/haystack), [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) |
| :heart: [Contributing](https://github.com/deepset-ai/haystack#heart-contributing)             | We welcome all contributions!                                                                                                                                                                                                                     |
| :bar_chart: [Benchmarks](https://haystack.deepset.ai/benchmarks/)                             | Speed & Accuracy of Retriever, Readers and DocumentStores                                                                                                                                                                                         |
| :telescope: [Roadmap](https://haystack.deepset.ai/overview/roadmap)                           | Public roadmap of Haystack                                                                                                                                                                                                                        |
| :newspaper: [Blog](https://medium.com/deepset-ai)                                             | Read our articles on Medium                                                                                                                                                                                                                       |
| :phone: [Jobs](https://www.deepset.ai/jobs)                                                   | We're hiring! Have a look at our open positions                                                                                                                                                                                                   |


## :floppy_disk: Installation

**Basic Installation**

Use [pip](https://github.com/pypa/pip) to install a basic version of Haystack's latest release:

```
    pip install farm-haystack
```

This command installs everything needed for basic Pipelines that use an Elasticsearch DocumentStore.

**Full Installation**

To use more advanced features, like certain DocumentStores, FileConverters, OCR, or Ray, install further dependencies. The following command installs the latest version of Haystack and all its dependencies from the main branch:

```
git clone https://github.com/deepset-ai/haystack.git
cd haystack
pip install --upgrade pip
pip install -e '.[all]' ## or 'all-gpu' for the GPU-enabled dependencies
```

**Custom Installation**
You can choose the dependencies you want to install. To do so, specify them in the `pip install` command:

```
pip install 'farm-haystack[DEPENDENCY_OPTION]'
```
You can find a full list of dependency options at [haystack/pyproject.toml](https://github.com/deepset-ai/haystack/blob/main/pyproject.toml#L96).

If you're running pip version earlier than 21.3, you can't install dependency groups that reference other groups. Instead, you can only specify groups that contain direct package references:

```
# instead of '[all]'
pip install 'farm-haystack[sql,only-faiss,only-milvus1,weaviate,pinecone,opensearch,graphdb,inmemorygraph,crawler,preprocessing,ocr,onnx,ray,dev]'

# instead of '[all-gpu]'
pip install 'farm-haystack[sql,only-faiss-gpu,only-milvus1,weaviatepinecone,opensearch,graphdb,inmemorygraph,crawler,preprocessing,ocr,onnx-gpu,ray,dev]'
```

**Installing the REST API**
Haystack comes packaged with a REST API so that you can deploy it as a service. Run the following command from the root directory of the Haystack repo to install REST_API:

```
pip install rest_api/
```

**Other Operating Systems**

**Windows**
We recommend installing [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to use Haystack on Windows:

```
pip install farm-haystack -f https://download.pytorch.org/whl/torch_stable.html
```

**Apple Silicon (M1)**

Macs with an M1 processor require some extra dependencies to install Haystack:

```
# some additional dependencies needed on m1 mac
brew install postgresql
brew install cmake
brew install rust

# haystack installation
GRPC_PYTHON_BUILD_SYSTEM_ZLIB=true pip install git+https://github.com/deepset-ai/haystack.git
```

**Learn More**

See our [installation guide](https://docs.haystack.deepset.ai/docs/installation) for more options.
You can find out more about our PyPi package on our [PyPi page](https://pypi.org/project/farm-haystack/).

## :mortar_board: Tutorials

![image](https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/concepts_haystack_handdrawn.png)

Follow our [introductory tutorial](https://haystack.deepset.ai/tutorials/first-qa-system)
to set up a question answering system using Python and start performing queries!
Explore [the rest of our tutorials](https://haystack.deepset.ai/tutorials)
to learn how to tweak pipelines, train models, and perform evaluation.

## :beginner: Quick Demo

**Hosted**

Try out our hosted [Explore The World](https://haystack-demo.deepset.ai/) live demo here!
Ask any question on countries or capital cities and let Haystack return the answers to you.

**Local**

To run the Explore The World demo on your own machine and customize it to your needs, check out the instructions on [Explore the World repository](https://github.com/deepset-ai/haystack-demos/tree/main/explore_the_world) on GitHub.

## :vulcan_salute: Community

There is a very vibrant and active community around Haystack which we are regularly interacting with!
If you have a feature request or a bug report, feel free to open an [issue in Github](https://github.com/deepset-ai/haystack/issues).
We regularly check these and you can expect a quick response.
If you'd like to discuss a topic, or get more general advice on how to make Haystack work for your project,
you can start a thread in [Github Discussions](https://github.com/deepset-ai/haystack/discussions) or our [Discord channel](https://haystack.deepset.ai/community).
We also check [Twitter](https://twitter.com/deepset_ai) and [Stack Overflow](https://stackoverflow.com/questions/tagged/haystack).


## :heart: Contributing

We are very open to the community's contributions - be it a quick fix of a typo, or a completely new feature!
You don't need to be a Haystack expert to provide meaningful improvements.
To learn how to get started, check out our [Contributor Guidelines](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md) first.

You can also find instructions to run the tests locally there.

Thanks so much to all those who have contributed to our project!

<a href="https://github.com/deepset-ai/haystack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=deepset-ai/haystack" />
</a>


## Who Uses Haystack

Here's a list of organizations that use Haystack. Don't hesitate to send a PR to let the world know that you use Haystack. Join our growing community!

- [Airbus](https://www.airbus.com/en)
- [Alcatel-Lucent](https://www.al-enterprise.com/)
- [BetterUp](https://www.betterup.com/)
- [Deepset](https://deepset.ai/)
- [Etalab](https://www.etalab.gouv.fr/)
- [Infineon](https://www.infineon.com/)
- [Sooth.ai](https://sooth.ai/)
