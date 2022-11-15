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
        <img alt="Twitter" src="https://img.shields.io/twitter/follow/deepset_ai?style=social">
    </a>
</p>

[Haystack](https://haystack.deepset.ai) is an end-to-end framework that enables you to build powerful and production-ready pipelines for different search use cases.
Whether you want to perform Question Answering or semantic document search, you can use the State-of-the-Art NLP models in Haystack to provide unique search experiences and allow your users to query in natural language.
Haystack is built in a modular fashion so that you can combine the best technology from other open-source projects like Huggingface's Transformers, Elasticsearch, or Milvus.

<p align="center"><img src="https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/main_example.gif"></p>

## What to build with Haystack

- **Ask questions in natural language** and find granular answers in your documents.
- Perform **semantic search** and retrieve documents according to meaning, not keywords
- Use **off-the-shelf models** or **fine-tune** them to your domain.
- Use **user feedback** to evaluate, benchmark, and continuously improve your live models.
- Leverage existing **knowledge bases** and better handle the long tail of queries that **chatbots** receive.
- **Automate processes** by automatically applying a list of questions to new documents and using the extracted answers.

## Core Features

- **Latest models**: Utilize all latest transformer-based models (e.g., BERT, RoBERTa, MiniLM) for extractive QA, generative QA, and document retrieval.
- **Modular**: Multiple choices to fit your tech stack and use case. Pick your favorite database, file converter, or modeling framework.
- **Pipelines**: The Node and Pipeline design of Haystack allows for custom routing of queries to only the relevant components.
- **Open**: 100% compatible with HuggingFace's model hub. Tight interfaces to other frameworks (e.g., Transformers, FARM, sentence-transformers)
- **Scalable**: Scale to millions of docs via retrievers, production-ready backends like Elasticsearch / FAISS, and a fastAPI REST API
- **End-to-End**: All tooling in one place: file conversion, cleaning, splitting, training, eval, inference, labeling, etc.
- **Developer friendly**: Easy to debug, extend and modify.
- **Customizable**: Fine-tune models to your domain or implement your custom DocumentStore.
- **Continuous Learning**: Collect new training data via user feedback in production & improve your models continuously

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

**1. Basic Installation**

You can install a basic version of Haystack's latest release by using [pip](https://github.com/pypa/pip).

```
    pip3 install farm-haystack
```

This command will install everything needed for basic Pipelines that use an Elasticsearch Document Store.

**2. Full Installation**

If you plan to be using more advanced features like Milvus, FAISS, Weaviate, OCR or Ray,
you will need to install a full version of Haystack.
The following command will install the latest version of Haystack from the main branch.

```
git clone https://github.com/deepset-ai/haystack.git
cd haystack
pip install --upgrade pip
pip install -e '.[all]' ## or 'all-gpu' for the GPU-enabled dependencies
```

If you cannot upgrade `pip` to version 21.3 or higher, you will need to replace:
- `'.[all]'` with `'.[sql,only-faiss,only-milvus,weaviate,graphdb,crawler,preprocessing,ocr,onnx,ray,dev]'`
- `'.[all-gpu]'` with `'.[sql,only-faiss-gpu,only-milvus,weaviate,graphdb,crawler,preprocessing,ocr,onnx-gpu,ray,dev]'`

For an complete list of the dependency groups available, have a look at the `haystack/pyproject.toml` file.

To install the REST API and UI, run the following from the root directory of the Haystack repo

```
pip install rest_api/
pip install ui/
```

**3. Installing on Windows**

```
pip install farm-haystack -f https://download.pytorch.org/whl/torch_stable.html
```

**4. Installing on Apple Silicon (M1)**

M1 Macbooks require some extra dependencies in order to install Haystack.

```
# some additional dependencies needed on m1 mac
brew install postgresql
brew install cmake
brew install rust

# haystack installation
GRPC_PYTHON_BUILD_SYSTEM_ZLIB=true pip install git+https://github.com/deepset-ai/haystack.git
```

**5. Learn More**

See our [installation guide](https://haystack.deepset.ai/overview/get-started) for more options.
You can find out more about our PyPi package on our [PyPi page](https://pypi.org/project/farm-haystack/).

## :mortar_board: Tutorials

![image](https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/concepts_haystack_handdrawn.png)

Follow our [introductory tutorial](https://haystack.deepset.ai/tutorials/first-qa-system)
to setup a question answering system using Python and start performing queries!
Explore [the rest of our tutorials](https://haystack.deepset.ai/tutorials)
to learn how to tweak pipelines, train models and perform evaluation.

## :beginner: Quick Demo

**Hosted**

Try out our hosted [Explore The World](https://haystack-demo.deepset.ai/) live demo here!
Ask any question on countries or capital cities and let Haystack return the answers to you.

**Local**

Start up a Haystack service via [Docker Compose](https://docs.docker.com/compose/).
With this you can begin calling it directly via the REST API or even interact with it using the included Streamlit UI.

<details>
  <summary>Click here for a step-by-step guide</summary>

**1. Update/install Docker and Docker Compose, then launch Docker**

```
    apt-get update && apt-get install docker && apt-get install docker-compose
    service docker start
```

**2. Clone Haystack repository**

```
    git clone https://github.com/deepset-ai/haystack.git
```

**3. Pull images & launch demo app**

```
    cd haystack
    docker-compose pull
    docker-compose up

    # Or on a GPU machine: docker-compose -f docker-compose-gpu.yml up
```

You should be able to see the following in your terminal window as part of the log output:

```
..
ui_1             |   You can now view your Streamlit app in your browser.
..
ui_1             |   External URL: http://192.168.108.218:8501
..
haystack-api_1   | [2021-01-01 10:21:58 +0000] [17] [INFO] Application startup complete.
```

**4. Open the Streamlit UI for Haystack by pointing your browser to the "External URL" from above.**

You should see the following:

![image](https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/streamlit_ui_screenshot.png)

You can then try different queries against a pre-defined set of indexed articles related to Game of Thrones.

**Note**: The following containers are started as a part of this demo:

* Haystack API: listens on port 8000
* DocumentStore (Elasticsearch): listens on port 9200
* Streamlit UI: listens on port 8501

Please note that the demo will [publish](https://docs.docker.com/config/containers/container-networking/) the container ports to the outside world. *We suggest that you review the firewall settings depending on your system setup and the security guidelines.*

</details>

## :vulcan_salute: Community

There is a very vibrant and active community around Haystack which we are regularly interacting with!
If you have a feature request or a bug report, feel free to open an [issue in Github](https://github.com/deepset-ai/haystack/issues).
We regularly check these and you can expect a quick response.
If you'd like to discuss a topic, or get more general advice on how to make Haystack work for your project,
you can start a thread in [Github Discussions](https://github.com/deepset-ai/haystack/discussions) or our [Discord channel](https://haystack.deepset.ai/community/join).
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


## Who uses Haystack

Here's a list of organizations who use Haystack. Don't hesitate to send a PR to let the world know that you use Haystack. Join our growing community!

- [Airbus](https://www.airbus.com/en)
- [Alcatel-Lucent](https://www.al-enterprise.com/)
- [BetterUp](https://www.betterup.com/)
- [Deepset](https://deepset.ai/)
- [Etalab](https://www.etalab.gouv.fr/)
- [Infineon](https://www.infineon.com/)
- [Sooth.ai](https://sooth.ai/)
