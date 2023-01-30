---
type: intro
date: 1.1.2023
---
```bash
pip install farm-haystack
```
## What to build with Haystack

- **Ask questions in natural language** and find granular answers in your own documents.
- Perform **semantic search** and retrieve documents according to meaning not keywords
- Use **off-the-shelf models** or **fine-tune** them to your own domain.
- Use **user feedback** to evaluate, benchmark and continuously improve your live models.
- Leverage existing **knowledge bases** and better handle the long tail of queries that **chatbots** receive.
- **Automate processes** by automatically applying a list of questions to new documents and using the extracted answers.

![Logo](https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/logo.png)


## Core Features

-   **Latest models**: Utilize all latest transformer based models (e.g. BERT, RoBERTa, MiniLM) for extractive QA, generative QA and document retrieval.
-   **Modular**: Multiple choices to fit your tech stack and use case. Pick your favorite database, file converter or modeling framwework.
-   **Open**: 100% compatible with HuggingFace's model hub. Tight interfaces to other frameworks (e.g. Transformers, FARM, sentence-transformers)
-   **Scalable**: Scale to millions of docs via retrievers, production-ready backends like Elasticsearch / FAISS and a fastAPI REST API
-   **End-to-End**: All tooling in one place: file conversion, cleaning, splitting, training, eval, inference, labeling ...
-   **Developer friendly**: Easy to debug, extend and modify.
-   **Customizable**: Fine-tune models to your own domain or implement your custom DocumentStore.
-   **Continuous Learning**: Collect new training data via user feedback in production & improve your models continuously

|  |  |
|-|-|
| :ledger: [Docs](https://haystack.deepset.ai/overview/intro) | Usage, Guides, API documentation ...|
| :beginner: [Quick Demo](https://github.com/deepset-ai/haystack/#quick-demo) | Quickly see what Haystack offers |
| :floppy_disk: [Installation](https://github.com/deepset-ai/haystack/#installation) | How to install Haystack |
| :art: [Key Components](https://github.com/deepset-ai/haystack/#key-components) | Overview of core concepts |
| :mortar_board: [Tutorials](https://github.com/deepset-ai/haystack/#tutorials) | Jupyter/Colab Notebooks & Scripts |
| :eyes: [How to use Haystack](https://github.com/deepset-ai/haystack/#how-to-use-haystack) | Basic explanation of concepts, options and usage |
| :heart: [Contributing](https://github.com/deepset-ai/haystack/#heart-contributing) | We welcome all contributions! |
| :bar_chart: [Benchmarks](https://haystack.deepset.ai/benchmarks/v0.9.0) | Speed & Accuracy of Retriever, Readers and DocumentStores |
| :telescope: [Roadmap](https://haystack.deepset.ai/overview/roadmap) | Public roadmap of Haystack |
| :pray: [Slack](https://haystack.deepset.ai/community/join) | Join our community on Slack |
| :bird: [Twitter](https://twitter.com/deepset_ai) | Follow us on Twitter for news and updates |
| :newspaper: [Blog](https://medium.com/deepset-ai) | Read our articles on Medium |


## Quick Demo

The quickest way to see what Haystack offers is to start a [Docker Compose](https://docs.docker.com/compose/) demo application:

**1. Update/install Docker and Docker Compose, then launch Docker**

```
    # apt-get update && apt-get install docker && apt-get install docker-compose
    # service docker start
```

**2. Clone Haystack repository**

```
    # git clone https://github.com/deepset-ai/haystack.git
```

### 2nd level headline for testing purposes
#### 3rd level headline for testing purposes
