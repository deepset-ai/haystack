*******************************************************
Haystack — Neural Question Answering At Scale
*******************************************************
.. image:: https://travis-ci.org/deepset-ai/haystack.svg?branch=master
	:target: https://travis-ci.org/deepset-ai/haystack
	:alt: Build

.. image:: https://img.shields.io/github/release/deepset-ai/haystack
	:target: https://github.com/deepset-ai/haystack/releases
	:alt: Release

.. image:: https://img.shields.io/github/license/deepset-ai/haystack
	:target: https://github.com/deepset-ai/haystack/blob/master/LICENSE
	:alt: License

.. image:: https://img.shields.io/github/last-commit/deepset-ai/haystack
	:target: https://github.com/deepset-ai/haystack/commits/master
	:alt: Last Commit

Introduction
============

The performance of **modern Question Answering Models** (BERT, ALBERT ...) has seen drastic improvements within the last year enabling many new opportunities for accessing information more efficiently. However, those models are designed to find answers within rather small text passages. **Haystack lets you scale QA models** to large collections of documents!
While QA is the focussed use case for haystack, we will soon support additional options to boost search (re-ranking, most-similar search, tagging ...). 

Haystack is designed in a modular way and lets you use any models trained with  `FARM <https://github.com/deepset-ai/FARM>`_ or `Transformers <https://github.com/huggingface/transformers>`_.

Switching between different backends allows fast prototyping (SQLite) and scalable deployment for production (elasticsearch).


Core Features
=============
- **Powerful models**: Utilize all latest transformer based models (BERT, ALBERT roBERTa ...)
- **Modular & future-proof**: Easily switch to newer models once they get published.
- **Developer friendly**: Easy to debug, extend and modify.
- **Scalable**: Production-ready deployments via Elasticsearch backend.
- **Customizable**: Fine-tune models to your own domain.

Components
==========

1. **Retriever**:  Fast, simple algorithm that identifies candidate passages from a large collection of documents. Algorithms include TF-IDF or BM25, which is similar to what's used in Elasticsearch. The Retriever helps to narrow down the scope for Reader to smaller units of text where a given question could be answered.

2. **Reader**: Powerful neural model that reads through texts in detail to find an answer. Use diverse models like BERT, Roberta or XLNet trained via `FARM <https://github.com/deepset-ai/FARM>`_ or `Transformers <https://github.com/huggingface/transformers>`_ on SQuAD like tasks. The Reader takes multiple passages of text as input and returns top-n answers with corresponding confidence scores. You can just load a pretrained model from  `huggingface's model hub <https://huggingface.co/models>`_ or fine-tune it to your own domain data. 

3. **Finder**: Glues together a Reader and a Retriever as a pipeline to provide an easy-to-use question answering interface.

4. **Labeling Tool**: `Hosted version <https://annotate.deepset.ai/login>`_  (Beta), Docker images (coming soon)


Resources
=========
- Tutorial 1  - Basic QA Pipeline: `Jupyter notebook  <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial1_Basic_QA_Pipeline.ipynb>`_  or `Colab <https://colab.research.google.com/drive/1Gj3JjPPcm8DMmctz66K68cOV53JZKqeX>`_
- Tutorial 2  - Fine-tuning a model on own data: `Jupyter notebook <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial2_Finetune_a_model_on_your_data.ipynb>`_ or `Colab <https://colab.research.google.com/drive/1nnkxLB7Sq7TJmIvTvxiYuWmNleWNLIdR>`_
- Tutorial 3  - Using Elasticsearch as Backend: `Jupyter notebook <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial3_Elasticsearch_backend.ipynb>`_ or `Colab <https://colab.research.google.com/drive/1NbtUskkt4NaUgj4KM9QTHTSR3GCQVyn7>`_

Quickstart
==========

Installation
------------
There are two ways to install:

* (recommended) from source, :code:`git clone <url>` and run :code:`pip install [--editable] .` from the root of the repositry.
* from PyPI, do a :code:`pip install farm-haystack`


Usage
-----
.. image:: https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/img/code_snippet_usage.png



Deployment
==========

Haystack has an extensible document store layer.
There are currently implementations of Elasticsearch and SQL (see :code:`haystack.database.elasticsearch.ElasticsearchDocumentStore`  and :code:`haystack.database.sql.SQLDocumentStore`).

Elasticsearch Backend
---------------------
Elasticsearch is recommended for deploying on a large scale. The documents can optionally be chunked into smaller units (e.g., paragraphs) before indexing to make the results returned by the Retriever more granular and accurate.
Retrievers can access an Elasticsearch index to find the relevant paragraphs(or documents) for a query. The default `ElasticsearchRetriever` uses Elasticsearch's native scoring (BM25), but can be extended easily with custom implementations.

You can get started by running a single Elasticsearch node using docker::

     docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:7.5.1

SQL Backend
-----------
The SQL backend layer is mainly meant to simplify the first development steps. By default, a local file-based SQLite database is initialized.
However, if you prefer a PostgreSQL or MySQL backend for production, you can easily configure this since our implementation is based on SQLAlchemy.

REST API
--------
A simple REST API based on `FastAPI <https://fastapi.tiangolo.com/>`_ is included to answer questions at inference time. To serve the API, run :code:`uvicorn haystack.api.inference:app`.
You will find the Swagger API documentation at http://127.0.0.1:8000/docs

Labeling Tool
=============
* Use the `hosted version <https://annotate.deepset.ai/login>`_  (Beta) or deploy it yourself via Docker images (coming soon)  
* Create labels with different techniques: Come up with questions (+ answers) while reading passages (SQuAD style) or have a set of predefined questions and look for answers in the document (~ Natural Questions).
* Structure your work via organizations, projects, users 
* Upload your documents or import labels from an existing SQuAD-style dataset
* Coming soon: more file formats for document upload, metrics for label quality ...
.. image:: https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/img/annotation_tool.png
