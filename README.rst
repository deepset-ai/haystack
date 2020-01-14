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

The performance of **modern Question Answering Models** (BERT, ALBERT ...) has seen drastic improvements within the last year enabling many new opportunities for finding information more efficiently. However, those models are usually designed to find answers within rather small text passages. **Haystack let's you scale QA models** to large collections of documents!

Haystack is designed in a modular way and let's you use any QA models trained with  `FARM <https://github.com/deepset-ai/FARM>`_ or `Transformers <https://github.com/huggingface/transformers>`_.

Swap your models easily from BERT to roBERTa and scale the database from dev (Sqlite) to production (PostgreSQL, elasticsearch ...).

Core Features
=============
- **Powerful models**: Utilize all the latest transformer based models (BERT, ALBERT roBERTa ...)
- **Modular & future-proof**: Avoid technical debt. With haystack you can easily switch to newer models once they get published.
- **Developer friendly**: Easy to debug, extend and modify.
- **Scalable**: Switch from dev to production within minutes.

Components
==========

1. **Retriever**:  Fast, simple model that identifies candidate passages from a large collection of documents. Algorithms include TF-IDF, which is similar to what's used in popular search systems like Elasticsearch. The Retriever helps to narrow down the scope for Reader to smaller units of text where a given question could be answered.

2. **Reader**: Powerful neural model that reads through texts in detail to find an answer. Use diverse models like BERT, Roberta or XLNet trained via `FARM <https://github.com/deepset-ai/FARM>`_ or `Transformers <https://github.com/huggingface/transformers>`_ on SQuAD like tasks. The Reader takes multiple passages of text as input and returns top-n answers with corresponding confidence scores.

3. **Finder**: Glues together a Reader and a Retriever as a pipeline to provide an easy-to-use question answering interface.

4. **Labeling Tool**: (Coming soon)

Resources
=========
- Tutorial 1  - Basic QA Pipeline: `Jupyter notebook  <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial1_Basic_QA_Pipeline.ipynb>`_  or `Colab <https://colab.research.google.com/drive/1Gj3JjPPcm8DMmctz66K68cOV53JZKqeX>`_

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


Configuration
-------------
The configuration can be supplied in a :code:`qa_config.py` placed in the PYTHONPATH. Alternatively, the :code:`DATABASE_URL` can also be set as an environment variable.


Deployment
==========

SQL Backend
-----------
The database ORM layer is implemented using SQLAlchemy library. By default, it uses the file-based SQLite database. For large scale deployments, the configuration can be changed to use other compatible databases like PostgreSQL or MySQL.

Elasticsearch Backend
----------------------
(Coming soon)

REST API
--------
A simple REST API based on `FastAPI <https://fastapi.tiangolo.com/>`_ is included to answer questions at inference time. To serve the API, run :code:`uvicorn haystack.api.inference:app`.
You will find the Swagger API documentation at http://127.0.0.1:8000/docs