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
While QA is the focussed use case for haystack, we will soon support additional options to boost search (re-ranking, most-similar search ...).

Haystack is designed in a modular way and lets you use any models trained with  `FARM <https://github.com/deepset-ai/FARM>`_ or `Transformers <https://github.com/huggingface/transformers>`_.



Core Features
=============
- **Powerful ML models**: Utilize all latest transformer based models (BERT, ALBERT, RoBERTa ...)
- **Modular & future-proof**: Easily switch to newer models once they get published.
- **Developer friendly**: Easy to debug, extend and modify.
- **Scalable**: Production-ready deployments via Elasticsearch backend & REST API
- **Customizable**: Fine-tune models to your own domain & improve them continuously via user feedback


Components
==========

1. **DocumentStore**: Database storing the documents for our search. We recommend Elasticsearch, but have also more light-weight options for fast prototyping (SQL or In-Memory).

2. **Retriever**:  Fast, simple algorithm that identifies candidate passages from a large collection of documents. Algorithms include TF-IDF or BM25, custom Elasticsearch queries, and embedding-based approaches. The Retriever helps to narrow down the scope for Reader to smaller units of text where a given question could be answered.

3. **Reader**: Powerful neural model that reads through texts in detail to find an answer. Use diverse models like BERT, RoBERTa or XLNet trained via `FARM <https://github.com/deepset-ai/FARM>`_ or `Transformers <https://github.com/huggingface/transformers>`_ on SQuAD like tasks. The Reader takes multiple passages of text as input and returns top-n answers with corresponding confidence scores. You can just load a pretrained model from  `Hugging Face's model hub <https://huggingface.co/models>`_ or fine-tune it to your own domain data.

4. **Finder**: Glues together a Reader and a Retriever as a pipeline to provide an easy-to-use question answering interface.

5. **REST API**: Exposes a simple API for running QA search, collecting feedback and monitoring requests

6. **Labeling Tool**: `Hosted version <https://annotate.deepset.ai/login>`_  (Beta), Docker images (coming soon)


Resources
=========
- Tutorial 1  - Basic QA Pipeline: `Jupyter notebook  <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial1_Basic_QA_Pipeline.ipynb>`__  or `Colab <https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial1_Basic_QA_Pipeline.ipynb>`__
- Tutorial 2  - Fine-tuning a model on own data: `Jupyter notebook <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial2_Finetune_a_model_on_your_data.ipynb>`__ or `Colab <https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial2_Finetune_a_model_on_your_data.ipynb>`__
- Tutorial 3  - Basic QA Pipeline without Elasticsearch: `Jupyter notebook <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial3_Basic_QA_Pipeline_without_Elasticsearch.ipynb>`__ or `Colab <https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial3_Basic_QA_Pipeline_without_Elasticsearch.ipynb>`__

Quick Start
===========

Installation
------------

Recommended (because of active development)::

    git clone https://github.com/deepset-ai/haystack.git
    cd haystack
    pip install --editable .

To update your installation, just do a git pull. The --editable flag will update changes immediately.

From PyPi::

    pip install farm-haystack

Usage
-----
.. image:: https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/img/code_snippet_usage.png


Quick Tour
==========


1) DocumentStores
---------------------

Haystack has an extensible DocumentStore-Layer, which is storing the documents for our search. We recommend Elasticsearch, but have also more light-weight options for fast prototyping.

Elasticsearch (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:code:`haystack.database.elasticsearch.ElasticsearchDocumentStore`

* Keeps all the logic to store and query documents from Elastic, incl. mapping of fields, adding filters or boosts to your queries, and storing embeddings
* You can either use an existing Elasticsearch index or create a new one via haystack
* Retrievers operate on top of this DocumentStore to find the relevant documents for a query
* Documents can optionally be chunked into smaller units (e.g. paragraphs) before indexing to make the results returned by the Retriever more granular and accurate.

You can get started by running a single Elasticsearch node using docker::

     docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.1

Or if docker is not possible for you::

     wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-linux-x86_64.tar.gz -q
     tar -xzf elasticsearch-7.6.2-linux-x86_64.tar.gz
     chown -R daemon:daemon elasticsearch-7.6.2
     elasticsearch-7.0.0/bin/elasticsearch

See Tutorial 1 on how to go on with indexing your docs.


SQL / InMemory (Alternative)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:code:`haystack.database.sql.SQLDocumentStore` & :code:`haystack.database.memory.InMemoryDocumentStore`

These DocumentStores are mainly intended to simplify the first development steps or test a prototype on an existing SQL Database containing your texts. The SQLDocumentStore initializes by default a local file-based SQLite database.
However, you can easily configure it for PostgreSQL or MySQL since our implementation is based on SQLAlchemy.
Limitations: Retrieval (e.g. via TfidfRetriever) happens in-memory here and will therefore only work efficiently on small datasets

2) Retrievers
---------------------
ElasticsearchRetriever
^^^^^^^^^^^^^^^^^^^^^^
Scoring text similarity via sparse Bag-of-words representations are strong and well-established baselines in Information Retrieval.
The default `ElasticsearchRetriever` uses Elasticsearch's native scoring (BM25), but can be extended easily with custom queries or filtering.

Example::

    retriever = ElasticsearchRetriever(document_store=document_store, custom_query=None)
    retriever.retrieve(query="Why did the revenue increase?", filters={"years": ["2019"], "company": ["Q1", "Q2"]})
    # returns: [Document, Document]

EmbeddingRetriever
^^^^^^^^^^^^^^^^^^^^^^
Using dense embeddings (i.e. vector representations) of texts is a powerful alternative to score similarity of texts.
This retriever allows you to transform your query into an embedding using a model (e.g. Sentence-BERT) and find similar texts by using cosine similarity.

Example::

    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model="deepset/sentence-bert",
                                   model_format="farm")
    retriever.retrieve(query="Why did the revenue increase?", filters={"years": ["2019"], "company": ["Q1", "Q2"]})
    # returns: [Document, Document]

We are working on extending this category of retrievers a lot as there's a lot of exciting work in research indicating substantial performance improvements (e.g. `DPR <https://arxiv.org/abs/2004.04906>`_ , `REALM <https://arxiv.org/abs/2002.08909>`_  )

TfidfRetriever
^^^^^^^^^^^^^^^^^^^^^^
Basic in-memory retriever getting texts from the DocumentStore, creating TF-IDF representations in-memory and allowing to query them.

3) Readers
---------------------
Neural networks (i.e. mostly Transformer-based) that read through texts in detail to find an answer. Use diverse models like BERT, RoBERTa or XLNet trained via `FARM <https://github.com/deepset-ai/FARM>`_ or  on SQuAD like tasks. The Reader takes multiple passages of text as input and returns top-n answers with corresponding confidence scores.
Both readers can load either a local model or any public model from  `Hugging Face's model hub <https://huggingface.co/models>`_

FARMReader
^^^^^^^^^^
Implementing various QA models via the `FARM <https://github.com/deepset-ai/FARM>`_ Framework.
Example::

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2",
                    use_gpu=False, no_ans_boost=-10, context_window_size=500,
                    top_k_per_candidate=3, top_k_per_sample=1,
                    num_processes=8, max_seq_len=256, doc_stride=128)

    # Optional: Training & eval
    reader.train(...)
    reader.eval(...)

    # Predict
    reader.predict(question="Who is the father of Arya Starck?", documents=documents, top_k=3)

This Reader comes with:
* quite many configuration options
* using multiple processes for preprocessing
* option to train
* option to evaluate

TransformersReader
^^^^^^^^^^^^^^^^^^
Implementing various QA models via the :code:`pipeline` class of `Transformers <https://github.com/huggingface/transformers>`_ Framework.

Example::

    reader = TransformersReader(model="distilbert-base-uncased-distilled-squad",
                                tokenizer="distilbert-base-uncased",
                                context_window_size=500,
                                use_gpu=-1)

    reader.predict(question="Who is the father of Arya Starck?", documents=documents, top_k=3)


5. REST API
---------------------
A simple REST API based on `FastAPI <https://fastapi.tiangolo.com/>`_ is provided to:

*  search answers in texts (`extractive QA  <https://github.com/deepset-ai/haystack/blob/master/haystack/api/controller/search.py>`_)
*  search answers by comparing user question to existing questions (`FAQ-style QA  <https://github.com/deepset-ai/haystack/blob/master/haystack/api/controller/search.py>`_)
*  collect & export user feedback on answers to gain domain-specific training data (`feedback  <https://github.com/deepset-ai/haystack/blob/master/haystack/api/controller/feedback.py>`_)
*  allow basic monitoring of requests (currently via APM in Kibana)

To serve the API, run::

    gunicorn haystack.api.application:app -b 0.0.0.0:80 -k uvicorn.workers.UvicornWorker`

You will find the Swagger API documentation at http://127.0.0.1:80/docs


6. Labeling Tool
---------------------
* Use the `hosted version <https://annotate.deepset.ai/login>`_  (Beta) or deploy it yourself via Docker images (coming soon)  
* Create labels with different techniques: Come up with questions (+ answers) while reading passages (SQuAD style) or have a set of predefined questions and look for answers in the document (~ Natural Questions).
* Structure your work via organizations, projects, users 
* Upload your documents or import labels from an existing SQuAD-style dataset
* Coming soon: more file formats for document upload, metrics for label quality ...

.. image:: https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/img/annotation_tool.png
