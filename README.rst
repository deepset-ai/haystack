*******************************************************
Haystack — Natural Language Question Answering At Scale
*******************************************************


Introduction
============

A system built on top of FARM Framework to perform NLP question answering on a collection of large documents.

Inference for QA using current state-of-the-art models is computationally expensive. To make scaling question answering on documents pragmatic, document retrieval
techniques are used to narrow down the scope to small subset of paragraphs across documents where an answer to the question could potentially be.

The system is designed with a goal to be modular. Individual components can be customized or new ones can be incorporated with minimal effort.


Components
==========

There are three major components for the question answering pipeline:

1. Reader implements inference on FARM Adaptive Models trained on SQuaD like tasks to perform question answering. It takes paragraphs of text as input and returns answers with corresponding confidence scores.


2. Retriever is an implementation of term frequency–inverse document frequency(tf-idf) numerical statistic similar to the query scoring functions used in popular search systems like Elasticsearch. Retriever helps to narrow down the scope for Reader to smaller units of text where a given question could be answered.


3. Finder is a pipeline to glue together instance of a Reader and a Retriever to provide an easy-to-use question answering interface.


Quickstart
==========

Installation
------------
There are two ways to install:

* (recommended) from source, :code:`git clone <url>` and run :code:`pip install [--editable] .` from the root of the repositry. 
* from PyPI, do a :code:`pip install farm_haystack`

Configuration
-------------
The configuration can be supplied in a :code:`qa_config.py` placed in the PYTHONPATH. Alternatively, the :code:`DATABASE_URL` can also be set an an environment variable.


Deployment
==========

SQL Backend
-----------
The database ORM layer is implemented using SQLAlchemy library. By default, it uses the file-based SQLite database. For large scale deployments, the configuration can be changed to use other compatible databases like PostgreSQL or MySQL.

REST API
--------
A Flask based HTTP REST API is included to use the QA Framework with UI or integrating with other systems. To serve the API, run :code:`FLASK_APP=farm_hackstack.api.inference flask run`. 


