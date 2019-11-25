*******************************************************
Haystack — Neural Question Answering At Scale
*******************************************************


Introduction
============

The performance of **modern Question Answering Models** (BERT, ALBERT ...) has seen drastic improvements within the last year enabling many new opportunities for finding information more efficiently. However, those models are usually designed to find answers within rather small text passages. **Haystack let's you scale QA models** to large collections of documents!

Haystack is designed in a modular way and is tightly integrated with the `FARM <https://github.com/deepset-ai/FARM>`_ framework for training QA models.
Swap your models easily from BERT to roBERTa and scale the database from dev (Sqlite) to production (PostgreSQL, elasticsearch ...).

Core Features
==========
- **Most powerful models**: Utilize all the latest transformer based models (BERT, ALBERT roBERTa ...)
- **Modular & future-proof**: Avoid technical debt. With haystack you can easily switch to newer models once they get published. 
- **Developer friendly**: Easy to debug, extend and modify
- **Scalable**: Switch from dev to production within minutes.  

Components
==========

1. **Retriever**:  Fast, simple model that identify candidate passages from a large collection of documents. Algorithms include TF-IDF, which is similar to what's used in popular search systems like Elasticsearch. The Retriever helps to narrow down the scope for Reader to smaller units of text where a given question could be answered.

2. **Reader**: Powerful neural model that read through texts in detail to find an answer. Use diverse models like BERT, Roberta or XLNet trained via the `FARM <https://github.com/deepset-ai/FARM>`_ Framework on SQuAD like tasks. The Reader takes multiple passages of text as input and returns top-n answers with corresponding confidence scores.

3. **Finder**: Glues together a Reader and a Retriever as a pipeline to provide an easy-to-use question answering interface.

4. **Labeling Tool**: (Coming soon)


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


