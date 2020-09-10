*******************************************************
Haystack — Neural Question Answering At Scale
*******************************************************
.. image:: https://github.com/deepset-ai/haystack/workflows/Build/badge.svg?branch=master
	:target: https://github.com/deepset-ai/haystack/actions
	:alt: Build

.. image:: https://camo.githubusercontent.com/34b3a249cd6502d0a521ab2f42c8830b7cfd03fa/687474703a2f2f7777772e6d7970792d6c616e672e6f72672f7374617469632f6d7970795f62616467652e737667
	:target: http://mypy-lang.org/
	:alt: Checked with mypy

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
While QA is the focussed use case for Haystack, we will address further options around neural search in the future (re-ranking, most-similar search ...).

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

.. image:: https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/img/sketched_concepts_white.png


1. **DocumentStore**: Database storing the documents for our search. We recommend Elasticsearch, but have also more light-weight options for fast prototyping (SQL or In-Memory).

2. **Retriever**:  Fast, simple algorithm that identifies candidate passages from a large collection of documents. Algorithms include TF-IDF or BM25, custom Elasticsearch queries, and embedding-based approaches. The Retriever helps to narrow down the scope for Reader to smaller units of text where a given question could be answered.

3. **Reader**: Powerful neural model that reads through texts in detail to find an answer. Use diverse models like BERT, RoBERTa or XLNet trained via `FARM <https://github.com/deepset-ai/FARM>`_ or `Transformers <https://github.com/huggingface/transformers>`_ on SQuAD like tasks. The Reader takes multiple passages of text as input and returns top-n answers with corresponding confidence scores. You can just load a pretrained model from  `Hugging Face's model hub <https://huggingface.co/models>`_ or fine-tune it to your own domain data.

4. **Finder**: Glues together a Reader and a Retriever as a pipeline to provide an easy-to-use question answering interface.

5. **REST API**: Exposes a simple API for running QA search, collecting feedback and monitoring requests

6. **Haystack Annotate**: Create custom QA labels, `Hosted version <https://annotate.deepset.ai/login>`_  (Beta), Docker images (coming soon)


Resources
=========

- Tutorial 1  - Basic QA Pipeline: `Jupyter notebook  <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial1_Basic_QA_Pipeline.ipynb>`__  or `Colab <https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial1_Basic_QA_Pipeline.ipynb>`_
- Tutorial 2  - Fine-tuning a model on own data: `Jupyter notebook <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial2_Finetune_a_model_on_your_data.ipynb>`__ or `Colab <https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial2_Finetune_a_model_on_your_data.ipynb>`__
- Tutorial 3  - Basic QA Pipeline without Elasticsearch: `Jupyter notebook <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial3_Basic_QA_Pipeline_without_Elasticsearch.ipynb>`__ or `Colab <https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial3_Basic_QA_Pipeline_without_Elasticsearch.ipynb>`__
- Tutorial 4  - FAQ-style QA: `Jupyter notebook <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial4_FAQ_style_QA.ipynb>`__ or `Colab <https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial4_FAQ_style_QA.ipynb>`__
- Tutorial 5  - Evaluation of the whole QA-Pipeline: `Jupyter noteboook <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial5_Evaluation.ipynb>`__ or `Colab <https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial5_Evaluation.ipynb>`__
- Tutorial 6  - Better Retrievers via "Dense Passage Retrieval": `Jupyter noteboook <https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial6_Better_Retrieval_via_DPR.ipynb>`__ or `Colab <https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial6_Better_Retrieval_via_DPR.ipynb>`__


Quick Start
===========

Installation
------------

PyPi::

    pip install farm-haystack

Master branch (if you wanna try the latest features)::

    git clone https://github.com/deepset-ai/haystack.git
    cd haystack
    pip install --editable .

To update your installation, just do a git pull. The --editable flag will update changes immediately.

Note: On Windows you might need :code:`pip install farm-haystack -f https://download.pytorch.org/whl/torch_stable.html` to install PyTorch correctly

Usage
-----
.. image:: https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/img/code_snippet_usage.png


Quick Tour
==========


1) DocumentStores
---------------------

Haystack offers different options for storing your documents for search. We recommend Elasticsearch, but have also light-weight options for fast prototyping and will soon add DocumentStores that are optimized for embeddings (FAISS & Co).

Elasticsearch (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:code:`haystack.database.elasticsearch.ElasticsearchDocumentStore`

* Keeps all the logic to store and query documents from Elastic, incl. mapping of fields, adding filters or boosts to your queries, and storing embeddings
* You can either use an existing Elasticsearch index or create a new one via haystack
* Retrievers operate on top of this DocumentStore to find the relevant documents for a query
* Documents should be chunked into smaller units (e.g. paragraphs) before indexing to make the results returned by the Retriever more granular and accurate.

You can get started by running a single Elasticsearch node using docker::

     docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2

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

DensePassageRetriever
^^^^^^^^^^^^^^^^^^^^^^
Using dense embeddings (i.e. vector representations) of texts is a powerful alternative to score similarity of texts.
This retriever uses two BERT models - one to embed your query, one to embed your passage. It's based on the work of
`Karpukhin et al <https://arxiv.org/abs/2004.04906>`_ and is especially an powerful alternative if there's no direct overlap between tokens in your queries and your texts.

Example

.. code-block:: python

    retriever = DensePassageRetriever(document_store=document_store,
                                      embedding_model="dpr-bert-base-nq",
                                      do_lower_case=True, use_gpu=True)
    retriever.retrieve(query="Why did the revenue increase?")
    # returns: [Document, Document]

ElasticsearchRetriever
^^^^^^^^^^^^^^^^^^^^^^
Scoring text similarity via sparse Bag-of-words representations are strong and well-established baselines in Information Retrieval.
The default :code:`ElasticsearchRetriever` uses Elasticsearch's native scoring (BM25), but can be extended easily with custom queries or filtering.

Example

.. code-block:: python

    retriever = ElasticsearchRetriever(document_store=document_store, custom_query=None)
    retriever.retrieve(query="Why did the revenue increase?", filters={"years": ["2019"], "company": ["Q1", "Q2"]})
    # returns: [Document, Document]


EmbeddingRetriever
^^^^^^^^^^^^^^^^^^^^^^
This retriever uses a single model to embed your query and passage (e.g. Sentence-BERT) and finds similar texts by using cosine similarity. This works well if your query and passage are a similar type of text, e.g. you want to find the most similar question in your FAQ given a user question.

Example

.. code-block:: python

    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model="deepset/sentence_bert",
                                   model_format="farm")
    retriever.retrieve(query="Why did the revenue increase?", filters={"years": ["2019"], "company": ["Q1", "Q2"]})
    # returns: [Document, Document]

TfidfRetriever
^^^^^^^^^^^^^^^^^^^^^^
Basic in-memory retriever getting texts from the DocumentStore, creating TF-IDF representations in-memory and allowing to query them.
Simple baseline for quick prototypes. Not recommended for production.

3) Readers
---------------------
Neural networks (i.e. mostly Transformer-based) that read through texts in detail to find an answer. Use diverse models like BERT, RoBERTa or XLNet trained via `FARM <https://github.com/deepset-ai/FARM>`_ or on SQuAD-like datasets. The Reader takes multiple passages of text as input and returns top-n answers with corresponding confidence scores.
Both readers can load either a local model or any public model from  `Hugging Face's model hub <https://huggingface.co/models>`_

FARMReader
^^^^^^^^^^
Implementing various QA models via the `FARM <https://github.com/deepset-ai/FARM>`_ Framework.

Example

.. code-block:: python

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

* extensive configuration options (no answer boost, aggregation options ...)
* multiprocessing to speed-up preprocessing
* option to train
* option to evaluate
* option to load all QA models directly from HuggingFace's model hub

TransformersReader
^^^^^^^^^^^^^^^^^^
Implementing various QA models via the :code:`pipeline` class of `Transformers <https://github.com/huggingface/transformers>`_ Framework.

Example

.. code-block:: python

    reader = TransformersReader(model="distilbert-base-uncased-distilled-squad",
                                tokenizer="distilbert-base-uncased",
                                context_window_size=500,
                                use_gpu=-1)

    reader.predict(question="Who is the father of Arya Starck?", documents=documents, top_k=3)


5. REST API
---------------------
A simple REST API based on `FastAPI <https://fastapi.tiangolo.com/>`_ is provided to:

*  search answers in texts (`extractive QA  <https://github.com/deepset-ai/haystack/blob/master/rest_api/controller/search.py>`_)
*  search answers by comparing user question to existing questions (`FAQ-style QA  <https://github.com/deepset-ai/haystack/blob/master/rest_api/controller/search.py>`_)
*  collect & export user feedback on answers to gain domain-specific training data (`feedback  <https://github.com/deepset-ai/haystack/blob/master/rest_api/controller/feedback.py>`_)
*  allow basic monitoring of requests (currently via APM in Kibana)

To serve the API, adjust the values in :code:`rest_api/config.py` and run::

    gunicorn rest_api.application:app -b 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker -t 300

You will find the Swagger API documentation at http://127.0.0.1:8000/docs


6. Labeling Tool
---------------------
* Use the `hosted version <https://annotate.deepset.ai/login>`_  (Beta) or deploy it yourself via Docker images (coming soon)
* Create labels with different techniques: Come up with questions (+ answers) while reading passages (SQuAD style) or have a set of predefined questions and look for answers in the document (~ Natural Questions).
* Structure your work via organizations, projects, users
* Upload your documents or import labels from an existing SQuAD-style dataset
* Coming soon: more file formats for document upload, metrics for label quality ...

.. image:: https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/img/annotation_tool.png


7. Indexing PDF / Docx files
-----------------------------

Haystack has basic converters to extract text from PDF and Docx files. While it's almost impossible to cover all types, layouts and special cases in PDFs, the implementation covers the most common formats and provides basic cleaning functions to remove header, footers, and tables. Multi-Column text layouts are also supported.
The converters are easily extendable, so that you can customize them for your files if needed.

Example:

.. code-block:: python

    #PDF
    from haystack.indexing.file_converters.pdf import PDFToTextConverter
    converter = PDFToTextConverter(remove_header_footer=True, remove_numeric_tables=True, valid_languages=["de","en"])
    pages = converter.extract_pages(file_path=file)
    # => list of str, one per page
    #DOCX
    from haystack.indexing.file_converters.docx import DocxToTextConverter
    converter = DocxToTextConverter()
    paragraphs = converter.extract_pages(file_path=file)
    #  => list of str, one per paragraph (as docx has no direct notion of pages)

Advanced document convertion is enabled by leveraging mature text extraction library `Apache Tika <https://tika.apache.org/>`_, which is mostly written in Java. Although it's possible to call Tika API from Python, the current :code:`TikaConverter` only supports RESTful call to a Tika server running at localhost. One may either run Tika as a REST service at port 9998 (default), or to start a `docker container for Tika <https://hub.docker.com/r/apache/tika/tags>`_. The latter is recommended, as it's easily scalable, and does not require setting up any Java runtime environment. What's more, future update is also taken care of by docker.
Either way, TikaConverter makes RESTful calls to convert any document format supported by Tika. Example code can be found at :code:`indexing/file_converters/utils.py`'s :code:`tika_convert)_files_to_dicts` function:

:code:`TikaConverter` supports 341 file formats, including

* most common text file formats, e.g. HTML, XML, Microsoft Office OLE2/XML/OOXML, OpenOffice ODF, iWorks, PDF, ePub, RTF, TXT, RSS, CHM...
* text embedded in media files, e.g. WAV, MP3, Vorbis, Flac, PNG, GIF, JPG, BMP, TIF, PSD, WebP, WMF, EMF, MP4, Quicktime, 3GPP, Ogg, FLV...
* mail and database files, e.g. Unitx mailboxes, Outlook PST/MSG/TNEF, SQLite3, Microsoft Access, dBase...
* and many more other formats...
* and all those file formats in archive files, e.g. TAR, ZIP, BZip2, GZip 7Zip, RAR!

Check out complete list of files supported by the most recent `Apache Tika 1.24.1 <https://tika.apache.org/1.24.1/formats.html>`_.
If you feel adventurous, Tika even supports some image OCR with Tesseract, or object recognition for image and video files. (not implemented yet)

:code:`TikaConverter` also makes a document's metadata available, including typical fields like file name,  file dates and a lot more (e.g. Author and keywords for PDF if they're available in the files), which may save you some time in data labeling or other downstream tasks.

.. code-block:: python

    converter = TikaConverter(remove_header_footer=True)
    pages = converter.extract_pages(file_path=path)
    pages, meta = converter.extract_pages(file_path=path, return_meta=True)

Contributing
=============
We are very open to contributions from the community - be it the fix of a small typo or a completely new feature! You don't need to be an Haystack expert for providing meaningful improvements.
To avoid any extra work on either side, please check our `Contributor Guidelines <https://github.com/deepset-ai/haystack/blob/master/CONTRIBUTING.md>`_ first.

Tests will automatically run for every commit you push to your PR. You can also run them locally by executing `pytest <https://docs.pytest.org/en/stable/>`_   in your terminal from the root folder of this repository: 

.. code-block:: bash

    pytest test/
