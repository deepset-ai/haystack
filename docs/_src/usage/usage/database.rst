Document Stores
===============

Initialisation
--------------


Initialising a new Document Store is straight forward

.. code-block:: python

    document_store = ElasticsearchDocumentStore()


Preparing Documents
-------------------


Document Stores expect the documents in your corpus to be passed in in the following dictionary form.

.. code-block:: python

    {
        'text': DOCUMENT_TEXT_HERE,
        'meta': {'name': DOCUMENT_NAME, ...}
    }

Haystack also has a convert_files_to_dicts() function that will convert
all txt or pdf files in a given folder into this dictionary format.

.. code-block:: python

    dicts = convert_files_to_dicts(dir_path=doc_dir)

Indexing Documents
------------------

To add documents, simply use the following method:

.. code-block:: python

    document_store.write_documents(dicts)

Note that this indexes the document and its meta data, but does not compute the embeddings
needed for dense retrievers such as the DensePassageRetriever or the EmbeddingRetriever.
For these models, you will also have to use the following lines.

.. code-block:: python

    document_store.update_embeddings(retriever)

This step is computationally intensive since it will engage the transformer based encoders.
Having a GPU acceleration will significantly speed up this step.

Connecting to the Retriever
---------------------------

The document store is passed in as an argument when the Retriever is being initialised.

.. code-block:: python

    retriever = EmbeddingRetriever(document_store=document_store)

Choosing the right database
---------------------------

Document storage is important
There are many types and each has implications on memory consumption, indexing and querying

Talk about trade offs
Elasticsearch vs SQL vs In Memory vs FAISS

Show some code snippets of each using tab elements

Use tabbed element to show how each is initialized

Indexing
--------

Code snippets of how to index to each
maybe use tab elements if diff for each

