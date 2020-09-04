Document Stores
===============

Initialisation
--------------

Initialising a new Document Store is straight forward.

.. tabs::

    .. tab:: Elasticsearch

        .. code-block:: python

            document_store = ElasticsearchDocumentStore()

    .. tab:: FAISS

        .. code-block:: python

            document_store = FAISSDocumentStore()

    .. tab:: SQL

        .. code-block:: python

            document_store = SQLDocumentStore()

    .. tab:: In Memory

        .. code-block:: python

            document_store = InMemoryDocumentStore()

Each DocumentStore constructor allows for arguments specifying how to connect to existing databases and the names of indexes.
See API documentation for more info.

Preparing Documents
-------------------

DocumentStores expect Documents in dictionary form, like that below.
They are loaded using the ``DocumentStore.write_documents()`` method.

.. code-block:: python

    document_store = ElasticsearchDocumentStore()
    dicts = [
        {
            'text': DOCUMENT_TEXT_HERE,
            'meta': {'name': DOCUMENT_NAME, ...}
        }, ...
    ]
    document_store.write_documents(dicts)

File Conversion
---------------

There are a range of different file converters in Haystack that can help get your data into the right format.
Haystack features support for txt, pdf and docx formats and there is even a converted that leverages Apache Tika.
See the File Converters section in the API docs for more information.

..
   _comment: !! Code snippets for each type !!

Haystack also has a ``convert_files_to_dicts()`` utility function that will convert
all txt or pdf files in a given folder into this dictionary format.

.. code-block:: python

    document_store = ElasticsearchDocumentStore()
    dicts = convert_files_to_dicts(dir_path=doc_dir)
    document_store.write_documents(dicts)

Writing Documents
-----------------

Haystack allows for you to write store documents in an optimised fashion so that query times can be kept low.

For Sparse Retrievers
~~~~~~~~~~~~~~~~~~~~~

For **sparse**, keyword based retrievers such as BM25 and TF-IDF,
you simply have to call ``DocumentStore.write_documents()``.
The creation of the inverted index which optimises querying speed is handled automatically.

.. code-block:: python

    document_store.write_documents(dicts)

For Dense Retrievers
~~~~~~~~~~~~~~~~~~~~

For **dense** neural network based retrievers like Dense Passage Retrieval, or Embedding Retrieval,
indexing involves computing the Document embeddings which will be compared against the Query embedding.

The storing of the text is handled by ``DocumentStore.write_documents()`` and the computation of the
embeddings is started by ``DocumentStore.update_embeddings()``.

.. code-block:: python

    document_store.write_documents(dicts)
    document_store.update_embeddings(retriever)

This step is computationally intensive since it will engage the transformer based encoders.
Having GPU acceleration will significantly speed this up.

..
   _comment: !! Diagrams of inverted index / document embeds !!
..
   _comment: !! Make this a tab element to show how different datastores are initialized !!

Choosing the right database
---------------------------

Document storage is important
There are many types and each has implications on memory consumption, indexing and querying

Talk about trade offs
Elasticsearch vs SQL vs In Memory vs FAISS

Show some code snippets of each using tab elements

Use tabbed element to show how each is initialized
