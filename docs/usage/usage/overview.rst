Conceptual Overview
===================

Retriever-Reader Pipeline
-------------------------

Queries in Haystack are processed in two distinct stages handled by a **Retriever** and a **Reader**.

!! Diagram of Reader Retriever Pipeline !!

**Readers**, also known as Open-Domain QA systems in Machine Learning speak,
are powerful models that do close analysis of documents and perform the core task of question answering.
The Readers in Haystack are trained from the latest transformer based language models and can be significantly sped up using GPU acceleration (!! benchmarks link !!)
However, it is not currently feasible to apply to use the Reader directly on large collection of documents.

!! Image of What a reader does and maybe architecture !!

The **Retriever** assists the Reader by proposing a small set of candidate documents for the Reader to process.
It functions as a lightweight filter that can scan through all documents in the database,
quickly identifying the relevant and dismissing the irrelevant.
Current methods are described as being either sparse (i.e. keyword based) or dense (i.e. neural network based).
Though dense methods require significantly more processing time during indexing,
both are designed to be fast enough that the Retriever can be applied to the full database with each query.

Indexing and Querying
---------------------

**Indexing** and **querying** are two separate but equally important processes in Haystack.

To **index** is to store your documents in a way that will optimize your search.
It is performed just once, at the point of loading the data into your database.
For sparse keyword based retrievers, this involves the creation of an inverted index that maps words to the documents which contain them.
For dense neural network based retrievers, indexing involves computing the document embeddings which will be compared against the query embedding.

!! Diagrams of inverted index / document embeds !!

Here is an example of how you index your documents in Haystack using an ``ElasticsearchDocumentStore``.

!! Make this a tab element to show how different datastores are initialized !!

.. code-block::

    # Database to store your docs
    document_store = ElasticsearchDocumentStore()

    # Clean & index your docs
    dicts = convert_files_to_dicts(doc_dir, clean_func=clean_wiki_text)
    document_store.write_documents(dicts)

**Querying** involves searching for an answer to a given question within the full document store.
This process will:
* make the Retriever filter for a small set of relevant candidate documents
* get the Reader to process this set of candidate documents
* return potential answers to the given question

Generally speaking, there are much tighter time constraints on querying and so in Haystack, it is a much more lightweight operation.
Indexing should precompute any of the results that might be useful at query time.

In Haystack, querying is performed on a ``Finder`` object which connects the reader to the retriever.

.. code-block::

    # The Finder sticks together reader and retriever in a pipeline to answer our questions.
    finder = Finder(reader, retriever)

    # Voilà! Ask a question!
    question = "Who is the father of Sansa Stark?"
    prediction = finder.get_answers(question)

Transformers
------------
What are they
not to be confused with HF Transformers