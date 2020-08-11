
Retriever
=========

The Retriever is a lightweight filter that can quickly go through the full document store and pass a set of candidate documents to the Reader.
It gets a sense of the topics being mentioned in a document but doesn't pay as much close attention to the finer details as the Reader does.

When searching through large collections of documents, its usually the case that more documents are irrelevant to the query than not.
The Retriever is an efficient tool for sifting out the obvious negative cases and saves the Reader from doing more work than it needs to.

We currently recommend using BM25 if you're looking for a sparse option, or Dense Passage Retrieval if you're looking for a dense option.
Haystack also has support for TF-IDF and embedding retrieval.


!! Example speedup from slides !!

!! Benchmarks !!

.. code-block:: python

    retriever = ElasticsearchRetriever(document_store)
    finder = Finder(reader, retriever)


Note that not all Retrievers can be paired with every DocumentStore.
Here is a table showing which combinations are supported:

+-----------+--------+---------------+-----+-------+
|           | Memory | Elasticsearch | SQL | FAISS |
+-----------+--------+---------------+-----+-------+
|    BM25   |    N   |       Y       |  N  |   N   |
+-----------+--------+---------------+-----+-------+
|   TF-IDF  |    Y   |       Y       |  Y  |   N   |
+-----------+--------+---------------+-----+-------+
| Embedding |    Y   |       Y       |  N  |   Y   |
+-----------+--------+---------------+-----+-------+
|    DPR    |    Y   |       Y       |  N  |   Y   |
+-----------+--------+---------------+-----+-------+

Dense vs Sparse
---------------

Broadly speaking, retrieval methods can be split into two categories: **dense** and **sparse**.

**Sparse** methods, like TF-IDF and BM25, operate by looking for shared keywords between the document and query.
They have proven to be a simple but effective approach to the problem of search.

More recently, **dense** approaches such as Dense Passage Retrieval (DPR) have shown even better performance than their sparse counter parts.
These methods embed both document and query into a shared embedding space using deep neural networks
and the top candidates are the nearest neighbour documents to the query.

Terminology
~~~~~~~~~~~

!! Diagram of what a sparse vector looks like vs dense vector !!

The terms **dense** and **sparse** refer to the representations that the algorithms build for each document and query.
**Sparse** methods characterise texts using vectors with one dimension corresponding to each word in the vocabulary.
Dimensions will be zero if the word is absent and non-zero if it is present.
Since most documents contain only a small subset of the full vocabulary,
these vectors are considered sparse since non-zero values are few and far between.

**Dense** methods, by contrast, pass text as input into neural network encoders
and represent text in a vector of a manually defined size (!! what is default size of DPR vecs? !!).
Though individual dimensions are not mapped to any corresponding vocabulary or linguistic feature,
each dimension encodes some information about the text.
There are rarely 0s in these vectors hence their relative density.

Qualitative Differences
~~~~~~~~~~~~~~~~~~~~~~~

Between these two types there are also some qualitative differences too.
For example, sparse methods treat text as a bag-of-words meaning that they do not take word order and syntax into account,
while the latest generation of dense methods use transformer based encoders
which are designed to be sensitive to these factors.

Also dense methods are very capable of building strong semantic representations of text,
but they struggle when encountering out-of-vocabulary words such as new names.
By contrast, sparse methods don't need to learn representations of words,
they only care about whether they are present or absent in the text.
As such, they handle out-of-vocabulary words with no problem.

!! Show example from DPR paper? !!

Indexing
~~~~~~~~

Dense methods perform indexing by processing all the documents through a neural network and storing the resultant vectors.
This is a much more expensive operation than the creation of the inverted-index in sparse methods
and will require significant computational power and time.
See their individual sections (!! link !!) for more details on this point.

!! Benchmark? !!

TF-IDF
------

TF-IDF is a commonly used baseline for information retrieval that exploits two key intuitions:

* documents that have more lexical overlap with the query are more likely to be relevant
* words that occur in fewer documents are more significant than words that occur in many documents

Given a query, a tf-idf score is computed for each document as follows:

.. code-block:: python

    score = tf * idf

Where ``tf`` is how many times words in the query occur in that document
and ``idf`` is the inverse of the fraction of documents containing the word.
In practice, both terms are usually log normalised.

In Haystack, you can use TF-IDF simply by initialising a ``TfidfRetriever``

.. code-block:: python

    document_store = InMemoryDocumentStore()
    ...
    retriever = TfidfRetriever(document_store)

If you'd like to learn more about the exact details of the algorithm,
have a look at !!link!!

BM25
----

BM25 is a variant of TF-IDF that we recommend you use if you are looking for a retrieval method that does not need a neural network for indexing.
It improves upon its predecessor in two main aspects:

* It saturates ``tf`` after a set number of occurrences of the given term in the document
* It normalises by document length so that short documents are favoured over long documents if they have the same amount of word overlap with the query

Haystack uses the Elasticsearch implementation of BM25 and as such needs to be paired with the ``ElasticsearchDocumentStore``

.. code-block:: python

    document_store = ElasticsearchDocumentStore()
    ...
    retriever = ElasticsearchRetriever(document_store)

See `this <https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables>`_ blog post for more details about the algorithm.

!! Diagram showing TFIDF vs BM25 !!

Dense Passage Retrieval
-----------------------

The Algorithm
~~~~~~~~~~~~~

The paper
Modern transformer baseed language models have shown great success in representing the semantics of natural language in limited length embeddings
They are good candidates for retrieval
Dense Passage Retrieval is conceptually straight forward but also very effective

Diagram!!

Turns both query and doc into embeddings
distance measure (WHCIH ONE) determines whether the model thinks the passage is relevant to query
Training is done effectively via in batch negatives

One encoder for query, one encoder for passage
This is necessary since the nature of the query and documents is very different
Yet they have to be embedded into a common vector space

Sets it apart from single encoder systems for similarity (Google's USE)
BERT doesn't really use sentence distribution in training, only word distribution
Siamese networks like sentence transformers have not proven to be so effective in practice

Index time significant
Query can be very fast with vector similarity database

In Haystack
~~~~~~~~~~~

Use pretrained Query and Document encoders (Can they be retrained?)
Performance is best using a vector optimized database e.g. FAISS (Can we use other databases?)
This will add time to indexing since all documents need to be put through an encoder
But this ensures strong performance without adding much time to querying (Is querying any slower?)

Code example and point to tutorial

Embedding Retrieval
-------------------

Choosing Top K
--------------

Top K is configurable
How to choose an appropriate K?
What about all those params? Top k per candidate / top k per sample etc
