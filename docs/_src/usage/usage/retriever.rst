
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
These retrievers don't need to be trained and will work on any language.

More recently, **dense** approaches such as Dense Passage Retrieval (DPR) have shown even better performance than their sparse counter parts.
These methods embed both document and query into a shared embedding space using deep neural networks
and the top candidates are the nearest neighbour documents to the query.
These models are usually langauge specific.

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

`Dense Passage Retrieval <https://arxiv.org/abs/2004.04906>`_ is a highly performant retrieval method that calculates relevance using dense representations.
Two separate transformer models are used to encode documents and queries
and the dot product simalrity of their resultant embeddings is the metric by which they are ranked.
The original implementation use two BERT base uncased models but DPR models could in theory be built for other model architectures and languages.

!! Diagram !!

Indexing using DPR is comparatively expensive in terms of required computation since all documents in the database need to be processed through the transformer.
The embeddings that are creating in this step can be stored in FAISS, a database optimized for vector similarity.
DPR can also work with the ElasticsearchDocumentStore or the InMemoryDocumentStore.

There are two design decisions that have made DPR particularly performant.
The use of separate passage and query encoders is well suited to the task of information retrieval
since the language of queries is very different to that of passages.
For one, they are usually significantly shorter.

Also DPR is trained using a method known as in-batch negatives.
This approach uses gold label passages in the same batch as negative examples
and makes for a highly efficient training regime when paired with dot product similarity.

In Haystack, you can simply download the pretrained encoders needed to start using DPR.
If you'd like to learn how to set up a DPR based system, have a look at our tutorial !! Link !!

!! Code Snippet !!

!! Training in future? !!

!! Talk more about benchmarks, SoTA, results !!

Embedding Retrieval
-------------------

In Haystack, you also have the option of using a single transformer model to encode document and query.
One style of model that is suited to this kind of retrieval is that of `Sentence Transformers <https://github.com/UKPLab/sentence-transformers>`_.
These models are trained in Siamese Networks and use triplet loss such that they learn to embed similar sentences near to each other in a shared embedding space.

They are particular suited to cases where your query input is similar in style to that of the documents in your database
i.e. when you are searching for most similar documents.
This is not inherently suited to query based search where the length, language and format of the query usually significantly differs from the searched for text.

!! Code Snippet !!

Choosing Top K
--------------

Top K is configurable
How to choose an appropriate K?
What about all those params? Top k per candidate / top k per sample etc
