
Retriever
=========

The Retriever is a lightweight filter that can quickly go through the full document store and propose a set of candidate passages to be passed to the reader
gets a sense of the topics being mentioned in a document but doesn't pay as much close attention as the Reader
In large document collections, majority of documents irrelevant to specific queries
retriever is a quick way of selecting only the most relevant
Important to speedup and scaling
Saves the Reader work

Traditional approaches like TF-IDF and BM25 are keyword based (a.k.a. sparse methods) and have proven to be very effective.
New deep learning based approaches (also referred to as dense methods) such as Dense Passage Retrieval offer even better performance.
While they are computationally intensive during indexing, they are designed to be fast during querying.

It is important to mention, however, that there are qualitative differences between the two styles of retrieval.
While dense methods are very capable of building strong semantic representations of text,
they struggle when encountering out-of-vocabulary words.
By contrast, sparse methods do not keep a learned vocab and are not thrown off by novel words or names.

!! Show example from DPR paper? !!


Points to doc store (link) when initialized
Combined with Reader using the Finder

Code snippets for both


BM25 and TF-IDF
---------------

The Algorithm
~~~~~~~~~~~~~

Industry standard keyword based doc retrieval algo
An improvement on TF IDF
Explain TF
Explain IDF
BM25 includes a couple improvements
Such as....

Inverted index speeds this up

In Haystack
~~~~~~~~~~~

Diagram showing TFIDF vs BM25

Is TFIDF implemented?
How does this interact with database? how is the inverted index built for BM25?

Basic code example - point to a tutorial

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
