Reader
======

Core tech that powers Haystack
Deep neural networks, currently almost exclusively transformers
a.k.a. closed domain QA systems in ML speak
Performs close reading of text to extract best answers

From Language Model to Reader
-----------------------------

See blog article for more about how these reader models
adapt language models
pick a best start and end token
deal with documents of any length
Handle other answer types like no answer

Choosing the Right Model
------------------------

A general purpose language model further trained on a QA dataset, currently SQuAD is standard
Many options out there right now (BERT, RoBERTa, ELECTRA)
Evaluate on a trade off of accuracy and speed
Look at our benchmarks page and also SQuAD leaderboards
Models such as RoBERTa base trained on SQuAD (give model hub model name) are a good starting point - performant and can also run on V100
Very promising next generation of distilled models which shrink down size but retain performance
ALBERT XL is best performance but practically speaking too large

Languages other than English
----------------------------

By default, we have so far been talking about English Readers
If you want to do QA in a language other than English, there are a few different routes

1) Train on a non-English dataset
e.g. if you're working in French, take a French LM like CamemBERT (link), train on FQuAD (link)
In the LM world there is a lot of coverage of different languages
In terms of QA datasets, it is limited (see blog) (German coming soon?)

2) Zero shot
Models like multilingual BERT, XLM and XLM-RoBERTa are trained to handle a broad range of languages
zero shot learning is to train a German Reader without any German QA data
The process goes somehting like this: Teach a multilingual model to do QA in English (or any other with a good training dataset)
If trained well, the hope is that the model should learn the language independent principles of QA
You can start using it for your language
In practice, zero shotting has effect but not yet really performant

Retriever
=========

The Retriever is a lightweight filter that can quickly go through the full document store and propose a set of candidate passages to be passed to the reader
gets a sense of the topics being mentioned in a document but doesn't pay as much close attention as the Reader
In large document collections, majority of documents irrelevant to specific queries
retriever is a quick way of selecting only the most relevant
Important to speedup and scaling
Saves the Reader work

BM25
----

The Algorithm
~~~~~~~~~~~~~

Industry standard keyword based doc retrieval algo
An improvement on TF IDF
Explain TF
Explain IDF
BM25 includes a couple improvements
Such as....

In Haystack
~~~~~~~~~~~

Diagram showing TFIDF vs BM25

Is TFIDF implemented?
How does this interact with database? how is the inverted index built for BM25?

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







Choosing Top K
--------------

Scaling to Open Domain???
=========================

Variable Length Documents
-------------------------

See blog article for sliding window approach

Variable Numbers of Documents
-----------------------------

How do we do this actually? Direct comparison of diff doc candidates? Top k?

Different Styles of QA?
-----------------------

Squad = no answer + span
yes/no
See blog article

Database
========

Domain Adaptation
=================

Other Features
==============

Rest API
--------

Labelling Tool
--------------

Input File Formats
------------------



