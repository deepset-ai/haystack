<!---
title: "Retriever"
metaTitle: "Retriever"
metaDescription: ""
slug: "/docs/retriever"
date: "2020-09-03"
id: "retrievermd"
--->

# Retriever

The Retriever is a lightweight filter that can quickly go through the full document store and pass on a set of candidate documents that are relevant to the query.
When used in combination with a Reader, it is a tool for sifting out the obvious negative cases, saving the Reader from doing more work than it needs to and speeding up the querying process.

<div class="recommendation">

**Recommendations**

* BM25 (sparse)

* Dense Passage Retrieval (dense)

</div>

<!-- _comment: !! Example speedup from slides !! -->
<!-- _comment: !! Benchmarks !! -->
Note that not all Retrievers can be paired with every DocumentStore.
Here are the combinations which are supported:

| | Memory | Elasticsearch | SQL | FAISS | Milvus |
| --- | --- | --- | ---- | ---- | ---- |
| BM25 | N | Y | N | N | N |
| TF-IDF | Y | Y | Y | N | N |
| Embedding | Y | Y | N | Y | Y |
| DPR | Y | Y | N | Y | Y |

See [Optimization](/docs/v0.9.0/optimizationmd) for suggestions on how to choose top-k values.


## TF-IDF

### Description

TF-IDF is a commonly used baseline for information retrieval that exploits two key intuitions:


* documents that have more lexical overlap with the query are more likely to be relevant

* words that occur in fewer documents are more significant than words that occur in many documents

Given a query, a tf-idf score is computed for each document as follows:

```python
score = tf * idf
```

Where:


* `tf` is how many times words in the query occur in that document.


* `idf` is the inverse of the fraction of documents containing the word.

In practice, both terms are usually log normalised.

### Initialisation

```python
from haystack.document_store import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
from haystack.pipeline import ExtractiveQAPipeline

document_store = InMemoryDocumentStore()
...
retriever = TfidfRetriever(document_store)
...
p = ExtractiveQAPipeline(reader, retriever)
```

## BM25 (Recommended)

### Description

BM25 is a variant of TF-IDF that we recommend you use if you are looking for a retrieval method that does not need a neural network for indexing.
It improves upon its predecessor in two main aspects:


* It saturates `tf` after a set number of occurrences of the given term in the document


* It normalises by document length so that short documents are favoured over long documents if they have the same amount of word overlap with the query

### Initialisation

```python
from haystack.document_store import ElasticsearchDocumentStore
from haystack.retriever import ElasticsearchRetriever
from haystack.pipeline import ExtractiveQAPipeline

document_store = ElasticsearchDocumentStore()
...
retriever = ElasticsearchRetriever(document_store)
...
p = ExtractiveQAPipeline(reader, retriever)
```

See [this](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) blog post for more details about the algorithm.

<!-- _comment: !! Diagram showing TFIDF vs BM25 !! -->
## Dense Passage Retrieval (Recommended)

### Description

[Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) is a highly performant retrieval method that calculates relevance using dense representations.
Key features:


* One BERT base model to encode documents


* One BERT base model to encode queries


* Ranking of documents done by dot product similarity between query and document embeddings

<!-- _comment: !! Diagram !! -->
Indexing using DPR is comparatively expensive in terms of required computation since all documents in the database need to be processed through the transformer.
The embeddings that are created in this step can be stored in FAISS, a database optimized for vector similarity.
DPR can also work with the ElasticsearchDocumentStore or the InMemoryDocumentStore.

There are two design decisions that have made DPR particularly performant.


* Separate encoders for document and query helps since queries are much shorter than documents


* Training with ‘In-batch negatives’ (gold labels are treated as negative examples for other samples in same batch) is highly efficient

In Haystack, you can simply download the pretrained encoders needed to start using DPR.
If you’d like to learn how to set up a DPR based system, have a look at the [tutorial](/docs/v0.9.0/tutorial6md)!

### Initialisation

<div class="recommendation">

**Tip**

When using DPR, it is recommended that you use the dot product similarity function since that is how it is trained.
To do so, simply provide `similarity="dot_product"` when initializing the DocumentStore 
as is done in the code example below.

</div>

```python
from haystack.document_store import FAISSDocumentStore
from haystack.retriever import DensePassageRetriever
from haystack.pipeline import ExtractiveQAPipeline

document_store = FAISSDocumentStore(similarity="dot_product")
...
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
)
...
finder = ExtractiveQAPipeline(reader, retriever)
```

<div class="recommendation">

**Training DPR:** Haystack supports training of your own DPR model! Check out the [tutorial](/docs/v0.9.0/tutorial9md) to see how this is done!

</div>

<!-- _comment: !! Training in future? !! -->
<!-- _comment: !! Talk more about benchmarks, SoTA, results !! -->
## Embedding Retrieval

### Description

In Haystack, you also have the option of using a single transformer model to encode document and query.
One style of model that is suited to this kind of retrieval is that of [Sentence Transformers](https://github.com/UKPLab/sentence-transformers).
These models are trained in Siamese Networks and use triplet loss such that they learn to embed similar sentences near to each other in a shared embedding space.

They are particular suited to cases where your query input is similar in style to that of the documents in your database
i.e. when you are searching for most similar documents.
This is not inherently suited to query based search where the length, language and format of the query usually significantly differs from the searched for text.

<div class="recommendation">

**Tip**

When using Sentence Transformer models, we recommend that you use a cosine similarity function. 
To do so, simply provide `similarity="cosine"` when initializing the DocumentStore 
as is done in the code example below.

</div>

### Initialisation

```python
from haystack.document_store import ElasticsearchDocumentStore
from haystack.retriever import EmbeddingRetriever
from haystack.pipeline import ExtractiveQAPipeline

document_store = ElasticsearchDocumentStore(similarity="cosine")
...
retriever = EmbeddingRetriever(document_store=document_store,
                               embedding_model="deepset/sentence_bert")
...
p = ExtractiveQAPipeline(reader, retriever)
```

## Deeper Dive: Dense vs Sparse

Broadly speaking, retrieval methods can be split into two categories: **dense** and **sparse**.

**Sparse** methods, like TF-IDF and BM25, operate by looking for shared keywords between the document and query.
They are:


* simple but effective


* don’t need to be trained


* work on any language

More recently, **dense** approaches such as Dense Passage Retrieval (DPR) have shown even better performance than their sparse counter parts.
These methods embed both document and query into a shared embedding space using deep neural networks
and the top candidates are the nearest neighbour documents to the query.
They are:


* powerful but computationally more expensive especially during indexing


* trained using labelled datasets


* language specific

### Qualitative Differences

Between these two types there are also some qualitative differences too.
For example, sparse methods treat text as a bag-of-words meaning that they **do not take word order and syntax into account**,
while the latest generation of dense methods use transformer based encoders
which are designed to be **sensitive** to these factors.

Also dense methods are very capable of building strong semantic representations of text,
but they **struggle when encountering out-of-vocabulary** words such as new names.
By contrast, sparse methods don’t need to learn representations of words,
they only care about whether they are present or absent in the text.
As such, **they handle out-of-vocabulary words with no problem**.

<!-- _comment: !! Show example from DPR paper? !! -->
### Indexing

Dense methods perform indexing by processing all the documents through a neural network and storing the resulting vectors.
This is a much more expensive operation than the creation of the inverted-index in sparse methods
and will require significant computational power and time.

<!-- _comment: !!See their individual sections (!! link !!) for more details on this point. Benchmarks too !! -->
### Terminology

<!-- _comment: !! Diagram of what a sparse vector looks like vs dense vector. !! -->
<!-- _comment: !! This section should be turned into something more like a side note !! -->
The terms **dense** and **sparse** refer to the representations that the algorithms build for each document and query.
**Sparse** methods characterise texts using vectors with one dimension corresponding to each word in the vocabulary.
Dimensions will be zero if the word is absent and non-zero if it is present.
Since most documents contain only a small subset of the full vocabulary,
these vectors are considered sparse since non-zero values are few and far between.

**Dense** methods, by contrast, pass text as input into neural network encoders
and represent text in a vector of a manually defined size (usually 768).
Though individual dimensions are not mapped to any corresponding vocabulary or linguistic feature,
each dimension encodes some information about the text.
There are rarely 0s in these vectors hence their relative density.
