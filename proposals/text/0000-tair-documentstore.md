- Title: (Addition of `Tair document store`)
- Decision driver: (Zhidong Tan)
- Start Date: (2023-07-13)
- Proposal PR: (fill in after opening the PR)
- Github Issue or Discussion: (https://github.com/deepset-ai/haystack/issues/5348)

# Summary

Add a `Tair document store` that makes use of [Tair](https://github.com/alibaba/tair), which is an enterprise-level in-memory database developed by Alibaba.
Functions for documents and labels such as insertion, counting, filtering, TopK query and deletion will be implemented.

# Basic example

Here we shows how to use `Tair document store` to store and query documents and labels.
```python
from haystack.document_stores.tairvector import TairDocumentStore

index = "haystack_tests"
url = os.environ.get("TAIR_VECTOR")

# Initial a Tair document store
def ds(self):
    return TairDocumentStore(
        url=self.url,
        embedding_dim=768,
        embedding_field="embedding",
        index=self.index,
        similarity="COSINE",
        index_type="HNSW",
        recreate_index=True,
    )

# write documents
ds.write_documents(documents, index)

# search documents by filter
search_result = ds.get_all_documents(filters={"year": {"$in": ["2020", "2021"]}})

# query documents by embedding
query_result = ds.query_by_embedding(query_emb=query_emb, filters=None, top_k=3, index=index)

# count documents by filters
ds.get_document_count(filters={"month": ["02"]})

# delete documents by ids and filters
ds.delete_documents(ids=ids, filters={"name": ["file_5.txt"]})

# write labels
ds.write_labels(labels)

# delete labels by filter and ids
ds.delete_labels(ids=[labels[0].id], filters={"query": "query_9"})
```

# Motivation

Users may want to store documents in a pure-memory vector database, using high-performance real-time document storage and
document query based on vector searching.

Tair is an enterprise-level in-memory database developed by Alibaba, providing out-of-the-box cloud-native database
services. Tair supports online expansion and contraction, including horizontal and vertical expansion. In addition, high
availability of active and standby systems is achieved, and the failover delay is short.

Tairvector is an extended data structure self-developed by Tair, which provides real-time high-performance vector database
services in pure memory mode. In terms of the HNSW real-time online method, the reading and writing time is shorter than
that of Milvus. It supports combinations of various algorithms and data types, and supports Cosine, IP, L2 and Jaccard distance functions.

Therefore, we would like to implement a document store which makes use of [Tair](https://github.com/alibaba/tair) in a similar way to the Milvus document store already implemented.

# Detailed design

We define class TairDocumentStore by inheriting from class BaseDocumentStore.
```python
class TairDocumentStore(BaseDocumentStore):
    top_k_limit = 10_000
    top_k_limit_vectors = 1_000

    def __init__(
        self,
        url: str,
        tair_index: Optional["tair.TairVectorIndex"] = None,
        embedding_dim: int = 768,
        return_embedding: bool = False,
        index: str = "document",
        similarity: str = "COSINE",
        index_type: str = "HNSW",
        data_type: str = "FLOAT32",
        embedding_field: str = "embedding",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        recreate_index: bool = False,
        **kwargs: Any,
    ):
        ...
```
The framework of our code is similar to pinecone and milvus, with several different parameters as follows.
```python
"""
:param url: Tair vector database url corresponding to a instance(https://www.alibabacloud.com/help/en/tair/latest/tairvector).
:param tair_index: tair-client Index object, an index will be initialized or loaded if not specified.
:param similarity: The similarity function used to compare document vectors. `"COSINE"` is the default
    and is recommended if you are using a Sentence-Transformer model. `"IP"` is more performant
    with DPR embeddings.
    In both cases, the returned values in Document.score are normalized to be in range [0,1]:
        - For `"IP"`: `expit(np.asarray(raw_score / 100))`
        - For `"COSINE"`: `(raw_score + 1) / 2`
:param index_type: The type of indexing algorithms. Valid values: "HNSW" creates graph-based vector indexes,
    "FLAT" uses the Flat Search algorithm to search for vectors without creating indexes.
:param data_type: the data type of the vector. Valid values: "FLOAT32", "FLOAT16", "BINARY"
"""
```
We use some python interfaces of [TairVector](https://www.alibabacloud.com/help/en/tair/latest/tairvector) to call services of Tair.
```python
from tair import Tair as TairClient

# connect to tair from url
client = Tair.from_url(url, **kwargs)

# create an index
client.tvs_create_index(name=index, dim=embedding_dim, distance_type=distance_type,
                        index_type=index_type, data_type=data_type, **kwargs)

# get the infomation of an index
index_connection = self.client.tvs_get_index(index)

# add/update a document/label to index
client.tvs_hset(index=index, key=id, vector=embedding, is_binary=False, **(meta))

# get the vector value and attributes for a document/label
client.tvs_hgetall(index, id)

# search for the top k approximate nearest neighbors of vector in an index
client.tvs_knnsearch(index=index, k=top_k,
                     vector=query_emb, filter_str=tair_syntax_filter)

# delete documents/labels by ids
client.tvs_del(index, ids)
```

# Drawbacks

- It would add another document store that needs to be maintained and documented.
- Users may need to consider more when choosing a document store.
- The cost of implementation is not small because the code size may be over 1000 lines.

# Alternatives

We considered implementing class TairDocumentStore by inheriting the child class of BaseDocumentStore
(eg: KeywordDocumentStore) instead of BaseDocumentStore. Doing so can reduce the code size, but it will
increase the limitation of code implementation.

# Adoption strategy

It doesn't introduce a breaking change and wouldn't require changes in existing pipelines.

# How we teach this

- It would be good to have this be part of the Guide (perhaps under File DocumentStore).
- Could also be mentioned in one of the tutorials.

# Unresolved questions

- Should metadata of documents be stored as a json or just several attribute keys?
It would be convient to reconstruct the document by storing as a json,
while convenient for filtering by storing as keys.
- Should we use embeddings generated by models or just use random vectors
when testing the function of query by embedding?
