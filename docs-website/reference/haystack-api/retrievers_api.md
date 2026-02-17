---
title: "Retrievers"
id: retrievers-api
description: "Sweeps through a Document Store and returns a set of candidate Documents that are relevant to the query."
slug: "/retrievers-api"
---


## `auto_merging_retriever`

### `AutoMergingRetriever`

A retriever which returns parent documents of the matched leaf nodes documents, based on a threshold setting.

The AutoMergingRetriever assumes you have a hierarchical tree structure of documents, where the leaf nodes
are indexed in a document store. See the HierarchicalDocumentSplitter for more information on how to create
such a structure. During retrieval, if the number of matched leaf documents below the same parent is
higher than a defined threshold, the retriever will return the parent document instead of the individual leaf
documents.

The rational is, given that a paragraph is split into multiple chunks represented as leaf documents, and if for
a given query, multiple chunks are matched, the whole paragraph might be more informative than the individual
chunks alone.

Currently the AutoMergingRetriever can only be used by the following DocumentStores:

- [AstraDB](https://haystack.deepset.ai/integrations/astradb)
- [ElasticSearch](https://haystack.deepset.ai/docs/latest/documentstore/elasticsearch)
- [OpenSearch](https://haystack.deepset.ai/docs/latest/documentstore/opensearch)
- [PGVector](https://haystack.deepset.ai/docs/latest/documentstore/pgvector)
- [Qdrant](https://haystack.deepset.ai/docs/latest/documentstore/qdrant)

```python
from haystack import Document
from haystack.components.preprocessors import HierarchicalDocumentSplitter
from haystack.components.retrievers.auto_merging_retriever import AutoMergingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

# create a hierarchical document structure with 3 levels, where the parent document has 3 children
text = "The sun rose early in the morning. It cast a warm glow over the trees. Birds began to sing."
original_document = Document(content=text)
builder = HierarchicalDocumentSplitter(block_sizes={10, 3}, split_overlap=0, split_by="word")
docs = builder.run([original_document])["documents"]

# store level-1 parent documents and initialize the retriever
doc_store_parents = InMemoryDocumentStore()
for doc in docs:
    if doc.meta["__children_ids"] and doc.meta["__level"] in [0,1]:  # store the root document and level 1 documents
        doc_store_parents.write_documents([doc])

retriever = AutoMergingRetriever(doc_store_parents, threshold=0.5)

# assume we retrieved 2 leaf docs from the same parent, the parent document should be returned,
# since it has 3 children and the threshold=0.5, and we retrieved 2 children (2/3 > 0.66(6))
leaf_docs = [doc for doc in docs if not doc.meta["__children_ids"]]
retrieved_docs = retriever.run(leaf_docs[4:6])
print(retrieved_docs["documents"])
# [Document(id=538..),
# content: 'warm glow over the trees. Birds began to sing.',
# meta: {'block_size': 10, 'parent_id': '835..', 'children_ids': ['c17...', '3ff...', '352...'], 'level': 1, 'source_id': '835...',
# 'page_number': 1, 'split_id': 1, 'split_idx_start': 45})]}
```

#### `__init__`

```python
__init__(document_store: DocumentStore, threshold: float = 0.5)
```

Initialize the AutoMergingRetriever.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – DocumentStore from which to retrieve the parent documents
- **threshold** (<code>float</code>) – Threshold to decide whether the parent instead of the individual documents is returned

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> AutoMergingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary with serialized data.

**Returns:**

- <code>AutoMergingRetriever</code> – An instance of the component.

#### `run`

```python
run(documents: list[Document])
```

Run the AutoMergingRetriever.

Recursively groups documents by their parents and merges them if they meet the threshold,
continuing up the hierarchy until no more merges are possible.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of leaf documents that were matched by a retriever

**Returns:**

- – List of documents (could be a mix of different hierarchy levels)

#### `run_async`

```python
run_async(documents: list[Document])
```

Asynchronously run the AutoMergingRetriever.

Recursively groups documents by their parents and merges them if they meet the threshold,
continuing up the hierarchy until no more merges are possible.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of leaf documents that were matched by a retriever

**Returns:**

- – List of documents (could be a mix of different hierarchy levels)

## `filter_retriever`

### `FilterRetriever`

Retrieves documents that match the provided filters.

### Usage example

```python
from haystack import Document
from haystack.components.retrievers import FilterRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

docs = [
    Document(content="Python is a popular programming language", meta={"lang": "en"}),
    Document(content="python ist eine beliebte Programmiersprache", meta={"lang": "de"}),
]

doc_store = InMemoryDocumentStore()
doc_store.write_documents(docs)
retriever = FilterRetriever(doc_store, filters={"field": "lang", "operator": "==", "value": "en"})

# if passed in the run method, filters override those provided at initialization
result = retriever.run(filters={"field": "lang", "operator": "==", "value": "de"})

print(result["documents"])
```

#### `__init__`

```python
__init__(document_store: DocumentStore, filters: dict[str, Any] | None = None)
```

Create the FilterRetriever component.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – An instance of a Document Store to use with the Retriever.
- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the search space.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> FilterRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>FilterRetriever</code> – The deserialized component.

#### `run`

```python
run(filters: dict[str, Any] | None = None)
```

Run the FilterRetriever on the given input data.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the search space.
  If not specified, the FilterRetriever uses the values provided at initialization.

**Returns:**

- – A list of retrieved documents.

#### `run_async`

```python
run_async(filters: dict[str, Any] | None = None)
```

Asynchronously run the FilterRetriever on the given input data.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the search space.
  If not specified, the FilterRetriever uses the values provided at initialization.

**Returns:**

- – A list of retrieved documents.

## `in_memory/bm25_retriever`

### `InMemoryBM25Retriever`

Retrieves documents that are most similar to the query using keyword-based algorithm.

Use this retriever with the InMemoryDocumentStore.

### Usage example

```python
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

docs = [
    Document(content="Python is a popular programming language"),
    Document(content="python ist eine beliebte Programmiersprache"),
]

doc_store = InMemoryDocumentStore()
doc_store.write_documents(docs)
retriever = InMemoryBM25Retriever(doc_store)

result = retriever.run(query="Programmiersprache")

print(result["documents"])
```

#### `__init__`

```python
__init__(
    document_store: InMemoryDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    scale_score: bool = False,
    filter_policy: FilterPolicy = FilterPolicy.REPLACE,
)
```

Create the InMemoryBM25Retriever component.

**Parameters:**

- **document_store** (<code>InMemoryDocumentStore</code>) – An instance of InMemoryDocumentStore where the retriever should search for relevant documents.
- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the retriever's search space in the document store.
- **top_k** (<code>int</code>) – The maximum number of documents to retrieve.
- **scale_score** (<code>bool</code>) – When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
  When `False`, uses raw similarity scores.
- **filter_policy** (<code>FilterPolicy</code>) – The filter policy to apply during retrieval.
  Filter policy determines how filters are applied when retrieving documents. You can choose:
- `REPLACE` (default): Overrides the initialization filters with the filters specified at runtime.
  Use this policy to dynamically change filtering for specific queries.
- `MERGE`: Combines runtime filters with initialization filters to narrow down the search.

**Raises:**

- <code>ValueError</code> – If the specified `top_k` is not > 0.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> InMemoryBM25Retriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>InMemoryBM25Retriever</code> – The deserialized component.

#### `run`

```python
run(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    scale_score: bool | None = None,
) -> dict[str, list[Document]]
```

Run the InMemoryBM25Retriever on the given input data.

**Parameters:**

- **query** (<code>str</code>) – The query string for the Retriever.
- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the search space when retrieving documents.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.
- **scale_score** (<code>bool | None</code>) – When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
  When `False`, uses raw similarity scores.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – The retrieved documents.

**Raises:**

- <code>ValueError</code> – If the specified DocumentStore is not found or is not a InMemoryDocumentStore instance.

#### `run_async`

```python
run_async(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    scale_score: bool | None = None,
) -> dict[str, list[Document]]
```

Run the InMemoryBM25Retriever on the given input data.

**Parameters:**

- **query** (<code>str</code>) – The query string for the Retriever.
- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the search space when retrieving documents.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.
- **scale_score** (<code>bool | None</code>) – When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
  When `False`, uses raw similarity scores.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – The retrieved documents.

**Raises:**

- <code>ValueError</code> – If the specified DocumentStore is not found or is not a InMemoryDocumentStore instance.

## `in_memory/embedding_retriever`

### `InMemoryEmbeddingRetriever`

Retrieves documents that are most semantically similar to the query.

Use this retriever with the InMemoryDocumentStore.

When using this retriever, make sure it has query and document embeddings available.
In indexing pipelines, use a DocumentEmbedder to embed documents.
In query pipelines, use a TextEmbedder to embed queries and send them to the retriever.

### Usage example

```python
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

docs = [
    Document(content="Python is a popular programming language"),
    Document(content="python ist eine beliebte Programmiersprache"),
]
doc_embedder = SentenceTransformersDocumentEmbedder()
doc_embedder.warm_up()
docs_with_embeddings = doc_embedder.run(docs)["documents"]

doc_store = InMemoryDocumentStore()
doc_store.write_documents(docs_with_embeddings)
retriever = InMemoryEmbeddingRetriever(doc_store)

query="Programmiersprache"
text_embedder = SentenceTransformersTextEmbedder()
text_embedder.warm_up()
query_embedding = text_embedder.run(query)["embedding"]

result = retriever.run(query_embedding=query_embedding)

print(result["documents"])
```

#### `__init__`

```python
__init__(
    document_store: InMemoryDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    scale_score: bool = False,
    return_embedding: bool = False,
    filter_policy: FilterPolicy = FilterPolicy.REPLACE,
)
```

Create the InMemoryEmbeddingRetriever component.

**Parameters:**

- **document_store** (<code>InMemoryDocumentStore</code>) – An instance of InMemoryDocumentStore where the retriever should search for relevant documents.
- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the retriever's search space in the document store.
- **top_k** (<code>int</code>) – The maximum number of documents to retrieve.
- **scale_score** (<code>bool</code>) – When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
  When `False`, uses raw similarity scores.
- **return_embedding** (<code>bool</code>) – When `True`, returns the embedding of the retrieved documents.
  When `False`, returns just the documents, without their embeddings.
- **filter_policy** (<code>FilterPolicy</code>) – The filter policy to apply during retrieval.
  Filter policy determines how filters are applied when retrieving documents. You can choose:
- `REPLACE` (default): Overrides the initialization filters with the filters specified at runtime.
  Use this policy to dynamically change filtering for specific queries.
- `MERGE`: Combines runtime filters with initialization filters to narrow down the search.

**Raises:**

- <code>ValueError</code> – If the specified top_k is not > 0.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> InMemoryEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>InMemoryEmbeddingRetriever</code> – The deserialized component.

#### `run`

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    scale_score: bool | None = None,
    return_embedding: bool | None = None,
) -> dict[str, list[Document]]
```

Run the InMemoryEmbeddingRetriever on the given input data.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the search space when retrieving documents.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.
- **scale_score** (<code>bool | None</code>) – When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
  When `False`, uses raw similarity scores.
- **return_embedding** (<code>bool | None</code>) – When `True`, returns the embedding of the retrieved documents.
  When `False`, returns just the documents, without their embeddings.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – The retrieved documents.

**Raises:**

- <code>ValueError</code> – If the specified DocumentStore is not found or is not an InMemoryDocumentStore instance.

#### `run_async`

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    scale_score: bool | None = None,
    return_embedding: bool | None = None,
) -> dict[str, list[Document]]
```

Run the InMemoryEmbeddingRetriever on the given input data.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the search space when retrieving documents.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return.
- **scale_score** (<code>bool | None</code>) – When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
  When `False`, uses raw similarity scores.
- **return_embedding** (<code>bool | None</code>) – When `True`, returns the embedding of the retrieved documents.
  When `False`, returns just the documents, without their embeddings.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – The retrieved documents.

**Raises:**

- <code>ValueError</code> – If the specified DocumentStore is not found or is not an InMemoryDocumentStore instance.

## `multi_query_embedding_retriever`

### `MultiQueryEmbeddingRetriever`

A component that retrieves documents using multiple queries in parallel with an embedding-based retriever.

This component takes a list of text queries, converts them to embeddings using a query embedder,
and then uses an embedding-based retriever to find relevant documents for each query in parallel.
The results are combined and sorted by relevance score.

### Usage example

```python
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers import MultiQueryEmbeddingRetriever

documents = [
    Document(content="Renewable energy is energy that is collected from renewable resources."),
    Document(content="Solar energy is a type of green energy that is harnessed from the sun."),
    Document(content="Wind energy is another type of green energy that is generated by wind turbines."),
    Document(content="Geothermal energy is heat that comes from the sub-surface of the earth."),
    Document(content="Biomass energy is produced from organic materials, such as plant and animal waste."),
    Document(content="Fossil fuels, such as coal, oil, and natural gas, are non-renewable energy sources."),
]

# Populate the document store
doc_store = InMemoryDocumentStore()
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()
doc_writer = DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.SKIP)
documents = doc_embedder.run(documents)["documents"]
doc_writer.run(documents=documents)

# Run the multi-query retriever
in_memory_retriever = InMemoryEmbeddingRetriever(document_store=doc_store, top_k=1)
query_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

multi_query_retriever = MultiQueryEmbeddingRetriever(
    retriever=in_memory_retriever,
    query_embedder=query_embedder,
    max_workers=3
)

queries = ["Geothermal energy", "natural gas", "turbines"]
result = multi_query_retriever.run(queries=queries)
for doc in result["documents"]:
    print(f"Content: {doc.content}, Score: {doc.score}")
# >> Content: Geothermal energy is heat that comes from the sub-surface of the earth., Score: 0.8509603046266574
# >> Content: Renewable energy is energy that is collected from renewable resources., Score: 0.42763211298893034
# >> Content: Solar energy is a type of green energy that is harnessed from the sun., Score: 0.40077417016494354
# >> Content: Fossil fuels, such as coal, oil, and natural gas, are non-renewable energy sources., Score: 0.3774863680
# >> Content: Wind energy is another type of green energy that is generated by wind turbines., Score: 0.30914239725622
# >> Content: Biomass energy is produced from organic materials, such as plant and animal waste., Score: 0.25173074243
```

#### `__init__`

```python
__init__(
    *,
    retriever: EmbeddingRetriever,
    query_embedder: TextEmbedder,
    max_workers: int = 3
) -> None
```

Initialize MultiQueryEmbeddingRetriever.

**Parameters:**

- **retriever** (<code>EmbeddingRetriever</code>) – The embedding-based retriever to use for document retrieval.
- **query_embedder** (<code>TextEmbedder</code>) – The query embedder to convert text queries to embeddings.
- **max_workers** (<code>int</code>) – Maximum number of worker threads for parallel processing.

#### `warm_up`

```python
warm_up() -> None
```

Warm up the query embedder and the retriever if any has a warm_up method.

#### `run`

```python
run(
    queries: list[str], retriever_kwargs: dict[str, Any] | None = None
) -> dict[str, list[Document]]
```

Retrieve documents using multiple queries in parallel.

**Parameters:**

- **queries** (<code>list\[str\]</code>) – List of text queries to process.
- **retriever_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional dictionary of arguments to pass to the retriever's run method.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary containing:
  - `documents`: List of retrieved documents sorted by relevance score.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary representing the serialized component.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> MultiQueryEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>MultiQueryEmbeddingRetriever</code> – The deserialized component.

## `multi_query_text_retriever`

### `MultiQueryTextRetriever`

A component that retrieves documents using multiple queries in parallel with a text-based retriever.

This component takes a list of text queries and uses a text-based retriever to find relevant documents for each
query in parallel, using a thread pool to manage concurrent execution. The results are combined and sorted by
relevance score.

You can use this component in combination with QueryExpander component to enhance the retrieval process.

### Usage example

```python
from haystack import Document
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.query import QueryExpander
from haystack.components.retrievers.multi_query_text_retriever import MultiQueryTextRetriever

documents = [
    Document(content="Renewable energy is energy that is collected from renewable resources."),
    Document(content="Solar energy is a type of green energy that is harnessed from the sun."),
    Document(content="Wind energy is another type of green energy that is generated by wind turbines."),
    Document(content="Hydropower is a form of renewable energy using the flow of water to generate electricity."),
    Document(content="Geothermal energy is heat that comes from the sub-surface of the earth.")
]

document_store = InMemoryDocumentStore()
doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
doc_writer.run(documents=documents)

in_memory_retriever = InMemoryBM25Retriever(document_store=document_store, top_k=1)
multiquery_retriever = MultiQueryTextRetriever(retriever=in_memory_retriever)
results = multiquery_retriever.run(queries=["renewable energy?", "Geothermal", "Hydropower"])
for doc in results["documents"]:
    print(f"Content: {doc.content}, Score: {doc.score}")
# >>
# >> Content: Geothermal energy is heat that comes from the sub-surface of the earth., Score: 1.6474448833731097
# >> Content: Hydropower is a form of renewable energy using the flow of water to generate electricity., Score: 1.615
# >> Content: Renewable energy is energy that is collected from renewable resources., Score: 1.5255309812344944
```

#### `__init__`

```python
__init__(*, retriever: TextRetriever, max_workers: int = 3) -> None
```

Initialize MultiQueryTextRetriever.

**Parameters:**

- **retriever** (<code>TextRetriever</code>) – The text-based retriever to use for document retrieval.
- **max_workers** (<code>int</code>) – Maximum number of worker threads for parallel processing. Default is 3.

#### `warm_up`

```python
warm_up() -> None
```

Warm up the retriever if it has a warm_up method.

#### `run`

```python
run(
    queries: list[str], retriever_kwargs: dict[str, Any] | None = None
) -> dict[str, list[Document]]
```

Retrieve documents using multiple queries in parallel.

**Parameters:**

- **queries** (<code>list\[str\]</code>) – List of text queries to process.
- **retriever_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional dictionary of arguments to pass to the retriever's run method.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary containing:
  `documents`: List of retrieved documents sorted by relevance score.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> MultiQueryTextRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>MultiQueryTextRetriever</code> – The deserialized component.

## `sentence_window_retriever`

### `SentenceWindowRetriever`

Retrieves neighboring documents from a DocumentStore to provide context for query results.

This component is intended to be used after a Retriever (e.g., BM25Retriever, EmbeddingRetriever).
It enhances retrieved results by fetching adjacent document chunks to give
additional context for the user.

The documents must include metadata indicating their origin and position:

- `source_id` is used to group sentence chunks belonging to the same original document.
- `split_id` represents the position/order of the chunk within the document.

The number of adjacent documents to include on each side of the retrieved document can be configured using the
`window_size` parameter. You can also specify which metadata fields to use for source and split ID
via `source_id_meta_field` and `split_id_meta_field`.

The SentenceWindowRetriever is compatible with the following DocumentStores:

- [Astra](https://docs.haystack.deepset.ai/docs/astradocumentstore)
- [Elasticsearch](https://docs.haystack.deepset.ai/docs/elasticsearch-document-store)
- [OpenSearch](https://docs.haystack.deepset.ai/docs/opensearch-document-store)
- [Pgvector](https://docs.haystack.deepset.ai/docs/pgvectordocumentstore)
- [Pinecone](https://docs.haystack.deepset.ai/docs/pinecone-document-store)
- [Qdrant](https://docs.haystack.deepset.ai/docs/qdrant-document-store)

### Usage example

```python
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.retrievers import SentenceWindowRetriever
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore

splitter = DocumentSplitter(split_length=10, split_overlap=5, split_by="word")
text = (
        "This is a text with some words. There is a second sentence. And there is also a third sentence. "
        "It also contains a fourth sentence. And a fifth sentence. And a sixth sentence. And a seventh sentence"
)
doc = Document(content=text)
docs = splitter.run([doc])
doc_store = InMemoryDocumentStore()
doc_store.write_documents(docs["documents"])


rag = Pipeline()
rag.add_component("bm25_retriever", InMemoryBM25Retriever(doc_store, top_k=1))
rag.add_component("sentence_window_retriever", SentenceWindowRetriever(document_store=doc_store, window_size=2))
rag.connect("bm25_retriever", "sentence_window_retriever")

rag.run({'bm25_retriever': {"query":"third"}})

# >> {'sentence_window_retriever': {'context_windows': ['some words. There is a second sentence.
# >> And there is also a third sentence. It also contains a fourth sentence. And a fifth sentence. And a sixth
# >> sentence. And a'], 'context_documents': [[Document(id=..., content: 'some words. There is a second sentence.
# >> And there is ', meta: {'source_id': '...', 'page_number': 1, 'split_id': 1, 'split_idx_start': 20,
# >> '_split_overlap': [{'doc_id': '...', 'range': (20, 43)}, {'doc_id': '...', 'range': (0, 30)}]}),
# >> Document(id=..., content: 'second sentence. And there is also a third sentence. It ',
# >> meta: {'source_id': '74ea87deb38012873cf8c07e...f19d01a26a098447113e1d7b83efd30c02987114', 'page_number': 1,
# >> 'split_id': 2, 'split_idx_start': 43, '_split_overlap': [{'doc_id': '...', 'range': (23, 53)}, {'doc_id': '.',
# >> 'range': (0, 26)}]}), Document(id=..., content: 'also a third sentence. It also contains a fourth sentence. ',
# >> meta: {'source_id': '...', 'page_number': 1, 'split_id': 3, 'split_idx_start': 73, '_split_overlap':
# >> [{'doc_id': '...', 'range': (30, 56)}, {'doc_id': '...', 'range': (0, 33)}]}), Document(id=..., content:
# >> 'also contains a fourth sentence. And a fifth sentence. And ', meta: {'source_id': '...', 'page_number': 1,
# >> 'split_id': 4, 'split_idx_start': 99, '_split_overlap': [{'doc_id': '...', 'range': (26, 59)},
# >> {'doc_id': '...', 'range': (0, 26)}]}), Document(id=..., content: 'And a fifth sentence. And a sixth sentence.
# >> And a ', meta: {'source_id': '...', 'page_number': 1, 'split_id': 5, 'split_idx_start': 132,
# >> '_split_overlap': [{'doc_id': '...', 'range': (33, 59)}, {'doc_id': '...', 'range': (0, 24)}]})]]}}}}
```

#### `__init__`

```python
__init__(
    document_store: DocumentStore,
    window_size: int = 3,
    *,
    source_id_meta_field: str | list[str] = "source_id",
    split_id_meta_field: str = "split_id",
    raise_on_missing_meta_fields: bool = True
)
```

Creates a new SentenceWindowRetriever component.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – The Document Store to retrieve the surrounding documents from.
- **window_size** (<code>int</code>) – The number of documents to retrieve before and after the relevant one.
  For example, `window_size: 2` fetches 2 preceding and 2 following documents.
- **source_id_meta_field** (<code>str | list\[str\]</code>) – The metadata field that contains the source ID of the document.
  This can be a single field or a list of fields. If multiple fields are provided, the retriever will
  consider the document as part of the same source if all the fields match.
- **split_id_meta_field** (<code>str</code>) – The metadata field that contains the split ID of the document.
- **raise_on_missing_meta_fields** (<code>bool</code>) – If True, raises an error if the documents do not contain the required
  metadata fields. If False, it will skip retrieving the context for documents that are missing
  the required metadata fields, but will still include the original document in the results.

#### `merge_documents_text`

```python
merge_documents_text(documents: list[Document]) -> str
```

Merge a list of document text into a single string.

This functions concatenates the textual content of a list of documents into a single string, eliminating any
overlapping content.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Documents to merge.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> SentenceWindowRetriever
```

Deserializes the component from a dictionary.

**Returns:**

- <code>SentenceWindowRetriever</code> – Deserialized component.

#### `run`

```python
run(retrieved_documents: list[Document], window_size: int | None = None)
```

Based on the `source_id` and on the `doc.meta['split_id']` get surrounding documents from the document store.

Implements the logic behind the sentence-window technique, retrieving the surrounding documents of a given
document from the document store.

**Parameters:**

- **retrieved_documents** (<code>list\[Document\]</code>) – List of retrieved documents from the previous retriever.
- **window_size** (<code>int | None</code>) – The number of documents to retrieve before and after the relevant one. This will overwrite
  the `window_size` parameter set in the constructor.

**Returns:**

- – A dictionary with the following keys:
  - `context_windows`: A list of strings, where each string represents the concatenated text from the
    context window of the corresponding document in `retrieved_documents`.
  - `context_documents`: A list `Document` objects, containing the retrieved documents plus the context
    document surrounding them. The documents are sorted by the `split_idx_start`
    meta field.

#### `run_async`

```python
run_async(retrieved_documents: list[Document], window_size: int | None = None)
```

Based on the `source_id` and on the `doc.meta['split_id']` get surrounding documents from the document store.

Implements the logic behind the sentence-window technique, retrieving the surrounding documents of a given
document from the document store.

**Parameters:**

- **retrieved_documents** (<code>list\[Document\]</code>) – List of retrieved documents from the previous retriever.
- **window_size** (<code>int | None</code>) – The number of documents to retrieve before and after the relevant one. This will overwrite
  the `window_size` parameter set in the constructor.

**Returns:**

- – A dictionary with the following keys:
  - `context_windows`: A list of strings, where each string represents the concatenated text from the
    context window of the corresponding document in `retrieved_documents`.
  - `context_documents`: A list `Document` objects, containing the retrieved documents plus the context
    document surrounding them. The documents are sorted by the `split_idx_start`
    meta field.
