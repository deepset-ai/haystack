---
title: "Valkey"
id: integrations-valkey
description: "Valkey integration for Haystack"
slug: "/integrations-valkey"
---

<a id="haystack_integrations.components.retrievers.valkey.embedding_retriever"></a>

## Module haystack\_integrations.components.retrievers.valkey.embedding\_retriever

<a id="haystack_integrations.components.retrievers.valkey.embedding_retriever.ValkeyEmbeddingRetriever"></a>

### ValkeyEmbeddingRetriever

A component for retrieving documents from a ValkeyDocumentStore using vector similarity search.

This retriever uses dense embeddings to find semantically similar documents. It supports
filtering by metadata fields and configurable similarity thresholds.

Key features:
- Vector similarity search using HNSW algorithm
- Metadata filtering with tag and numeric field support
- Configurable top-k results
- Filter policy management for runtime filter application

Usage example:
```python
from haystack.document_stores.types import DuplicatePolicy
from haystack import Document
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack_integrations.components.retrievers.valkey import ValkeyEmbeddingRetriever
from haystack_integrations.document_stores.valkey import ValkeyDocumentStore

document_store = ValkeyDocumentStore(index_name="my_index", embedding_dim=768)

documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(content="Elephants have been observed to behave in a way that indicates..."),
    Document(content="In certain places, you can witness the phenomenon of bioluminescent waves."),
]

document_embedder = SentenceTransformersDocumentEmbedder()
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)

document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
query_pipeline.add_component("retriever", ValkeyEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "How many languages are there?"

res = query_pipeline.run({"text_embedder": {"text": query}})
assert res["retriever"]["documents"][0].content == "There are over 7,000 languages spoken around the world today."
```

<a id="haystack_integrations.components.retrievers.valkey.embedding_retriever.ValkeyEmbeddingRetriever.__init__"></a>

#### ValkeyEmbeddingRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: ValkeyDocumentStore,
             filters: dict[str, Any] | None = None,
             top_k: int = 10,
             filter_policy: str | FilterPolicy = FilterPolicy.REPLACE)
```

**Arguments**:

- `document_store`: The Valkey Document Store.
- `filters`: Filters applied to the retrieved Documents.
- `top_k`: Maximum number of Documents to return.
- `filter_policy`: Policy to determine how filters are applied.

**Raises**:

- `ValueError`: If `document_store` is not an instance of `ValkeyDocumentStore`.

<a id="haystack_integrations.components.retrievers.valkey.embedding_retriever.ValkeyEmbeddingRetriever.to_dict"></a>

#### ValkeyEmbeddingRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.valkey.embedding_retriever.ValkeyEmbeddingRetriever.from_dict"></a>

#### ValkeyEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ValkeyEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.valkey.embedding_retriever.ValkeyEmbeddingRetriever.run"></a>

#### ValkeyEmbeddingRetriever.run

```python
@component.output_types(documents=list[Document])
def run(query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None) -> dict[str, list[Document]]
```

Retrieve documents from the `ValkeyDocumentStore`, based on their dense embeddings.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of `Document`s to return.

**Returns**:

List of Document similar to `query_embedding`.

<a id="haystack_integrations.components.retrievers.valkey.embedding_retriever.ValkeyEmbeddingRetriever.run_async"></a>

#### ValkeyEmbeddingRetriever.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(query_embedding: list[float],
                    filters: dict[str, Any] | None = None,
                    top_k: int | None = None) -> dict[str, list[Document]]
```

Asynchronously retrieve documents from the `ValkeyDocumentStore`, based on their dense embeddings.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of `Document`s to return.

**Returns**:

List of Document similar to `query_embedding`.

<a id="haystack_integrations.document_stores.valkey.document_store"></a>

## Module haystack\_integrations.document\_stores.valkey.document\_store

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore"></a>

### ValkeyDocumentStore

A document store implementation using Valkey with vector search capabilities.

This document store provides persistent storage for documents with embeddings and supports
vector similarity search using the Valkey Search module. It's designed for high-performance
retrieval applications requiring both semantic search and metadata filtering.

Key features:
- Vector similarity search with HNSW algorithm
- Metadata filtering on tag and numeric fields
- Configurable distance metrics (L2, cosine, inner product)
- Batch operations for efficient document management
- Both synchronous and asynchronous operations
- Cluster and standalone mode support

Supported filterable Document metadata fields:
- meta_category (TagField): exact string matches
- meta_status (TagField): status filtering
- meta_priority (NumericField): numeric comparisons
- meta_score (NumericField): score filtering
- meta_timestamp (NumericField): date/time filtering

Usage example:
```python
from haystack import Document
from haystack_integrations.document_stores.valkey import ValkeyDocumentStore

# Initialize document store
document_store = ValkeyDocumentStore(
    nodes_list=[("localhost", 6379)],
    index_name="my_documents",
    embedding_dim=768,
    distance_metric="cosine",
)

# Store documents with embeddings
documents = [
    Document(
        content="Valkey is a Redis-compatible database",
        embedding=[0.1, 0.2, ...],  # 768-dim vector
        meta={"category": "database", "priority": 1},
    ),
]
document_store.write_documents(documents)

# Search with filters
results = document_store._embedding_retrival(
    embedding=[0.1, 0.15, ...],
    filters={"field": "meta.category", "operator": "==", "value": "database"},
    limit=10,
)
```

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.__init__"></a>

#### ValkeyDocumentStore.\_\_init\_\_

```python
def __init__(nodes_list: list[tuple[str, int]] | None = None,
             *,
             cluster_mode: bool = False,
             use_tls: bool = False,
             username: Secret | None = Secret.from_env_var("VALKEY_USERNAME",
                                                           strict=False),
             password: Secret | None = Secret.from_env_var("VALKEY_PASSWORD",
                                                           strict=False),
             request_timeout: int = 500,
             retry_attempts: int = 3,
             retry_base_delay_ms: int = 1000,
             retry_exponent_base: int = 2,
             batch_size: int = 100,
             index_name: str = "default",
             distance_metric: Literal["l2", "cosine", "ip"] = "cosine",
             embedding_dim: int = 768,
             metadata_fields: dict[str, type[str] | type[int]] | None = None)
```

Creates a new ValkeyDocumentStore instance.

**Arguments**:

- `nodes_list`: List of (host, port) tuples for Valkey nodes. Defaults to [("localhost", 6379)].
- `cluster_mode`: Whether to connect in cluster mode. Defaults to False.
- `use_tls`: Whether to use TLS for connections. Defaults to False.
- `username`: Username for authentication. If not provided, reads from VALKEY_USERNAME environment variable.
Defaults to None.
- `password`: Password for authentication. If not provided, reads from VALKEY_PASSWORD environment variable.
Defaults to None.
- `request_timeout`: Request timeout in milliseconds. Defaults to 500.
- `retry_attempts`: Number of retry attempts for failed operations. Defaults to 3.
- `retry_base_delay_ms`: Base delay in milliseconds for exponential backoff. Defaults to 1000.
- `retry_exponent_base`: Exponent base for exponential backoff calculation. Defaults to 2.
- `batch_size`: Number of documents to process in a single batch for async operations. Defaults to 100.
- `index_name`: Name of the search index. Defaults to "haystack_document".
- `distance_metric`: Distance metric for vector similarity. Options: "l2", "cosine", "ip" (inner product).
Defaults to "cosine".
- `embedding_dim`: Dimension of document embeddings. Defaults to 768.
- `metadata_fields`: Dictionary mapping metadata field names to Python types for filtering.
Supported types: str (for exact matching), int (for numeric comparisons).
Example: `{"category": str, "priority": int}`.
If not provided, no metadata fields will be indexed for filtering.

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.to_dict"></a>

#### ValkeyDocumentStore.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes this store to a dictionary.

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.from_dict"></a>

#### ValkeyDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> ValkeyDocumentStore
```

Deserializes the store from a dictionary.

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.count_documents"></a>

#### ValkeyDocumentStore.count\_documents

```python
def count_documents() -> int
```

Return the number of documents stored in the document store.

This method queries the Valkey Search index to get the total count of indexed documents.
If the index doesn't exist, it returns 0.

**Raises**:

- `ValkeyDocumentStoreError`: If there's an error accessing the index or counting documents.
Example:
```python
document_store = ValkeyDocumentStore()
count = document_store.count_documents()
print(f"Total documents: {count}")
```

**Returns**:

The number of documents in the document store.

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.count_documents_async"></a>

#### ValkeyDocumentStore.count\_documents\_async

```python
async def count_documents_async() -> int
```

Asynchronously return the number of documents stored in the document store.

This method queries the Valkey Search index to get the total count of indexed documents.
If the index doesn't exist, it returns 0. This is the async version of count_documents().

**Raises**:

- `ValkeyDocumentStoreError`: If there's an error accessing the index or counting documents.
Example:
```python
document_store = ValkeyDocumentStore()
count = await document_store.count_documents_async()
print(f"Total documents: {count}")
```

**Returns**:

The number of documents in the document store.

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.filter_documents"></a>

#### ValkeyDocumentStore.filter\_documents

```python
def filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Filter documents by metadata without vector search.

This method retrieves documents based on metadata filters without performing vector similarity search.
Since Valkey Search requires vector queries, this method uses a dummy vector internally and removes
the similarity scores from results.

**Arguments**:

- `filters`: Optional metadata filters in Haystack format. Supports filtering on:
- meta.category (string equality)
- meta.status (string equality)
- meta.priority (numeric comparisons)
- meta.score (numeric comparisons)
- meta.timestamp (numeric comparisons)

**Raises**:

- `ValkeyDocumentStoreError`: If there's an error filtering documents.
Example:
```python
# Filter by category
docs = document_store.filter_documents(filters={"field": "meta.category", "operator": "==", "value": "news"})

# Filter by numeric range
docs = document_store.filter_documents(filters={"field": "meta.priority", "operator": ">=", "value": 5})
```

**Returns**:

List of documents matching the filters, with score set to None.

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.filter_documents_async"></a>

#### ValkeyDocumentStore.filter\_documents\_async

```python
async def filter_documents_async(
        filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously filter documents by metadata without vector search.

This is the async version of filter_documents(). It retrieves documents based on metadata filters
without performing vector similarity search. Since Valkey Search requires vector queries, this method
uses a dummy vector internally and removes the similarity scores from results.

**Arguments**:

- `filters`: Optional metadata filters in Haystack format. Supports filtering on:
- meta.category (string equality)
- meta.status (string equality)
- meta.priority (numeric comparisons)
- meta.score (numeric comparisons)
- meta.timestamp (numeric comparisons)

**Raises**:

- `ValkeyDocumentStoreError`: If there's an error filtering documents.
Example:
```python
# Filter by category
docs = await document_store.filter_documents_async(
    filters={"field": "meta.category", "operator": "==", "value": "news"},
)

# Filter by numeric range
docs = await document_store.filter_documents_async(filters={"field": "meta.priority", "operator": ">=", "value": 5})
```

**Returns**:

List of documents matching the filters, with score set to None.

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.write_documents"></a>

#### ValkeyDocumentStore.write\_documents

```python
def write_documents(documents: list[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Write documents to the document store.

This method stores documents with their embeddings and metadata in Valkey. The search index is
automatically created if it doesn't exist. Documents without embeddings will be assigned a
dummy vector for indexing purposes.

**Arguments**:

- `documents`: List of Document objects to store. Each document should have:
- content: The document text
- embedding: Vector representation (optional, dummy vector used if missing)
- meta: Optional metadata dict with supported fields (category, status, priority, score, timestamp)
- `policy`: How to handle duplicate documents. Only NONE and OVERWRITE are supported.
Defaults to DuplicatePolicy.NONE.

**Raises**:

- `ValkeyDocumentStoreError`: If there's an error writing documents.
- `ValueError`: If documents list contains invalid objects.
Example:
```python
documents = [
    Document(content="First document", embedding=[0.1, 0.2, 0.3], meta={"category": "news", "priority": 1}),
    Document(content="Second document", embedding=[0.4, 0.5, 0.6], meta={"category": "blog", "priority": 2}),
]
count = document_store.write_documents(documents)
print(f"Wrote {count} documents")
```

**Returns**:

Number of documents successfully written.

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.write_documents_async"></a>

#### ValkeyDocumentStore.write\_documents\_async

```python
async def write_documents_async(
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Asynchronously write documents to the document store.

This is the async version of write_documents(). It stores documents with their embeddings and
metadata in Valkey using batch processing for improved performance. The search index is
automatically created if it doesn't exist.

**Arguments**:

- `documents`: List of Document objects to store. Each document should have:
- content: The document text
- embedding: Vector representation (optional, dummy vector used if missing)
- meta: Optional metadata dict with supported fields (category, status, priority, score, timestamp)
- `policy`: How to handle duplicate documents. Only NONE and OVERWRITE are supported.
Defaults to DuplicatePolicy.NONE.

**Raises**:

- `ValkeyDocumentStoreError`: If there's an error writing documents.
- `ValueError`: If documents list contains invalid objects.
Example:
```python
documents = [
    Document(content="First document", embedding=[0.1, 0.2, 0.3], meta={"category": "news", "priority": 1}),
    Document(content="Second document", embedding=[0.4, 0.5, 0.6], meta={"category": "blog", "priority": 2}),
]
count = await document_store.write_documents_async(documents)
print(f"Wrote {count} documents")
```

**Returns**:

Number of documents successfully written.

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.delete_documents"></a>

#### ValkeyDocumentStore.delete\_documents

```python
def delete_documents(document_ids: list[str]) -> None
```

Delete documents from the document store by their IDs.

This method removes documents from both the Valkey database and the search index.
If some documents are not found, a warning is logged but the operation continues.

**Arguments**:

- `document_ids`: List of document IDs to delete. These should be the same IDs
used when the documents were originally stored.

**Raises**:

- `ValkeyDocumentStoreError`: If there's an error deleting documents.
Example:
```python
# Delete specific documents
document_store.delete_documents(["doc1", "doc2", "doc3"])

# Delete a single document
document_store.delete_documents(["single_doc_id"])
```

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.delete_documents_async"></a>

#### ValkeyDocumentStore.delete\_documents\_async

```python
async def delete_documents_async(document_ids: list[str]) -> None
```

Asynchronously delete documents from the document store by their IDs.

This is the async version of delete_documents(). It removes documents from both the Valkey
database and the search index. If some documents are not found, a warning is logged but
the operation continues.

**Arguments**:

- `document_ids`: List of document IDs to delete. These should be the same IDs
used when the documents were originally stored.

**Raises**:

- `ValkeyDocumentStoreError`: If there's an error deleting documents.
Example:
```python
# Delete specific documents
await document_store.delete_documents_async(["doc1", "doc2", "doc3"])

# Delete a single document
await document_store.delete_documents_async(["single_doc_id"])
```

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.delete_all_documents"></a>

#### ValkeyDocumentStore.delete\_all\_documents

```python
def delete_all_documents() -> None
```

Delete all documents from the document store.

This method removes all documents by dropping the entire search index. This is an efficient
way to clear all data but requires recreating the index for future operations. If the index
doesn't exist, the operation completes without error.

**Raises**:

- `ValkeyDocumentStoreError`: If there's an error dropping the index.
Warning:
    This operation is irreversible and will permanently delete all documents and the search index.

Example:
```python
# Clear all documents from the store
document_store.delete_all_documents()

# The index will be automatically recreated on next write operation
document_store.write_documents(new_documents)
```

<a id="haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore.delete_all_documents_async"></a>

#### ValkeyDocumentStore.delete\_all\_documents\_async

```python
async def delete_all_documents_async() -> None
```

Asynchronously delete all documents from the document store.

This is the async version of delete_all_documents(). It removes all documents by dropping
the entire search index. This is an efficient way to clear all data but requires recreating
the index for future operations. If the index doesn't exist, the operation completes without error.

**Raises**:

- `ValkeyDocumentStoreError`: If there's an error dropping the index.
Warning:
    This operation is irreversible and will permanently delete all documents and the search index.

Example:
```python
# Clear all documents from the store
await document_store.delete_all_documents_async()

# The index will be automatically recreated on next write operation
await document_store.write_documents_async(new_documents)
```
