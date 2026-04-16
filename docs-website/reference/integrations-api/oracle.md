---
title: "Oracle AI Vector Search"
id: integrations-oracle
description: "Oracle AI Vector Search integration for Haystack"
slug: "/integrations-oracle"
---


## haystack_integrations.components.retrievers.oracle.embedding_retriever

### OracleEmbeddingRetriever

Retrieves documents from an OracleDocumentStore using vector similarity.

Use inside a Haystack pipeline after a text embedder::

```
pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
pipeline.add_component("retriever", OracleEmbeddingRetriever(
    document_store=store, top_k=5
))
pipeline.connect("embedder.embedding", "retriever.query_embedding")
```

#### run

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents by vector similarity.

Args:
query_embedding: Dense float vector from an embedder component.
filters: Runtime filters, merged with constructor filters according to filter_policy.
top_k: Override the constructor top_k for this call.

Returns:
`{"documents": [Document, ...]}`

#### run_async

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Async variant of :meth:`run`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OracleEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OracleEmbeddingRetriever</code> – Deserialized component.

## haystack_integrations.document_stores.oracle.document_store

### OracleConnectionConfig

Connection parameters for Oracle Database.

Supports both thin (direct TCP) and thick (wallet / ADB-S) modes.
Thin mode requires no Oracle Instant Client; thick mode is activated
automatically when *wallet_location* is provided.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OracleConnectionConfig
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OracleConnectionConfig</code> – Deserialized component.

### OracleDocumentStore

Haystack DocumentStore backed by Oracle AI Vector Search.

Requires Oracle Database 23ai or later (for VECTOR data type and
IF NOT EXISTS DDL support).

Usage::

```
from haystack.utils import Secret
from haystack_integrations.document_stores.oracle import (
    OracleDocumentStore, OracleConnectionConfig,
)

store = OracleDocumentStore(
    connection_config=OracleConnectionConfig(
        user=Secret.from_env_var("ORACLE_USER"),
        password=Secret.from_env_var("ORACLE_PASSWORD"),
        dsn=Secret.from_env_var("ORACLE_DSN"),
    ),
    embedding_dim=1536,
)
```

#### create_hnsw_index

```python
create_hnsw_index() -> None
```

Create an HNSW vector index on the embedding column.

Safe to call multiple times — uses IF NOT EXISTS.

#### create_hnsw_index_async

```python
create_hnsw_index_async() -> None
```

Async variant of create_hnsw_index.

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Writes documents to the document store.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to write to the document store.
- **policy** (<code>DuplicatePolicy</code>) – The duplicate policy to use when writing documents.

**Returns:**

- <code>int</code> – The number of documents written to the document store.

**Raises:**

- <code>DuplicateDocumentError</code> – If a document with the same id already exists in the document store
  and the policy is set to `DuplicatePolicy.FAIL` or `DuplicatePolicy.NONE`.

#### write_documents_async

```python
write_documents_async(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Asynchronously writes documents to the document store.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to write to the document store.
- **policy** (<code>DuplicatePolicy</code>) – The duplicate policy to use when writing documents.

**Returns:**

- <code>int</code> – The number of documents written to the document store.

**Raises:**

- <code>DuplicateDocumentError</code> – If a document with the same id already exists in the document store
  and the policy is set to `DuplicatePolicy.FAIL` or `DuplicatePolicy.NONE`.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – The filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

#### filter_documents_async

```python
filter_documents_async(filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – The filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes documents that match the provided `document_ids` from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete

#### delete_documents_async

```python
delete_documents_async(document_ids: list[str]) -> None
```

Asynchronously deletes documents that match the provided `document_ids` from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete

#### count_documents

```python
count_documents() -> int
```

Returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – Number of documents in the document store.

#### count_documents_async

```python
count_documents_async() -> int
```

Asynchronously returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – Number of documents in the document store.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OracleDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OracleDocumentStore</code> – Deserialized component.
