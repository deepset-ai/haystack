---
title: "IBM Db2"
id: integrations-ibm-db
description: "IBM Db2 integration for Haystack"
slug: "/integrations-ibm-db"
---


## haystack_integrations.components.retrievers.ibm_db.embedding_retriever

### IBMDb2EmbeddingRetriever

Retrieves documents from a IBMDb2DocumentStore using vector similarity.

Use inside a Haystack pipeline after a text embedder:

```python
pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
pipeline.add_component("retriever", IBMDb2EmbeddingRetriever(
    document_store=store, top_k=5
))
pipeline.connect("embedder.embedding", "retriever.query_embedding")
```

#### __init__

```python
__init__(
    *,
    document_store: IBMDb2DocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Initialize the IBMDb2EmbeddingRetriever.

**Parameters:**

- **document_store** (<code>IBMDb2DocumentStore</code>) – An instance of `IBMDb2DocumentStore`.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents.
- **top_k** (<code>int</code>) – Maximum number of Documents to return.
- **filter_policy** (<code>FilterPolicy</code>) – Policy to determine how filters are applied.

**Raises:**

- <code>TypeError</code> – If `document_store` is not an instance of `IBMDb2DocumentStore`.

#### run

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents by vector similarity.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Dense float vector from an embedder component.
- **filters** (<code>dict\[str, Any\] | None</code>) – Runtime filters, merged with constructor filters according to filter_policy.
- **top_k** (<code>int | None</code>) – Override the constructor top_k for this call.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with key `documents` containing a list of matching :class:`Document` objects.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> IBMDb2EmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>IBMDb2EmbeddingRetriever</code> – Deserialized component.

## haystack_integrations.document_stores.ibm_db.document_store

IBM Db2 Document Store for Haystack.

### IBMDb2DocumentStore

IBM Db2 Document Store for Haystack using vector search capabilities.

This document store uses IBM Db2's native vector search functionality
to store and retrieve documents with embeddings.

#### __init__

```python
__init__(
    *,
    database: str,
    hostname: str,
    username: Secret = Secret.from_env_var("DB2_USERNAME"),
    password: Secret = Secret.from_env_var("DB2_PASSWORD"),
    port: int = 50000,
    protocol: str = "TCPIP",
    schema: str | None = None,
    use_ssl: bool = False,
    ssl_certificate: str | None = None,
    connection_options: dict[str, Any] | None = None,
    table_name: str = "haystack_documents",
    embedding_dim: int = 768,
    distance_metric: Literal["EUCLIDEAN", "COSINE", "MANHATTAN"] = "COSINE",
    recreate_table: bool = False
)
```

Initialize the IBM Db2 Document Store.

**Parameters:**

- **database** (<code>str</code>) – Database name
- **hostname** (<code>str</code>) – Database server hostname
- **username** (<code>Secret</code>) – Database username as a `Secret`, e.g. `Secret.from_env_var("DB2_USERNAME")`.
- **password** (<code>Secret</code>) – Database password as a `Secret`, e.g. `Secret.from_env_var("DB2_PASSWORD")`.
- **port** (<code>int</code>) – Database server port (default: 50000)
- **protocol** (<code>str</code>) – Connection protocol (default: "TCPIP")
- **schema** (<code>str | None</code>) – Database schema (optional)
- **use_ssl** (<code>bool</code>) – Enable SSL/TLS connection (default: False)
- **ssl_certificate** (<code>str | None</code>) – Path to SSL certificate file (optional, required if use_ssl is True)
- **connection_options** (<code>dict\[str, Any\] | None</code>) – Additional connection options as dict (optional)
- **table_name** (<code>str</code>) – Name of the table to store documents (default: "haystack_documents")
- **embedding_dim** (<code>int</code>) – Dimension of embedding vectors (default: 768)
- **distance_metric** (<code>Literal['EUCLIDEAN', 'COSINE', 'MANHATTAN']</code>) – Distance metric for similarity search (default: "COSINE")
- **recreate_table** (<code>bool</code>) – If True, drop and recreate the table (default: False)

#### count_documents

```python
count_documents() -> int
```

Count all documents in the store.

**Returns:**

- <code>int</code> – Number of documents

#### count_documents_by_filter

```python
count_documents_by_filter(filters: dict[str, Any] | None = None) -> int
```

Count documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Filters to apply. See Haystack documentation for filter syntax.

**Returns:**

- <code>int</code> – Number of documents matching the filters

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Write documents to the store.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of documents to write
- **policy** (<code>DuplicatePolicy</code>) – Policy for handling duplicate documents

**Returns:**

- <code>int</code> – Number of documents written

**Raises:**

- <code>ValueError</code> – If documents is not a list of Document objects or has invalid embeddings
- <code>TypeError</code> – If embeddings have invalid types
- <code>DuplicateDocumentError</code> – If a document with the same id already exists and policy is FAIL or NONE

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Filter documents using SQL-based metadata and field conditions.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Optional filter dictionary to constrain the returned documents.

**Returns:**

- <code>list\[Document\]</code> – List of matching documents.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Delete documents by their IDs.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – List of document IDs to delete

#### delete_by_filter

```python
delete_by_filter(filters: dict[str, Any] | None = None) -> int
```

Delete documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Filters to apply. See Haystack documentation for filter syntax.

**Returns:**

- <code>int</code> – Number of documents deleted

#### delete_all_documents

```python
delete_all_documents(recreate_index: bool = False) -> int
```

Delete all documents from the document store.

**Parameters:**

- **recreate_index** (<code>bool</code>) – If True, recreate the table after deletion

**Returns:**

- <code>int</code> – Number of documents deleted

#### update_by_filter

```python
update_by_filter(
    filters: dict[str, Any] | None = None, meta: dict[str, Any] | None = None
) -> int
```

Update documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Filters to apply. See Haystack documentation for filter syntax.
- **meta** (<code>dict\[str, Any\] | None</code>) – Dictionary of metadata fields to update

**Returns:**

- <code>int</code> – Number of documents updated

#### get_metadata_field_unique_values

```python
get_metadata_field_unique_values(field: str) -> list[Any]
```

Get all unique values for a given metadata field.

**Parameters:**

- **field** (<code>str</code>) – The metadata field name (can include 'meta.' prefix)

**Returns:**

- <code>list\[Any\]</code> – List of unique values for the field

#### get_metadata_field_min_max

```python
get_metadata_field_min_max(field: str) -> dict[str, Any]
```

Get the minimum and maximum values for a numeric metadata field.

**Parameters:**

- **field** (<code>str</code>) – The metadata field name (can include 'meta.' prefix)

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with 'min' and 'max' keys

#### get_metadata_fields_info

```python
get_metadata_fields_info() -> dict[str, dict[str, Any]]
```

Get information about all metadata fields including their types.

**Returns:**

- <code>dict\[str, dict\[str, Any\]\]</code> – Dictionary mapping field names to their type information

#### count_unique_metadata_by_filter

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any] | None = None,
    metadata_fields: list[str] | None = None,
) -> dict[str, int]
```

Count unique values for specified metadata fields, optionally filtered.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Optional filters to apply before counting
- **metadata_fields** (<code>list\[str\] | None</code>) – List of metadata field names to count unique values for

**Returns:**

- <code>dict\[str, int\]</code> – Dictionary mapping field names to their unique value counts

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the document store to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary representation

#### from_dict

```python
from_dict(data: dict[str, Any]) -> IBMDb2DocumentStore
```

Deserialize the document store from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary representation

**Returns:**

- <code>IBMDb2DocumentStore</code> – IBMDb2DocumentStore instance
