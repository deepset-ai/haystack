---
title: "Astra"
id: integrations-astra
description: "Astra integration for Haystack"
slug: "/integrations-astra"
---


## `haystack_integrations.components.retrievers.astra.retriever`

### `AstraEmbeddingRetriever`

A component for retrieving documents from an AstraDocumentStore.

Usage example:

```python
from haystack_integrations.document_stores.astra import AstraDocumentStore
from haystack_integrations.components.retrievers.astra import AstraEmbeddingRetriever

document_store = AstraDocumentStore(
    api_endpoint=api_endpoint,
    token=token,
    collection_name=collection_name,
    duplicates_policy=DuplicatePolicy.SKIP,
    embedding_dim=384,
)

retriever = AstraEmbeddingRetriever(document_store=document_store)
```

#### `__init__`

```python
__init__(
    document_store: AstraDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
)
```

**Parameters:**

- **document_store** (<code>AstraDocumentStore</code>) – An instance of AstraDocumentStore.
- **filters** (<code>dict\[str, Any\] | None</code>) – a dictionary with filters to narrow down the search space.
- **top_k** (<code>int</code>) – the maximum number of documents to retrieve.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

#### `run`

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents from the AstraDocumentStore.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – floats representing the query embedding
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – the maximum number of documents to retrieve.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – a dictionary with the following keys:
- `documents`: A list of documents retrieved from the AstraDocumentStore.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> AstraEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AstraEmbeddingRetriever</code> – Deserialized component.

## `haystack_integrations.document_stores.astra.document_store`

### `AstraDocumentStore`

An AstraDocumentStore document store for Haystack.

Example Usage:

```python
from haystack_integrations.document_stores.astra import AstraDocumentStore

document_store = AstraDocumentStore(
    api_endpoint=api_endpoint,
    token=token,
    collection_name=collection_name,
    duplicates_policy=DuplicatePolicy.SKIP,
    embedding_dim=384,
)
```

#### `__init__`

```python
__init__(
    api_endpoint: Secret = Secret.from_env_var("ASTRA_DB_API_ENDPOINT"),
    token: Secret = Secret.from_env_var("ASTRA_DB_APPLICATION_TOKEN"),
    collection_name: str = "documents",
    embedding_dimension: int = 768,
    duplicates_policy: DuplicatePolicy = DuplicatePolicy.NONE,
    similarity: str = "cosine",
    namespace: str | None = None,
)
```

The connection to Astra DB is established and managed through the JSON API.
The required credentials (api endpoint and application token) can be generated
through the UI by clicking and the connect tab, and then selecting JSON API and
Generate Configuration.

**Parameters:**

- **api_endpoint** (<code>Secret</code>) – the Astra DB API endpoint.
- **token** (<code>Secret</code>) – the Astra DB application token.
- **collection_name** (<code>str</code>) – the current collection in the keyspace in the current Astra DB.
- **embedding_dimension** (<code>int</code>) – dimension of embedding vector.
- **duplicates_policy** (<code>DuplicatePolicy</code>) – handle duplicate documents based on DuplicatePolicy parameter options.
  Parameter options : (`SKIP`, `OVERWRITE`, `FAIL`, `NONE`)
- `DuplicatePolicy.NONE`: Default policy, If a Document with the same ID already exists,
  it is skipped and not written.
- `DuplicatePolicy.SKIP`: if a Document with the same ID already exists, it is skipped and not written.
- `DuplicatePolicy.OVERWRITE`: if a Document with the same ID already exists, it is overwritten.
- `DuplicatePolicy.FAIL`: if a Document with the same ID already exists, an error is raised.
- **similarity** (<code>str</code>) – the similarity function used to compare document vectors.

**Raises:**

- <code>ValueError</code> – if the API endpoint or token is not set.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> AstraDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AstraDocumentStore</code> – Deserialized component.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `write_documents`

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Indexes documents for later queries.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – a list of Haystack Document objects.
- **policy** (<code>DuplicatePolicy</code>) – handle duplicate documents based on DuplicatePolicy parameter options.
  Parameter options : (`SKIP`, `OVERWRITE`, `FAIL`, `NONE`)
- `DuplicatePolicy.NONE`: Default policy, If a Document with the same ID already exists,
  it is skipped and not written.
- `DuplicatePolicy.SKIP`: If a Document with the same ID already exists,
  it is skipped and not written.
- `DuplicatePolicy.OVERWRITE`: If a Document with the same ID already exists, it is overwritten.
- `DuplicatePolicy.FAIL`: If a Document with the same ID already exists, an error is raised.

**Returns:**

- <code>int</code> – number of documents written.

**Raises:**

- <code>ValueError</code> – if the documents are not of type Document or dict.
- <code>DuplicateDocumentError</code> – if a document with the same ID already exists and policy is set to FAIL.
- <code>Exception</code> – if the document ID is not a string or if `id` and `_id` are both present in the document.

#### `count_documents`

```python
count_documents() -> int
```

Counts the number of documents in the document store.

**Returns:**

- <code>int</code> – the number of documents in the document store.

#### `filter_documents`

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns at most 1000 documents that match the filter.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – filters to apply.

**Returns:**

- <code>list\[Document\]</code> – matching documents.

**Raises:**

- <code>AstraDocumentStoreFilterError</code> – if the filter is invalid or not supported by this class.

#### `get_documents_by_id`

```python
get_documents_by_id(ids: list[str]) -> list[Document]
```

Gets documents by their IDs.

**Parameters:**

- **ids** (<code>list\[str\]</code>) – the IDs of the documents to retrieve.

**Returns:**

- <code>list\[Document\]</code> – the matching documents.

#### `get_document_by_id`

```python
get_document_by_id(document_id: str) -> Document
```

Gets a document by its ID.

**Parameters:**

- **document_id** (<code>str</code>) – the ID to filter by

**Returns:**

- <code>Document</code> – the found document

**Raises:**

- <code>MissingDocumentError</code> – if the document is not found

#### `search`

```python
search(
    query_embedding: list[float],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[Document]
```

Perform a search for a list of queries.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – a list of query embeddings.
- **top_k** (<code>int</code>) – the number of results to return.
- **filters** (<code>dict\[str, Any\] | None</code>) – filters to apply during search.

**Returns:**

- <code>list\[Document\]</code> – matching documents.

#### `delete_documents`

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes documents from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – IDs of the documents to delete.

**Raises:**

- <code>MissingDocumentError</code> – if no document was deleted but document IDs were provided.

#### `delete_all_documents`

```python
delete_all_documents() -> None
```

Deletes all documents from the document store.

#### `delete_by_filter`

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to find documents to delete.

**Returns:**

- <code>int</code> – The number of documents deleted.

**Raises:**

- <code>AstraDocumentStoreFilterError</code> – if the filter is invalid or not supported.

#### `update_by_filter`

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Updates documents that match the provided filters with the given metadata.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to find documents to update.
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update. This will be merged with existing metadata.

**Returns:**

- <code>int</code> – The number of documents updated.

**Raises:**

- <code>AstraDocumentStoreFilterError</code> – if the filter is invalid or not supported.

## `haystack_integrations.document_stores.astra.errors`

### `AstraDocumentStoreError`

Bases: <code>DocumentStoreError</code>

Parent class for all AstraDocumentStore errors.

### `AstraDocumentStoreFilterError`

Bases: <code>FilterError</code>

Raised when an invalid filter is passed to AstraDocumentStore.

### `AstraDocumentStoreConfigError`

Bases: <code>AstraDocumentStoreError</code>

Raised when an invalid configuration is passed to AstraDocumentStore.
