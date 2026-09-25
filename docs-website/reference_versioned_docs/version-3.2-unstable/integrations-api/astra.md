---
title: "Astra"
id: integrations-astra
description: "Astra integration for Haystack"
slug: "/integrations-astra"
---


## haystack_integrations.components.retrievers.astra.retriever

### AstraEmbeddingRetriever

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

#### __init__

```python
__init__(
    document_store: AstraDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
) -> None
```

Initialize the AstraEmbeddingRetriever.

**Parameters:**

- **document_store** (<code>AstraDocumentStore</code>) – An instance of AstraDocumentStore.
- **filters** (<code>dict\[str, Any\] | None</code>) – a dictionary with filters to narrow down the search space.
- **top_k** (<code>int</code>) – the maximum number of documents to retrieve.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

#### close

```python
close() -> None
```

Release the synchronous resources of the underlying Document Store.

#### close_async

```python
close_async() -> None
```

Release the asynchronous resources of the underlying Document Store.

#### run

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

#### run_async

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents from the AstraDocumentStore asynchronously.

Uses the native async Astra DB API with a reusable connection. Call `close_async()` (or
`Pipeline.close_async()`) when finished, before closing the event loop.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – floats representing the query embedding
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – the maximum number of documents to retrieve.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – a dictionary with the following keys:
- `documents`: A list of documents retrieved from the AstraDocumentStore.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AstraEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AstraEmbeddingRetriever</code> – Deserialized component.

## haystack_integrations.document_stores.astra.document_store

### AstraDocumentStore

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

#### __init__

```python
__init__(
    api_endpoint: Secret = Secret.from_env_var("ASTRA_DB_API_ENDPOINT"),
    token: Secret = Secret.from_env_var("ASTRA_DB_APPLICATION_TOKEN"),
    collection_name: str = "documents",
    embedding_dimension: int = 768,
    duplicates_policy: DuplicatePolicy = DuplicatePolicy.NONE,
    similarity: str = "cosine",
    namespace: str | None = None,
) -> None
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
- **similarity** (<code>str</code>) – Similarity metric for new collections: `cosine`, `dot_product`, or `euclidean`.
  Existing collections retain their configured metric.
- **namespace** (<code>str | None</code>) – The keyspace containing the collection, or the SDK default when omitted.

**Raises:**

- <code>ValueError</code> – if the API endpoint or token is not set.

#### close

```python
close() -> None
```

Drop the cached synchronous collection without deleting documents.

AstraPy 2 exposes no way to release synchronous connections, so this only discards the collection; the next
synchronous operation creates a new one.

#### close_async

```python
close_async() -> None
```

Release the cached async collection connection without deleting documents.

Call this on the same event loop as the async methods, after they have finished and before
closing the loop. Repeated calls are safe; a later call opens a new connection. Calls on a new
event loop open a new connection automatically.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AstraDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AstraDocumentStore</code> – Deserialized component.

**Raises:**

- <code>ValueError</code> – The serialized `duplicates_policy` is not a valid policy name.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### write_documents

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

#### write_documents_async

```python
write_documents_async(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Asynchronously indexes documents for later queries.

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

#### count_documents

```python
count_documents() -> int
```

Counts the number of documents in the document store.

**Returns:**

- <code>int</code> – the number of documents in the document store.

#### count_documents_async

```python
count_documents_async() -> int
```

Asynchronously counts the number of documents in the document store.

**Returns:**

- <code>int</code> – the number of documents in the document store.

#### filter_documents

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

#### filter_documents_async

```python
filter_documents_async(filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously returns at most 1000 documents that match the filter.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – filters to apply.

**Returns:**

- <code>list\[Document\]</code> – matching documents.

**Raises:**

- <code>AstraDocumentStoreFilterError</code> – if the filter is invalid or not supported by this class.

#### get_documents_by_id

```python
get_documents_by_id(ids: list[str]) -> list[Document]
```

Gets documents by their IDs.

**Parameters:**

- **ids** (<code>list\[str\]</code>) – the IDs of the documents to retrieve.

**Returns:**

- <code>list\[Document\]</code> – the matching documents.

#### get_documents_by_id_async

```python
get_documents_by_id_async(ids: list[str]) -> list[Document]
```

Asynchronously gets documents by their IDs.

**Parameters:**

- **ids** (<code>list\[str\]</code>) – the IDs of the documents to retrieve.

**Returns:**

- <code>list\[Document\]</code> – the matching documents.

#### get_document_by_id

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

#### get_document_by_id_async

```python
get_document_by_id_async(document_id: str) -> Document
```

Asynchronously gets a document by its ID.

**Parameters:**

- **document_id** (<code>str</code>) – the ID to filter by

**Returns:**

- <code>Document</code> – the found document

**Raises:**

- <code>MissingDocumentError</code> – if the document is not found

#### search

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

#### search_async

```python
search_async(
    query_embedding: list[float],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[Document]
```

Search using AstraPy's native async API.

The collection connection is initialized lazily and reused across searches on the same event loop.
Call `close_async()` when finished, including after failures or cancellation, before closing the loop.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – A list of query embeddings.
- **top_k** (<code>int</code>) – The number of results to return.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters to apply during search.

**Returns:**

- <code>list\[Document\]</code> – Matching documents, including embeddings, metadata and similarity scores.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes documents from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – IDs of the documents to delete.

**Raises:**

- <code>MissingDocumentError</code> – if no document was deleted but document IDs were provided.

#### delete_documents_async

```python
delete_documents_async(document_ids: list[str]) -> None
```

Asynchronously deletes documents from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – IDs of the documents to delete.

**Raises:**

- <code>MissingDocumentError</code> – if no document was deleted but document IDs were provided.

#### delete_all_documents

```python
delete_all_documents(*, recreate_index: bool = False) -> None
```

Deletes all documents from the document store.

**Parameters:**

- **recreate_index** (<code>bool</code>) – If `True`, drops the collection and recreates it with its current definition (vector
  dimension, metric and indexing settings) instead of deleting its documents. Dropping and creating a
  collection takes several seconds and is not atomic: if the creation fails, the next operation creates the
  collection from this store's settings.

**Raises:**

- <code>DocumentStoreError</code> – if the documents could not be deleted or the collection could not be recreated.

#### delete_all_documents_async

```python
delete_all_documents_async(*, recreate_index: bool = False) -> None
```

Asynchronously deletes all documents from the document store.

**Parameters:**

- **recreate_index** (<code>bool</code>) – If `True`, drops the collection and recreates it with its current definition (vector
  dimension, metric and indexing settings) instead of deleting its documents. Dropping and creating a
  collection takes several seconds and is not atomic: if the creation fails, the next operation creates the
  collection from this store's settings.

**Raises:**

- <code>DocumentStoreError</code> – if the documents could not be deleted or the collection could not be recreated.

#### delete_by_filter

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

#### delete_by_filter_async

```python
delete_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously deletes documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to find documents to delete.

**Returns:**

- <code>int</code> – The number of documents deleted.

**Raises:**

- <code>AstraDocumentStoreFilterError</code> – if the filter is invalid or not supported.

#### update_by_filter

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

#### update_by_filter_async

```python
update_by_filter_async(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Asynchronously updates documents that match the provided filters with the given metadata.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to find documents to update.
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update. This will be merged with existing metadata.

**Returns:**

- <code>int</code> – The number of documents updated.

**Raises:**

- <code>AstraDocumentStoreFilterError</code> – if the filter is invalid or not supported.

#### count_documents_by_filter

```python
count_documents_by_filter(filters: dict[str, Any]) -> int
```

Applies a filter and counts the documents that matched it.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to the document list.

**Returns:**

- <code>int</code> – The number of documents that match the filter.

#### count_documents_by_filter_async

```python
count_documents_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously applies a filter and counts the documents that matched it.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to the document list.

**Returns:**

- <code>int</code> – The number of documents that match the filter.

#### count_unique_metadata_by_filter

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Applies a filter selecting documents and counts the unique values for each meta field of the matched documents.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to the document list.
- **metadata_fields** (<code>list\[str\]</code>) – The metadata fields to count unique values for.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary where the keys are the metadata field names and the values are the count of unique
  values.

#### count_unique_metadata_by_filter_async

```python
count_unique_metadata_by_filter_async(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Asynchronously counts the unique values of each meta field across the documents matching a filter.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to the document list.
- **metadata_fields** (<code>list\[str\]</code>) – The metadata fields to count unique values for.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary where the keys are the metadata field names and the values are the count of unique
  values.

#### get_metadata_fields_info

```python
get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Returns the metadata fields and the corresponding types.

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – A dictionary mapping field names to dictionaries with a `type` key.

#### get_metadata_fields_info_async

```python
get_metadata_fields_info_async() -> dict[str, dict[str, str]]
```

Asynchronously returns the metadata fields and the corresponding types.

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – A dictionary mapping field names to dictionaries with a `type` key.

#### get_metadata_field_min_max

```python
get_metadata_field_min_max(metadata_field: str) -> dict[str, Any]
```

For a given metadata field, find its max and min value.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to inspect.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with `min` and `max`.

#### get_metadata_field_min_max_async

```python
get_metadata_field_min_max_async(metadata_field: str) -> dict[str, Any]
```

Asynchronously, for a given metadata field, find its max and min value.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to inspect.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with `min` and `max`.

#### get_metadata_field_unique_values

```python
get_metadata_field_unique_values(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10,
    filters: dict[str, Any] | None = None,
) -> tuple[list[Any], int]
```

Retrieves unique values for a field matching a search term or all possible values if no search term is given.

**Note**: values of different types are kept distinct even when they compare equal in Python
(e.g. the int `1`, the bool `True` and the str `"1"` are returned as three separate values), with
one exception.
AstraDB's Data API canonicalizes any whole-number float (e.g. `1.0`) to an int on storage, unconditionally so a
whole-number float is always returned back as an int, never as a float.
Example: 1.0 (float) is sent to storage and comes back a 1 (int)

Exception are floats with a fractional part (e.g. `1.5`) are unaffected and round-trip normally.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to inspect.
- **search_term** (<code>str | None</code>) – Optional case-insensitive substring search term.
- **from\_** (<code>int</code>) – The starting index for pagination.
- **size** (<code>int</code>) – The number of values to return.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional filters to restrict the documents considered.

**Returns:**

- <code>tuple\[list\[Any\], int\]</code> – A tuple containing the paginated values (in their original type) and the total count.

#### get_metadata_field_unique_values_async

```python
get_metadata_field_unique_values_async(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10,
    filters: dict[str, Any] | None = None,
) -> tuple[list[Any], int]
```

Asynchronously retrieves unique values for a field, optionally matching a search term.

**Note**: values of different types are kept distinct even when they compare equal in Python
(e.g. the int `1`, the bool `True` and the str `"1"` are returned as three separate values), with
one exception.
AstraDB's Data API canonicalizes any whole-number float (e.g. `1.0`) to an int on storage, unconditionally so a
whole-number float is always returned back as an int, never as a float.
Example: 1.0 (float) is sent to storage and comes back a 1 (int)

Exception are floats with a fractional part (e.g. `1.5`) are unaffected and round-trip normally.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to inspect.
- **search_term** (<code>str | None</code>) – Optional case-insensitive substring search term.
- **from\_** (<code>int</code>) – The starting index for pagination.
- **size** (<code>int</code>) – The number of values to return.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional filters to restrict the documents considered.

**Returns:**

- <code>tuple\[list\[Any\], int\]</code> – A tuple containing the paginated values (in their original type) and the total count.

## haystack_integrations.document_stores.astra.errors

### AstraDocumentStoreError

Bases: <code>DocumentStoreError</code>

Parent class for all AstraDocumentStore errors.

### AstraDocumentStoreFilterError

Bases: <code>FilterError</code>

Raised when an invalid filter is passed to AstraDocumentStore.

### AstraDocumentStoreConfigError

Bases: <code>AstraDocumentStoreError</code>

Raised when an invalid configuration is passed to AstraDocumentStore.
