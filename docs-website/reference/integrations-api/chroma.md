---
title: "Chroma"
id: integrations-chroma
description: "Chroma integration for Haystack"
slug: "/integrations-chroma"
---


## `haystack_integrations.components.retrievers.chroma.retriever`

### `ChromaQueryTextRetriever`

A component for retrieving documents from a [Chroma database](https://docs.trychroma.com/) using the `query` API.

Example usage:

```python
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever

file_paths = ...

# Chroma is used in-memory so we use the same instances in the two pipelines below
document_store = ChromaDocumentStore()

indexing = Pipeline()
indexing.add_component("converter", TextFileToDocument())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("converter", "writer")
indexing.run({"converter": {"sources": file_paths}})

querying = Pipeline()
querying.add_component("retriever", ChromaQueryTextRetriever(document_store))
results = querying.run({"retriever": {"query": "Variable declarations", "top_k": 3}})

for d in results["retriever"]["documents"]:
    print(d.meta, d.score)
```

#### `__init__`

```python
__init__(
    document_store: ChromaDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
)
```

**Parameters:**

- **document_store** (<code>ChromaDocumentStore</code>) – an instance of `ChromaDocumentStore`.
- **filters** (<code>dict\[str, Any\] | None</code>) – filters to narrow down the search space.
- **top_k** (<code>int</code>) – the maximum number of documents to retrieve.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

#### `run`

```python
run(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, Any]
```

Run the retriever on the given input data.

**Parameters:**

- **query** (<code>str</code>) – The input data for the retriever. In this case, a plain-text query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to retrieve.
  If not specified, the default value from the constructor is used.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

**Raises:**

- <code>ValueError</code> – If the specified document store is not found or is not a MemoryDocumentStore instance.

#### `run_async`

```python
run_async(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, Any]
```

Asynchronously run the retriever on the given input data.

Asynchronous methods are only supported for HTTP connections.

**Parameters:**

- **query** (<code>str</code>) – The input data for the retriever. In this case, a plain-text query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to retrieve.
  If not specified, the default value from the constructor is used.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

**Raises:**

- <code>ValueError</code> – If the specified document store is not found or is not a MemoryDocumentStore instance.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ChromaQueryTextRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ChromaQueryTextRetriever</code> – Deserialized component.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `ChromaEmbeddingRetriever`

A component for retrieving documents from a [Chroma database](https://docs.trychroma.com/) using embeddings.

#### `__init__`

```python
__init__(
    document_store: ChromaDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
)
```

**Parameters:**

- **document_store** (<code>ChromaDocumentStore</code>) – an instance of `ChromaDocumentStore`.
- **filters** (<code>dict\[str, Any\] | None</code>) – filters to narrow down the search space.
- **top_k** (<code>int</code>) – the maximum number of documents to retrieve.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

#### `run`

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]
```

Run the retriever on the given input data.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – the query embeddings.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – the maximum number of documents to retrieve.
  If not specified, the default value from the constructor is used.

**Returns:**

- <code>dict\[str, Any\]</code> – a dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

#### `run_async`

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]
```

Asynchronously run the retriever on the given input data.

Asynchronous methods are only supported for HTTP connections.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – the query embeddings.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – the maximum number of documents to retrieve.
  If not specified, the default value from the constructor is used.

**Returns:**

- <code>dict\[str, Any\]</code> – a dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ChromaEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ChromaEmbeddingRetriever</code> – Deserialized component.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

## `haystack_integrations.document_stores.chroma.document_store`

### `ChromaDocumentStore`

A document store using [Chroma](https://docs.trychroma.com/) as the backend.

We use the `collection.get` API to implement the document store protocol,
the `collection.search` API will be used in the retriever instead.

#### `__init__`

```python
__init__(
    collection_name: str = "documents",
    embedding_function: str = "default",
    persist_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
    distance_function: Literal["l2", "cosine", "ip"] = "l2",
    metadata: dict | None = None,
    client_settings: dict[str, Any] | None = None,
    **embedding_function_params: Any
)
```

Creates a new ChromaDocumentStore instance.
It is meant to be connected to a Chroma collection.

Note: for the component to be part of a serializable pipeline, the __init__
parameters must be serializable, reason why we use a registry to configure the
embedding function passing a string.

**Parameters:**

- **collection_name** (<code>str</code>) – the name of the collection to use in the database.
- **embedding_function** (<code>str</code>) – the name of the embedding function to use to embed the query
- **persist_path** (<code>str | None</code>) – Path for local persistent storage. Cannot be used in combination with `host` and `port`.
  If none of `persist_path`, `host`, and `port` is specified, the database will be `in-memory`.
- **host** (<code>str | None</code>) – The host address for the remote Chroma HTTP client connection. Cannot be used with `persist_path`.
- **port** (<code>int | None</code>) – The port number for the remote Chroma HTTP client connection. Cannot be used with `persist_path`.
- **distance_function** (<code>Literal['l2', 'cosine', 'ip']</code>) – The distance metric for the embedding space.
- `"l2"` computes the Euclidean (straight-line) distance between vectors,
  where smaller scores indicate more similarity.
- `"cosine"` computes the cosine similarity between vectors,
  with higher scores indicating greater similarity.
- `"ip"` stands for inner product, where higher scores indicate greater similarity between vectors.
  **Note**: `distance_function` can only be set during the creation of a collection.
  To change the distance metric of an existing collection, consider cloning the collection.
- **metadata** (<code>dict | None</code>) – a dictionary of chromadb collection parameters passed directly to chromadb's client
  method `create_collection`. If it contains the key `"hnsw:space"`, the value will take precedence over the
  `distance_function` parameter above.
- **client_settings** (<code>dict\[str, Any\] | None</code>) – a dictionary of Chroma Settings configuration options passed to
  `chromadb.config.Settings`. These settings configure the underlying Chroma client behavior.
  For available options, see [Chroma's config.py](https://github.com/chroma-core/chroma/blob/main/chromadb/config.py).
  **Note**: specifying these settings may interfere with standard client initialization parameters.
  This option is intended for advanced customization.
- **embedding_function_params** (<code>Any</code>) – additional parameters to pass to the embedding function.

#### `count_documents`

```python
count_documents() -> int
```

Returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – how many documents are present in the document store.

#### `count_documents_async`

```python
count_documents_async() -> int
```

Asynchronously returns how many documents are present in the document store.

Asynchronous methods are only supported for HTTP connections.

**Returns:**

- <code>int</code> – how many documents are present in the document store.

#### `filter_documents`

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – the filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – a list of Documents that match the given filters.

#### `filter_documents_async`

```python
filter_documents_async(filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously returns the documents that match the filters provided.

Asynchronous methods are only supported for HTTP connections.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – the filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – a list of Documents that match the given filters.

#### `write_documents`

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL
) -> int
```

Writes (or overwrites) documents into the store.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to write into the document store.
- **policy** (<code>DuplicatePolicy</code>) – Not supported at the moment.

**Returns:**

- <code>int</code> – The number of documents written

**Raises:**

- <code>ValueError</code> – When input is not valid.

#### `write_documents_async`

```python
write_documents_async(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL
) -> int
```

Asynchronously writes (or overwrites) documents into the store.

Asynchronous methods are only supported for HTTP connections.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to write into the document store.
- **policy** (<code>DuplicatePolicy</code>) – Not supported at the moment.

**Returns:**

- <code>int</code> – The number of documents written

**Raises:**

- <code>ValueError</code> – When input is not valid.

#### `delete_documents`

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes all documents with a matching document_ids from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete

#### `delete_documents_async`

```python
delete_documents_async(document_ids: list[str]) -> None
```

Asynchronously deletes all documents with a matching document_ids from the document store.

Asynchronous methods are only supported for HTTP connections.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete

#### `delete_by_filter`

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for deletion.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns:**

- <code>int</code> – The number of documents deleted.

#### `delete_by_filter_async`

```python
delete_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously deletes all documents that match the provided filters.

Asynchronous methods are only supported for HTTP connections.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for deletion.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns:**

- <code>int</code> – The number of documents deleted.

#### `update_by_filter`

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Updates the metadata of all documents that match the provided filters.

**Note**: This operation is not atomic. Documents matching the filter are fetched first,
then updated. If documents are modified between the fetch and update operations,
those changes may be lost.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update. This will be merged with existing metadata.

**Returns:**

- <code>int</code> – The number of documents updated.

#### `update_by_filter_async`

```python
update_by_filter_async(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Asynchronously updates the metadata of all documents that match the provided filters.

Asynchronous methods are only supported for HTTP connections.

**Note**: This operation is not atomic. Documents matching the filter are fetched first,
then updated. If documents are modified between the fetch and update operations,
those changes may be lost.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update. This will be merged with existing metadata.

**Returns:**

- <code>int</code> – The number of documents updated.

#### `delete_all_documents`

```python
delete_all_documents(*, recreate_index: bool = False) -> None
```

Deletes all documents in the document store.

A fast way to clear all documents from the document store while preserving any collection settings and mappings.

**Parameters:**

- **recreate_index** (<code>bool</code>) – Whether to recreate the index after deleting all documents.

#### `delete_all_documents_async`

```python
delete_all_documents_async(*, recreate_index: bool = False) -> None
```

Asynchronously deletes all documents in the document store.

A fast way to clear all documents from the document store while preserving any collection settings and mappings.

**Parameters:**

- **recreate_index** (<code>bool</code>) – Whether to recreate the index after deleting all documents.

#### `search`

```python
search(
    queries: list[str], top_k: int, filters: dict[str, Any] | None = None
) -> list[list[Document]]
```

Search the documents in the store using the provided text queries.

**Parameters:**

- **queries** (<code>list\[str\]</code>) – the list of queries to search for.
- **top_k** (<code>int</code>) – top_k documents to return for each query.
- **filters** (<code>dict\[str, Any\] | None</code>) – a dictionary of filters to apply to the search. Accepts filters in haystack format.

**Returns:**

- <code>list\[list\[Document\]\]</code> – matching documents for each query.

#### `search_async`

```python
search_async(
    queries: list[str], top_k: int, filters: dict[str, Any] | None = None
) -> list[list[Document]]
```

Asynchronously search the documents in the store using the provided text queries.

Asynchronous methods are only supported for HTTP connections.

**Parameters:**

- **queries** (<code>list\[str\]</code>) – the list of queries to search for.
- **top_k** (<code>int</code>) – top_k documents to return for each query.
- **filters** (<code>dict\[str, Any\] | None</code>) – a dictionary of filters to apply to the search. Accepts filters in haystack format.

**Returns:**

- <code>list\[list\[Document\]\]</code> – matching documents for each query.

#### `search_embeddings`

```python
search_embeddings(
    query_embeddings: list[list[float]],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[list[Document]]
```

Perform vector search on the stored document, pass the embeddings of the queries instead of their text.

**Parameters:**

- **query_embeddings** (<code>list\[list\[float\]\]</code>) – a list of embeddings to use as queries.
- **top_k** (<code>int</code>) – the maximum number of documents to retrieve.
- **filters** (<code>dict\[str, Any\] | None</code>) – a dictionary of filters to apply to the search. Accepts filters in haystack format.

**Returns:**

- <code>list\[list\[Document\]\]</code> – a list of lists of documents that match the given filters.

#### `search_embeddings_async`

```python
search_embeddings_async(
    query_embeddings: list[list[float]],
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[list[Document]]
```

Asynchronously perform vector search on the stored document, pass the embeddings of the queries instead of
their text.

Asynchronous methods are only supported for HTTP connections.

**Parameters:**

- **query_embeddings** (<code>list\[list\[float\]\]</code>) – a list of embeddings to use as queries.
- **top_k** (<code>int</code>) – the maximum number of documents to retrieve.
- **filters** (<code>dict\[str, Any\] | None</code>) – a dictionary of filters to apply to the search. Accepts filters in haystack format.

**Returns:**

- <code>list\[list\[Document\]\]</code> – a list of lists of documents that match the given filters.

#### `count_documents_by_filter`

```python
count_documents_by_filter(filters: dict[str, Any]) -> int
```

Returns the number of documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to count documents.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns:**

- <code>int</code> – The number of documents that match the filters.

#### `count_documents_by_filter_async`

```python
count_documents_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously returns the number of documents that match the provided filters.

Asynchronous methods are only supported for HTTP connections.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to count documents.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns:**

- <code>int</code> – The number of documents that match the filters.

#### `count_unique_metadata_by_filter`

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Returns the number of unique values for each specified metadata field
of the documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to count documents.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **metadata_fields** (<code>list\[str\]</code>) – List of field names to calculate unique values for.
  Field names can include or omit the "meta." prefix.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary mapping each metadata field name to the count of
  its unique values among the filtered documents.

#### `count_unique_metadata_by_filter_async`

```python
count_unique_metadata_by_filter_async(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Asynchronously returns the number of unique values for each specified metadata field
of the documents that match the provided filters.

Asynchronous methods are only supported for HTTP connections.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to count documents.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **metadata_fields** (<code>list\[str\]</code>) – List of field names to calculate unique values for.
  Field names can include or omit the "meta." prefix.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary mapping each metadata field name to the count of
  its unique values among the filtered documents.

#### `get_metadata_fields_info`

```python
get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Returns information about the metadata fields in the collection.

Since ChromaDB doesn't maintain a schema, this method samples documents
to infer field types.

If we populated the collection with documents like:

```python
Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1})
Document(content="Doc 2", meta={"category": "B", "status": "inactive"})
```

This method would return:

```python
{
    'category': {'type': 'keyword'},
    'status': {'type': 'keyword'},
    'priority': {'type': 'long'},
}
```

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – Dictionary mapping field names to their type information.

#### `get_metadata_fields_info_async`

```python
get_metadata_fields_info_async() -> dict[str, dict[str, str]]
```

Asynchronously returns information about the metadata fields in the collection.

Asynchronous methods are only supported for HTTP connections.

Since ChromaDB doesn't maintain a schema, this method samples documents
to infer field types.

If we populated the collection with documents like:

```python
Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1})
Document(content="Doc 2", meta={"category": "B", "status": "inactive"})
```

This method would return:

```python
{
    'category': {'type': 'keyword'},
    'status': {'type': 'keyword'},
    'priority': {'type': 'long'},
}
```

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – Dictionary mapping field names to their type information.

#### `get_metadata_field_min_max`

```python
get_metadata_field_min_max(metadata_field: str) -> dict[str, Any]
```

Returns the minimum and maximum values for the given metadata field.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to get the minimum and maximum values for.
  Can include or omit the "meta." prefix.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the keys "min" and "max", where each value is
  the minimum or maximum value of the metadata field across all documents.
  Returns:

```python
  {"min": None, "max": None}
```

if field doesn't exist or has no values.

#### `get_metadata_field_min_max_async`

```python
get_metadata_field_min_max_async(metadata_field: str) -> dict[str, Any]
```

Asynchronously returns the minimum and maximum values for the given metadata field.

Asynchronous methods are only supported for HTTP connections.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to get the minimum and maximum values for.
  Can include or omit the "meta." prefix.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the keys "min" and "max", where each value is
  the minimum or maximum value of the metadata field across all documents.
  Returns:

```python
  {"min": None, "max": None}
```

if field doesn't exist or has no values.

#### `get_metadata_field_unique_values`

```python
get_metadata_field_unique_values(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10,
) -> tuple[list[str], int]
```

Returns unique values for a metadata field, optionally filtered by
a search term in the content field, with pagination support.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to get unique values for.
  Can include or omit the "meta." prefix.
- **search_term** (<code>str | None</code>) – Optional search term to filter documents by matching
  in the content field.
- **from\_** (<code>int</code>) – The offset to start returning values from (for pagination).
- **size** (<code>int</code>) – The maximum number of unique values to return.

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – A tuple containing list of unique values and total count of unique values.

#### `get_metadata_field_unique_values_async`

```python
get_metadata_field_unique_values_async(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10,
) -> tuple[list[str], int]
```

Asynchronously returns unique values for a metadata field, optionally filtered by
a search term in the content field, with pagination support.

Asynchronous methods are only supported for HTTP connections.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to get unique values for.
  Can include or omit the "meta." prefix.
- **search_term** (<code>str | None</code>) – Optional search term to filter documents by matching
  in the content field.
- **from\_** (<code>int</code>) – The offset to start returning values from (for pagination).
- **size** (<code>int</code>) – The maximum number of unique values to return.

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – A tuple containing list of unique values and total count of unique values.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ChromaDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ChromaDocumentStore</code> – Deserialized component.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

## `haystack_integrations.document_stores.chroma.errors`

### `ChromaDocumentStoreError`

Bases: <code>DocumentStoreError</code>

Parent class for all ChromaDocumentStore exceptions.

### `ChromaDocumentStoreFilterError`

Bases: <code>FilterError</code>, <code>ValueError</code>

Raised when a filter is not valid for a ChromaDocumentStore.

### `ChromaDocumentStoreConfigError`

Bases: <code>ChromaDocumentStoreError</code>

Raised when a configuration is not valid for a ChromaDocumentStore.

## `haystack_integrations.document_stores.chroma.utils`

### `get_embedding_function`

```python
get_embedding_function(function_name: str, **kwargs: Any) -> EmbeddingFunction
```

Load an embedding function by name.

**Parameters:**

- **function_name** (<code>str</code>) – the name of the embedding function.
- **kwargs** (<code>Any</code>) – additional arguments to pass to the embedding function.

**Returns:**

- <code>EmbeddingFunction</code> – the loaded embedding function.

**Raises:**

- <code>ChromaDocumentStoreConfigError</code> – if the function name is invalid.
