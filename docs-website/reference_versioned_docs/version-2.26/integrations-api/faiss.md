---
title: "FAISS"
id: integrations-faiss
description: "FAISS integration for Haystack"
slug: "/integrations-faiss"
---


## haystack_integrations.components.retrievers.faiss.embedding_retriever

### FAISSEmbeddingRetriever

Retrieves documents from the `FAISSDocumentStore`, based on their dense embeddings.

Example usage:

```python
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.faiss import FAISSDocumentStore
from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever

document_store = FAISSDocumentStore(embedding_dim=768)

documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(content="Elephants have been observed to behave in a way that indicates a high level of intelligence."),
    Document(content="In certain places, you can witness the phenomenon of bioluminescent waves."),
]

document_embedder = SentenceTransformersDocumentEmbedder()
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)["documents"]

document_store.write_documents(documents_with_embeddings, policy=DuplicatePolicy.OVERWRITE)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
query_pipeline.add_component("retriever", FAISSEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "How many languages are there?"
res = query_pipeline.run({"text_embedder": {"text": query}})

assert res["retriever"]["documents"][0].content == "There are over 7,000 languages spoken around the world today."
```

#### __init__

```python
__init__(
    *,
    document_store: FAISSDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
)
```

**Parameters:**

- **document_store** (<code>FAISSDocumentStore</code>) – An instance of `FAISSDocumentStore`.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents at initialisation time. At runtime, these are merged
  with any runtime filters according to the `filter_policy`.
- **top_k** (<code>int</code>) – Maximum number of Documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how init-time and runtime filters are combined.
  See `FilterPolicy` for details. Defaults to `FilterPolicy.REPLACE`.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of `FAISSDocumentStore`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> FAISSEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>FAISSEmbeddingRetriever</code> – Deserialized component.

#### run

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents from the `FAISSDocumentStore`, based on their embeddings.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – Maximum number of Documents to return. Overrides the value set at initialization.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of `Document`s that are similar to `query_embedding`.

#### run_async

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents from the `FAISSDocumentStore`, based on their embeddings.

Since FAISS search is CPU-bound and fully in-memory, this delegates directly to the synchronous
`run()` method. No I/O or network calls are involved.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – Maximum number of Documents to return. Overrides the value set at initialization.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of `Document`s that are similar to `query_embedding`.

## haystack_integrations.document_stores.faiss.document_store

### FAISSDocumentStore

A Document Store using FAISS for vector search and a simple JSON file for metadata storage.

This Document Store is suitable for small to medium-sized datasets where simplicity is preferred over scalability.
It supports basic persistence by saving the FAISS index to a `.faiss` file and documents to a `.json` file.

#### __init__

```python
__init__(
    index_path: str | None = None,
    index_string: str = "Flat",
    embedding_dim: int = 768,
)
```

Initializes the FAISSDocumentStore.

**Parameters:**

- **index_path** (<code>str | None</code>) – Path to save/load the index and documents. If None, the store is in-memory only.
- **index_string** (<code>str</code>) – The FAISS index factory string. Default is "Flat".
- **embedding_dim** (<code>int</code>) – The dimension of the embeddings. Default is 768.

**Raises:**

- <code>DocumentStoreError</code> – If the FAISS index cannot be initialized.
- <code>ValueError</code> – If `index_path` points to a missing `.faiss` file when loading persisted data.

#### count_documents

```python
count_documents() -> int
```

Returns the number of documents in the store.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary of filters to apply.

**Returns:**

- <code>list\[Document\]</code> – A list of matching Documents.

**Raises:**

- <code>FilterError</code> – If the filter structure is invalid.

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL
) -> int
```

Writes documents to the store.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The list of documents to write.
- **policy** (<code>DuplicatePolicy</code>) – The policy to handle duplicate documents.

**Returns:**

- <code>int</code> – The number of documents written.

**Raises:**

- <code>ValueError</code> – If `documents` is not an iterable of `Document` objects.
- <code>DuplicateDocumentError</code> – If a duplicate document is found and `policy` is `DuplicatePolicy.FAIL`.
- <code>DocumentStoreError</code> – If the FAISS index is unexpectedly unavailable when adding embeddings.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes documents from the store.

**Raises:**

- <code>DocumentStoreError</code> – If the FAISS index is unexpectedly unavailable when removing embeddings.

#### delete_all_documents

```python
delete_all_documents() -> None
```

Deletes all documents from the store.

#### search

```python
search(
    query_embedding: list[float],
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[Document]
```

Performs a vector search.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – The query embedding.
- **top_k** (<code>int</code>) – The number of results to return.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters to apply.

**Returns:**

- <code>list\[Document\]</code> – A list of matching Documents.

**Raises:**

- <code>FilterError</code> – If the filter structure is invalid.

#### delete_by_filter

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes documents that match the provided filters from the store.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – A dictionary of filters to apply to find documents to delete.

**Returns:**

- <code>int</code> – The number of documents deleted.

**Raises:**

- <code>FilterError</code> – If the filter structure is invalid.
- <code>DocumentStoreError</code> – If the FAISS index is unexpectedly unavailable when removing embeddings.

#### count_documents_by_filter

```python
count_documents_by_filter(filters: dict[str, Any]) -> int
```

Returns the number of documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – A dictionary of filters to apply.

**Returns:**

- <code>int</code> – The number of matching documents.

**Raises:**

- <code>FilterError</code> – If the filter structure is invalid.

#### update_by_filter

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Updates documents that match the provided filters with the new metadata.

Note: Updates are performed in-memory only. To persist these changes,
you must explicitly call `save()` after updating.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – A dictionary of filters to apply to find documents to update.
- **meta** (<code>dict\[str, Any\]</code>) – A dictionary of metadata key-value pairs to update in the matching documents.

**Returns:**

- <code>int</code> – The number of documents updated.

**Raises:**

- <code>FilterError</code> – If the filter structure is invalid.

#### get_metadata_fields_info

```python
get_metadata_fields_info() -> dict[str, dict[str, Any]]
```

Infers and returns the types of all metadata fields from the stored documents.

**Returns:**

- <code>dict\[str, dict\[str, Any\]\]</code> – A dictionary mapping field names to dictionaries with a "type" key
  (e.g. `{"field": {"type": "long"}}`).

#### get_metadata_field_min_max

```python
get_metadata_field_min_max(field_name: str) -> dict[str, Any]
```

Returns the minimum and maximum values for a specific metadata field.

**Parameters:**

- **field_name** (<code>str</code>) – The name of the metadata field.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with keys "min" and "max" containing the respective min and max values.

#### get_metadata_field_unique_values

```python
get_metadata_field_unique_values(field_name: str) -> list[Any]
```

Returns all unique values for a specific metadata field.

**Parameters:**

- **field_name** (<code>str</code>) – The name of the metadata field.

**Returns:**

- <code>list\[Any\]</code> – A list of unique values for the specified field.

#### count_unique_metadata_by_filter

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any], fields: list[str]
) -> dict[str, int]
```

Returns a count of unique values for multiple metadata fields, optionally scoped by a filter.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – A dictionary of filters to apply.
- **fields** (<code>list\[str\]</code>) – A list of metadata field names to count unique values for.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary mapping each field name to the count of its unique values.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the store to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> FAISSDocumentStore
```

Deserializes the store from a dictionary.

#### save

```python
save(index_path: str | Path) -> None
```

Saves the index and documents to disk.

**Raises:**

- <code>DocumentStoreError</code> – If the FAISS index is unexpectedly unavailable.

#### load

```python
load(index_path: str | Path) -> None
```

Loads the index and documents from disk.

**Raises:**

- <code>ValueError</code> – If the `.faiss` file does not exist.
