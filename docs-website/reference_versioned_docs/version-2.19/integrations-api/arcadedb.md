---
title: "ArcadeDB"
id: integrations-arcadedb
description: "ArcadeDB integration for Haystack"
slug: "/integrations-arcadedb"
---


## haystack_integrations.components.retrievers.arcadedb.embedding_retriever

### ArcadeDBEmbeddingRetriever

Retrieve documents from ArcadeDB using vector similarity (LSM_VECTOR / HNSW index).

Usage example:

```python
from haystack import Document
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.arcadedb import ArcadeDBEmbeddingRetriever
from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore

store = ArcadeDBDocumentStore(database="mydb")
retriever = ArcadeDBEmbeddingRetriever(document_store=store, top_k=5)

# Add documents to DocumentStore
documents = [
    Document(text="My name is Carla and I live in Berlin"),
    Document(text="My name is Paul and I live in New York"),
    Document(text="My name is Silvano and I live in Matera"),
    Document(text="My name is Usagi Tsukino and I live in Tokyo"),
]
document_store.write_documents(documents)

embedder = SentenceTransformersTextEmbedder()
query_embeddings = embedder.run("Who lives in Berlin?")["embedding"]

result = retriever.run(query=query_embeddings)
for doc in result["documents"]:
    print(doc.content)
```

#### __init__

```python
__init__(
    *,
    document_store: ArcadeDBDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Create an ArcadeDBEmbeddingRetriever.

**Parameters:**

- **document_store** (<code>ArcadeDBDocumentStore</code>) – An instance of `ArcadeDBDocumentStore`.
- **filters** (<code>dict\[str, Any\] | None</code>) – Default filters applied to every retrieval call.
- **top_k** (<code>int</code>) – Maximum number of documents to return.
- **filter_policy** (<code>FilterPolicy</code>) – How runtime filters interact with default filters.

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

- **query_embedding** (<code>list\[float\]</code>) – The embedding vector to search with.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional filters to narrow results.
- **top_k** (<code>int | None</code>) – Maximum number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of `Document`s most similar to the given `query_embedding`

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ArcadeDBEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ArcadeDBEmbeddingRetriever</code> – Deserialized component.

## haystack_integrations.document_stores.arcadedb.document_store

ArcadeDB DocumentStore for Haystack 2.x — document storage + vector search via HTTP/JSON API.

### ArcadeDBDocumentStore

An ArcadeDB-backed DocumentStore for Haystack 2.x.

Uses ArcadeDB's HTTP/JSON API for all operations — no special drivers required.
Supports HNSW vector search (LSM_VECTOR) and SQL metadata filtering.

Usage example:

```python
from haystack.dataclasses.document import Document
from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore

document_store = ArcadeDBDocumentStore(
    url="http://localhost:2480",
    database="haystack",
    embedding_dimension=768,
)
document_store.write_documents([
    Document(content="This is first", embedding=[0.0]*5),
    Document(content="This is second", embedding=[0.1, 0.2, 0.3, 0.4, 0.5])
])
```

#### __init__

```python
__init__(
    *,
    url: str = "http://localhost:2480",
    database: str = "haystack",
    username: Secret = Secret.from_env_var("ARCADEDB_USERNAME", strict=False),
    password: Secret = Secret.from_env_var("ARCADEDB_PASSWORD", strict=False),
    type_name: str = "Document",
    embedding_dimension: int = 768,
    similarity_function: str = "cosine",
    recreate_type: bool = False,
    create_database: bool = True
) -> None
```

Create an ArcadeDBDocumentStore instance.

**Parameters:**

- **url** (<code>str</code>) – ArcadeDB HTTP endpoint.
- **database** (<code>str</code>) – Database name.
- **username** (<code>Secret</code>) – HTTP Basic Auth username (default: `ARCADEDB_USERNAME` env var).
- **password** (<code>Secret</code>) – HTTP Basic Auth password (default: `ARCADEDB_PASSWORD` env var).
- **type_name** (<code>str</code>) – Vertex type name for documents.
- **embedding_dimension** (<code>int</code>) – Vector dimension for the HNSW index.
- **similarity_function** (<code>str</code>) – Distance metric — `"cosine"`, `"euclidean"`, or `"dot"`.
- **recreate_type** (<code>bool</code>) – If `True`, drop and recreate the type on initialization.
- **create_database** (<code>bool</code>) – If `True`, create the database if it doesn't exist.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the DocumentStore to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ArcadeDBDocumentStore
```

Deserializes the DocumentStore from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>ArcadeDBDocumentStore</code> – The deserialized DocumentStore.

#### count_documents

```python
count_documents() -> int
```

Returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – Number of documents in the document store.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Return documents matching the given filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Haystack filter dictionary.

**Returns:**

- <code>list\[Document\]</code> – List of matching documents.

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Write documents to the store.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Haystack Documents to write.
- **policy** (<code>DuplicatePolicy</code>) – How to handle duplicate document IDs.

**Returns:**

- <code>int</code> – Number of documents written.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Delete documents by their IDs.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – List of document IDs to delete.

#### delete_all_documents

```python
delete_all_documents() -> None
```

Deletes all documents in the document store.

#### delete_by_filter

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for deletion.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns:**

- <code>int</code> – The number of documents deleted.

#### update_by_filter

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Updates the metadata of all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update.

**Returns:**

- <code>int</code> – The number of documents updated.

#### count_documents_by_filter

```python
count_documents_by_filter(filters: dict[str, Any]) -> int
```

Counts the number of documents matching the provided filter

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to the documents

**Returns:**

- <code>int</code> – The number of documents that match the filter

#### count_unique_metadata_by_filter

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Counts unique values for each metadata field in documents matching the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to the document list.
- **metadata_fields** (<code>list\[str\]</code>) – Metadata fields for which to count unique values.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary where keys are metadata field names and values are the
  counts of unique values for that field.

#### get_metadata_fields_info

```python
get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Returns the metadata fields and their corresponding types based on sampled documents.

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – A dictionary mapping field names to dictionaries with a `type` key.

#### get_metadata_field_min_max

```python
get_metadata_field_min_max(metadata_field: str) -> dict[str, Any]
```

For a given metadata field, finds its min and max values.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to inspect.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with `min` and `max` keys and their corresponding values.

#### get_metadata_field_unique_values

```python
get_metadata_field_unique_values(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10,
) -> tuple[list[str], int]
```

Retrieves unique values for a field matching a search term or all possible values
if no search term is given.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to inspect.
- **search_term** (<code>str | None</code>) – Optional case-insensitive substring search term.
- **from\_** (<code>int</code>) – The starting index for pagination.
- **size** (<code>int</code>) – The number of values to return.

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – A tuple containing the paginated values and the total count.
