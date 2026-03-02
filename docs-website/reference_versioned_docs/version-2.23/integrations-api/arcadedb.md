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
)
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
)
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
