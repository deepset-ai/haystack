---
title: "Arangodb"
id: integrations-arangodb
description: "Arangodb integration for Haystack"
slug: "/integrations-arangodb"
---


## haystack_integrations.components.retrievers.arangodb.embedding_retriever

### ArangoEmbeddingRetriever

Retrieves documents from an `ArangoDocumentStore` using vector similarity on embeddings.

The similarity function is configured on the `ArangoDocumentStore` (cosine, dot product, or L2).

Example usage:

```python
from haystack_integrations.document_stores.arangodb import ArangoDocumentStore
from haystack_integrations.components.retrievers.arangodb import ArangoEmbeddingRetriever

store = ArangoDocumentStore(host="http://localhost:8529", database="haystack",
                            username="root", collection_name="docs", embedding_dimension=768)
retriever = ArangoEmbeddingRetriever(document_store=store, top_k=5)
result = retriever.run(query_embedding=[0.1, 0.2, ...])
```

#### __init__

```python
__init__(
    *,
    document_store: ArangoDocumentStore,
    top_k: int = 10,
    filters: dict[str, Any] | None = None
) -> None
```

Creates a new ArangoEmbeddingRetriever.

**Parameters:**

- **document_store** (<code>ArangoDocumentStore</code>) – The `ArangoDocumentStore` to retrieve documents from.
- **top_k** (<code>int</code>) – Maximum number of documents to return.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional Haystack metadata filters applied at retrieval time.

#### run

```python
run(
    query_embedding: list[float],
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, list[Document]]
```

Retrieves documents most similar to `query_embedding`.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – The query vector.
- **top_k** (<code>int | None</code>) – Overrides the instance-level `top_k` for this call.
- **filters** (<code>dict\[str, Any\] | None</code>) – Overrides the instance-level `filters` for this call.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with `documents` — a list of `Document` objects sorted by score.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ArangoEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ArangoEmbeddingRetriever</code> – Deserialized component.

## haystack_integrations.document_stores.arangodb.document_store

### ArangoDocumentStore

A Haystack DocumentStore backed by [ArangoDB](https://www.arangodb.com/).

Documents are stored in an ArangoDB collection and support vector similarity search
via AQL vector functions (requires ArangoDB 3.12+).

Example usage:

```python
from haystack_integrations.document_stores.arangodb import ArangoDocumentStore
from haystack.utils import Secret

store = ArangoDocumentStore(
    host="http://localhost:8529",
    database="haystack",
    username=Secret.from_env_var("ARANGO_USERNAME", strict=False),
    password=Secret.from_env_var("ARANGO_PASSWORD"),
    collection_name="documents",
    embedding_dimension=768,
)
```

#### __init__

```python
__init__(
    *,
    host: str = "http://localhost:8529",
    database: str = "haystack",
    username: Secret = Secret.from_env_var("ARANGO_USERNAME", strict=False),
    password: Secret = Secret.from_env_var("ARANGO_PASSWORD"),
    collection_name: str = "haystack_documents",
    embedding_dimension: int = 768,
    recreate_collection: bool = False,
    similarity_function: Literal["cosine", "dot_product", "l2"] = "cosine"
) -> None
```

Creates a new ArangoDocumentStore instance.

**Parameters:**

- **host** (<code>str</code>) – ArangoDB server URL, e.g. `http://localhost:8529`.
- **database** (<code>str</code>) – Name of the ArangoDB database to use. Created if it does not exist.
- **username** (<code>Secret</code>) – ArangoDB username as a `Secret`. Defaults to `ARANGO_USERNAME` env var,
  falling back to `root` if the variable is not set.
- **password** (<code>Secret</code>) – ArangoDB password as a `Secret`. Defaults to `ARANGO_PASSWORD` env var.
- **collection_name** (<code>str</code>) – Name of the collection to store documents in.
- **embedding_dimension** (<code>int</code>) – Dimensionality of document embeddings.
- **recreate_collection** (<code>bool</code>) – If `True`, drop and recreate the collection on startup.
- **similarity_function** (<code>Literal['cosine', 'dot_product', 'l2']</code>) – Vector similarity function to use for embedding retrieval.
  One of `"cosine"` (default), `"dot_product"`, or `"l2"`.

#### count_documents

```python
count_documents() -> int
```

Returns the number of documents in the store.

**Returns:**

- <code>int</code> – Document count.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns documents matching the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Haystack metadata filters. If `None`, all documents are returned.

**Returns:**

- <code>list\[Document\]</code> – List of matching `Document` objects.

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Writes documents to the store.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to write.
- **policy** (<code>DuplicatePolicy</code>) – How to handle duplicates — `OVERWRITE`, `SKIP`, or `FAIL` (default).

**Returns:**

- <code>int</code> – Number of documents written.

**Raises:**

- <code>ValueError</code> – If `documents` contains non-`Document` objects.
- <code>DuplicateDocumentError</code> – If a duplicate is found and policy is `FAIL`.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes documents by their IDs.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – List of document IDs to delete.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ArangoDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ArangoDocumentStore</code> – Deserialized component.
