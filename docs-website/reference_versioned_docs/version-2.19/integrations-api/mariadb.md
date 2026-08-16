---
title: "MariaDB"
id: integrations-mariadb
description: "MariaDB integration for Haystack"
slug: "/integrations-mariadb"
---


## haystack_integrations.components.retrievers.mariadb.embedding_retriever

### MariaDBEmbeddingRetriever

Retrieves documents from `MariaDBDocumentStore` using vector similarity search.

Uses MariaDB's native `VEC_DISTANCE_COSINE` or `VEC_DISTANCE_EUCLIDEAN` functions
with MHNSW indexing for efficient approximate nearest-neighbour search.

### Usage example

```python
from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore
from haystack_integrations.components.retrievers.mariadb import MariaDBEmbeddingRetriever

store = MariaDBDocumentStore(host="localhost", database="haystack", embedding_dimension=768)
retriever = MariaDBEmbeddingRetriever(document_store=store, top_k=5)
result = retriever.run(query_embedding=[0.1] * 768)
documents = result["documents"]
```

#### __init__

```python
__init__(
    *,
    document_store: MariaDBDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    score_threshold: float | None = None,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Initialize the MariaDBEmbeddingRetriever.

**Parameters:**

- **document_store** (<code>MariaDBDocumentStore</code>) – A `MariaDBDocumentStore` instance.
- **filters** (<code>dict\[str, Any\] | None</code>) – Default Haystack metadata filters applied to every query.
- **top_k** (<code>int</code>) – Maximum number of documents to return.
- **score_threshold** (<code>float | None</code>) – Minimum score to include a document. Documents below this score are excluded.
- **filter_policy** (<code>str | FilterPolicy</code>) – How runtime filters interact with init-time filters.

**Raises:**

- <code>ValueError</code> – If `document_store` is not a `MariaDBDocumentStore`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> MariaDBEmbeddingRetriever
```

Deserialize the component from a dictionary.

#### run

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents similar to the query embedding.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – The query vector.
- **filters** (<code>dict\[str, Any\] | None</code>) – Runtime filters merged with init-time filters per `filter_policy`.
- **top_k** (<code>int | None</code>) – Override the retriever's `top_k`.
- **score_threshold** (<code>float | None</code>) – Override the retriever's `score_threshold`.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – Dictionary with `"documents"` key containing the ranked results.

## haystack_integrations.components.retrievers.mariadb.keyword_retriever

### MariaDBKeywordRetriever

Retrieves documents from `MariaDBDocumentStore` using full-text keyword search.

Uses MariaDB's `MATCH ... AGAINST` full-text search in natural language mode,
backed by a FULLTEXT index on the `content` column.

### Usage example

```python
from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore
from haystack_integrations.components.retrievers.mariadb import MariaDBKeywordRetriever

store = MariaDBDocumentStore(host="localhost", database="haystack", embedding_dimension=768)
retriever = MariaDBKeywordRetriever(document_store=store, top_k=5)
result = retriever.run(query="climate change")
documents = result["documents"]
```

#### __init__

```python
__init__(
    *,
    document_store: MariaDBDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Initialize the MariaDBKeywordRetriever.

**Parameters:**

- **document_store** (<code>MariaDBDocumentStore</code>) – A `MariaDBDocumentStore` instance.
- **filters** (<code>dict\[str, Any\] | None</code>) – Default Haystack metadata filters.
- **top_k** (<code>int</code>) – Maximum number of documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – How runtime filters interact with init-time filters.

**Raises:**

- <code>ValueError</code> – If `document_store` is not a `MariaDBDocumentStore`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> MariaDBKeywordRetriever
```

Deserialize the component from a dictionary.

#### run

```python
run(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, list[Document]]
```

Retrieve documents matching the query via full-text search.

**Parameters:**

- **query** (<code>str</code>) – The keyword query string.
- **filters** (<code>dict\[str, Any\] | None</code>) – Runtime filters merged with init-time filters per `filter_policy`.
- **top_k** (<code>int | None</code>) – Override the retriever's `top_k`.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – Dictionary with `"documents"` key containing results ranked by relevance.

## haystack_integrations.document_stores.mariadb.document_store

### MariaDBDocumentStore

A Document Store backed by MariaDB 11.7+ using native VECTOR support.

Uses MariaDB's `VECTOR` datatype with `MHNSW` indexing for approximate nearest-neighbour
vector search, and `MATCH ... AGAINST` for full-text keyword search.

### Usage example

```python
from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore

store = MariaDBDocumentStore(
    host="localhost",
    port=3306,
    database="haystack",
    embedding_dimension=768,
)
store.write_documents(documents)
```

#### __init__

```python
__init__(
    *,
    host: str = "localhost",
    port: int = 3306,
    database: str = "haystack",
    user: Secret = Secret.from_env_var("MARIADB_USER"),
    password: Secret = Secret.from_env_var("MARIADB_PASSWORD"),
    table_name: str = "haystack_documents",
    recreate_table: bool = False,
    embedding_dimension: int = 768,
    distance: str = "cosine",
    create_vector_index: bool = False
) -> None
```

Initialize the MariaDBDocumentStore.

**Parameters:**

- **host** (<code>str</code>) – MariaDB host.
- **port** (<code>int</code>) – MariaDB port.
- **database** (<code>str</code>) – Database name.
- **user** (<code>Secret</code>) – Database user, read from the `MARIADB_USER` environment variable.
- **password** (<code>Secret</code>) – Database password, read from the `MARIADB_PASSWORD` environment variable.
- **table_name** (<code>str</code>) – Table used to store documents. Must contain only letters, digits, and underscores.
- **recreate_table** (<code>bool</code>) – Drop and recreate the table on init. **Deletes all data.**
- **embedding_dimension** (<code>int</code>) – Dimension of embedding vectors. Applied only when the table is created;
  ignored on an existing table.
- **distance** (<code>str</code>) – Distance function for vector similarity — `"cosine"` or `"euclidean"`. Applied only when
  the table is created; ignored on an existing table.
- **create_vector_index** (<code>bool</code>) – If `True`, creates an MHNSW vector index for fast ANN search. Requires every
  document to have a non-null embedding. Applied only when the table is created; ignored on an existing
  table.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this document store to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> MariaDBDocumentStore
```

Deserialize this document store from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>MariaDBDocumentStore</code> – Deserialized document store.

#### close

```python
close() -> None
```

Release the associated synchronous resources.

#### delete_table

```python
delete_table() -> None
```

Drop the documents table

#### count_documents

```python
count_documents() -> int
```

Return how many documents are present in the document store.

**Returns:**

- <code>int</code> – Number of documents in the document store.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Return the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – The filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

**Raises:**

- <code>TypeError</code> – If `filters` is not a dictionary.
- <code>ValueError</code> – If `filters` syntax is invalid.

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Write documents to the store.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to write to the document store.
- **policy** (<code>DuplicatePolicy</code>) – The duplicate policy to use when writing documents.

**Returns:**

- <code>int</code> – The number of documents written to the document store.

**Raises:**

- <code>ValueError</code> – If `documents` contains objects that are not of type `Document`.
- <code>DuplicateDocumentError</code> – If a document with the same id already exists in the document store
  and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
- <code>DocumentStoreError</code> – If the write operation fails for any other reason.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Delete documents that match the provided `document_ids` from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – The document ids to delete.
