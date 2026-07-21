---
title: "FalkorDB"
id: integrations-falkordb
description: "FalkorDB integration for Haystack"
slug: "/integrations-falkordb"
---


## haystack_integrations.components.retrievers.falkordb.cypher_retriever

### FalkorDBCypherRetriever

A power-user retriever for executing arbitrary OpenCypher queries against FalkorDB.

This retriever allows you to leverage graph traversal and multi-hop queries in
GraphRAG pipelines. The query must return nodes or dictionaries that can be
mapped exactly to a Haystack `Document`.

**Security Warning:** Raw Cypher queries must only come from trusted sources. Do
not use un-sanitised user input directly in query strings. Use `parameters` instead.

Usage example:

```python
from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore
from haystack_integrations.components.retrievers.falkordb import FalkorDBCypherRetriever

store = FalkorDBDocumentStore(host="localhost", port=6379)
retriever = FalkorDBCypherRetriever(
    document_store=store,
    custom_cypher_query="MATCH (d:Document)-[:RELATES_TO]->(:Concept {name: $concept}) RETURN d"
)

res = retriever.run(parameters={"concept": "GraphRAG"})
print(res["documents"])
```

#### __init__

```python
__init__(
    document_store: FalkorDBDocumentStore,
    custom_cypher_query: str | None = None,
) -> None
```

Create a new FalkorDBCypherRetriever.

**Parameters:**

- **document_store** (<code>FalkorDBDocumentStore</code>) – The FalkorDBDocumentStore instance.
- **custom_cypher_query** (<code>str | None</code>) – A static OpenCypher query to execute. Can be
  overridden at runtime by passing `query` to `run()`.

**Raises:**

- <code>ValueError</code> – If the provided `document_store` is not a `FalkorDBDocumentStore`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialise the retriever to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary representation of the retriever.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> FalkorDBCypherRetriever
```

Deserialise a `FalkorDBCypherRetriever` produced by `to_dict`.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Serialised retriever dictionary.

**Returns:**

- <code>FalkorDBCypherRetriever</code> – Reconstructed `FalkorDBCypherRetriever` instance.

#### run

```python
run(
    query: str | None = None, parameters: dict[str, Any] | None = None
) -> dict[str, list[Document]]
```

Retrieve documents by executing an OpenCypher query.

If a `query` is provided here, it overrides the `custom_cypher_query`
set during initialisation.

**Parameters:**

- **query** (<code>str | None</code>) – Optional OpenCypher query string.
- **parameters** (<code>dict\[str, Any\] | None</code>) – Optional dictionary of query parameters (referenced as
  `$param_name` in the Cypher string).

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – Dictionary containing a `"documents"` key with the retrieved documents.

**Raises:**

- <code>ValueError</code> – If no query string is provided (both here and at init).

## haystack_integrations.components.retrievers.falkordb.embedding_retriever

### FalkorDBEmbeddingRetriever

A component for retrieving documents from a FalkorDBDocumentStore using vector similarity.

The retriever uses FalkorDB's native vector search index to find documents whose embeddings
are most similar to the provided query embedding.

Usage example:

```python
from haystack.dataclasses import Document
from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore
from haystack_integrations.components.retrievers.falkordb import FalkorDBEmbeddingRetriever

store = FalkorDBDocumentStore(host="localhost", port=6379)
store.write_documents([
    Document(content="GraphRAG is powerful.", embedding=[0.1, 0.2, 0.3]),
    Document(content="FalkorDB is fast.", embedding=[0.8, 0.9, 0.1]),
])

retriever = FalkorDBEmbeddingRetriever(document_store=store)
res = retriever.run(query_embedding=[0.1, 0.2, 0.3])
print(res["documents"][0].content)  # "GraphRAG is powerful."
```

#### __init__

```python
__init__(
    document_store: FalkorDBDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: FilterPolicy = FilterPolicy.REPLACE,
) -> None
```

Create a new FalkorDBEmbeddingRetriever.

**Parameters:**

- **document_store** (<code>FalkorDBDocumentStore</code>) – The FalkorDBDocumentStore instance.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional Haystack filters to narrow down the search space.
- **top_k** (<code>int</code>) – Maximum number of documents to retrieve.
- **filter_policy** (<code>FilterPolicy</code>) – Policy to determine how runtime filters are combined with
  initialization filters.

**Raises:**

- <code>ValueError</code> – If the provided `document_store` is not a `FalkorDBDocumentStore`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialise the retriever to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary representation of the retriever.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> FalkorDBEmbeddingRetriever
```

Deserialise a `FalkorDBEmbeddingRetriever` produced by `to_dict`.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Serialised retriever dictionary.

**Returns:**

- <code>FalkorDBEmbeddingRetriever</code> – Reconstructed `FalkorDBEmbeddingRetriever` instance.

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

- **query_embedding** (<code>list\[float\]</code>) – Query embedding vector.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional Haystack filters to be combined with the init filters based
  on the configured filter policy.
- **top_k** (<code>int | None</code>) – Maximum number of documents to return. If not provided, the default
  top_k from initialization is used.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – Dictionary containing a `"documents"` key with the retrieved documents.

## haystack_integrations.document_stores.falkordb.document_store

### FalkorDBDocumentStore

Bases: <code>DocumentStore</code>

A Haystack DocumentStore backed by FalkorDB — a high-performance graph database.

Optimised for GraphRAG workloads.

Documents are stored as graph nodes (labelled `Document` by default) in a named
FalkorDB graph. Document properties, including `meta` fields, are stored
**flat** at the same level as `id` and `content` — exactly the same layout as
the `neo4j-haystack` reference integration.

Vector search is performed via FalkorDB's native vector index —
**no APOC is required**. All bulk writes use `UNWIND` + `MERGE` for safe,
idiomatic OpenCypher upserts.

Usage example:

```python
from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore
from haystack.dataclasses import Document

store = FalkorDBDocumentStore(host="localhost", port=6379)
store.write_documents([
    Document(content="Hello, GraphRAG!", meta={"year": 2024}),
])
print(store.count_documents())  # 1
```

#### __init__

```python
__init__(
    *,
    host: str = "localhost",
    port: int = 6379,
    graph_name: str = "haystack",
    username: str | None = None,
    password: Secret | None = None,
    node_label: str = "Document",
    embedding_dim: int = 768,
    embedding_field: str = "embedding",
    similarity: SimilarityFunction = "cosine",
    write_batch_size: int = 100,
    recreate_graph: bool = False,
    verify_connectivity: bool = False
) -> None
```

Create a new FalkorDBDocumentStore.

**Parameters:**

- **host** (<code>str</code>) – Hostname of the FalkorDB server.
- **port** (<code>int</code>) – Port the FalkorDB server listens on.
- **graph_name** (<code>str</code>) – Name of the FalkorDB graph to use. Each graph is an isolated
  namespace.
- **username** (<code>str | None</code>) – Optional username for FalkorDB authentication.
- **password** (<code>Secret | None</code>) – Optional :class:`haystack.utils.Secret` holding the FalkorDB
  password. The secret value is resolved lazily on first connection.
- **node_label** (<code>str</code>) – Label used for document nodes in the graph.
- **embedding_dim** (<code>int</code>) – Dimensionality of the vector embeddings. Used when
  creating the vector index.
- **embedding_field** (<code>str</code>) – Name of the node property that stores the embedding
  vector.
- **similarity** (<code>SimilarityFunction</code>) – Similarity function for the vector index. Accepted values
  are `"cosine"` and `"euclidean"`.
- **write_batch_size** (<code>int</code>) – Number of documents written per `UNWIND` batch.
- **recreate_graph** (<code>bool</code>) – When `True` the existing graph (and all its data) is
  dropped and recreated on initialisation. Useful for tests.
- **verify_connectivity** (<code>bool</code>) – When `True` a connectivity probe is run
  immediately in `__init__` — raises if the server is unreachable.

**Raises:**

- <code>ValueError</code> – If `similarity` is not `"cosine"` or `"euclidean"`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialise the store to a dictionary suitable for `from_dict`.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary representation of the store.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> FalkorDBDocumentStore
```

Deserialise a `FalkorDBDocumentStore` produced by `to_dict`.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Serialised store dictionary.

**Returns:**

- <code>FalkorDBDocumentStore</code> – Reconstructed `FalkorDBDocumentStore` instance.

#### count_documents

```python
count_documents() -> int
```

Return the number of documents currently stored in the graph.

**Returns:**

- <code>int</code> – Integer count of document nodes.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Retrieve all documents that match the provided Haystack filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Optional Haystack filter dict. When `None` all documents are
  returned. For filter syntax see
  [Metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns:**

- <code>list\[Document\]</code> – List of matching :class:`haystack.dataclasses.Document` objects.

**Raises:**

- <code>ValueError</code> – If the filter dict is malformed.

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Write documents to the FalkorDB graph using `UNWIND` + `MERGE` for batching.

Document `meta` fields are stored **flat** at the same level as `id` and
`content` — no prefix is added. This matches the layout used by the
`neo4j-haystack` reference integration.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of :class:`haystack.dataclasses.Document` objects.
- **policy** (<code>DuplicatePolicy</code>) – How to handle documents whose `id` already exists.
  Defaults to :attr:`DuplicatePolicy.NONE` (treated as FAIL).

**Returns:**

- <code>int</code> – Number of documents written or updated.

**Raises:**

- <code>ValueError</code> – If `documents` contains non-Document elements.
- <code>DuplicateDocumentError</code> – If `policy` is FAIL / NONE and a duplicate
  ID is encountered.
- <code>DocumentStoreError</code> – If any other DB error occurs.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Delete documents by their IDs using a single `UNWIND`-based query.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – List of document IDs to remove from the graph.

#### delete_all_documents

```python
delete_all_documents() -> None
```

Delete all documents from the graph.

#### delete_by_filter

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Delete all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack filter dict.

**Returns:**

- <code>int</code> – Number of documents deleted.

#### update_by_filter

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Update metadata fields on all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack filter dict selecting which documents to update.
- **meta** (<code>dict\[str, Any\]</code>) – Metadata fields to set. Keys may include or omit the `meta.` prefix.

**Returns:**

- <code>int</code> – Number of documents updated.

#### count_documents_by_filter

```python
count_documents_by_filter(filters: dict[str, Any]) -> int
```

Return the number of documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack filter dict.

**Returns:**

- <code>int</code> – Integer count of matching document nodes.

#### count_unique_metadata_by_filter

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Return the number of unique values for each metadata field among matching documents.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack filter dict. Pass an empty dict to count across all documents.
- **metadata_fields** (<code>list\[str\]</code>) – List of metadata field names. May include or omit the `meta.` prefix.

**Returns:**

- <code>dict\[str, int\]</code> – Dict mapping each field name (without `meta.` prefix) to its unique value count.

#### get_metadata_fields_info

```python
get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Return type information for each metadata field present on document nodes.

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – Dict mapping field names to a `{"type": <typename>}` dict.
  Type names are `"str"`, `"int"`, `"float"`, or `"bool"`.

#### get_metadata_field_min_max

```python
get_metadata_field_min_max(metadata_field: str) -> dict[str, Any]
```

Return the minimum and maximum values for the given metadata field.

**Parameters:**

- **metadata_field** (<code>str</code>) – Metadata field name. May include or omit the `meta.` prefix.

**Returns:**

- <code>dict\[str, Any\]</code> – Dict with keys `"min"` and `"max"`. Values are `None` when no documents
  have a non-null value for the field.

#### get_metadata_field_unique_values

```python
get_metadata_field_unique_values(
    metadata_field: str,
    search_term: str | None = None,
    size: int | None = 10000,
    after: dict[str, Any] | None = None,
) -> tuple[list[Any], dict[str, Any] | None]
```

Return distinct values for the given metadata field with optional filtering and pagination.

**Parameters:**

- **metadata_field** (<code>str</code>) – Metadata field name. May include or omit the `meta.` prefix.
- **search_term** (<code>str | None</code>) – Optional substring filter applied to string field values.
- **size** (<code>int | None</code>) – Maximum number of values to return per page. Defaults to 10 000.
- **after** (<code>dict\[str, Any\] | None</code>) – Pagination cursor returned by a previous call. Pass `None` for the first page.

**Returns:**

- <code>tuple\[list\[Any\], dict\[str, Any\] | None\]</code> – Tuple of `(values, next_cursor)`. `next_cursor` is `None` on the last page.
