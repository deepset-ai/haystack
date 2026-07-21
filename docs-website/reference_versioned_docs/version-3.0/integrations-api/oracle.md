---
title: "Oracle AI Vector Search"
id: integrations-oracle
description: "Oracle AI Vector Search integration for Haystack"
slug: "/integrations-oracle"
---


## haystack_integrations.components.retrievers.oracle.embedding_retriever

### OracleEmbeddingRetriever

Retrieves documents from an OracleDocumentStore using vector similarity.

Use inside a Haystack pipeline after a text embedder::

```
pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
pipeline.add_component("retriever", OracleEmbeddingRetriever(
    document_store=store, top_k=5
))
pipeline.connect("embedder.embedding", "retriever.query_embedding")
```

#### run

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents by vector similarity.

Args:
query_embedding: Dense float vector from an embedder component.
filters: Runtime filters, merged with constructor filters according to filter_policy.
top_k: Override the constructor top_k for this call.

Returns:
`{"documents": [Document, ...]}`

#### run_async

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Async variant of :meth:`run`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OracleEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OracleEmbeddingRetriever</code> – Deserialized component.

## haystack_integrations.document_stores.oracle.document_store

### OracleConnectionConfig

Connection parameters for Oracle Database.

Supports both thin (direct TCP) and thick (wallet / ADB-S) modes.
Thin mode requires no Oracle Instant Client; thick mode is activated
automatically when *wallet_location* is provided.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OracleConnectionConfig
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OracleConnectionConfig</code> – Deserialized component.

### OracleDocumentStore

Haystack DocumentStore backed by Oracle AI Vector Search.

Requires Oracle Database 23ai or later (for VECTOR data type and
IF NOT EXISTS DDL support).

Usage::

```
from haystack.utils import Secret
from haystack_integrations.document_stores.oracle import (
    OracleDocumentStore, OracleConnectionConfig,
)

store = OracleDocumentStore(
    connection_config=OracleConnectionConfig(
        user=Secret.from_env_var("ORACLE_USER"),
        password=Secret.from_env_var("ORACLE_PASSWORD"),
        dsn=Secret.from_env_var("ORACLE_DSN"),
    ),
    embedding_dim=1536,
)
```

#### __init__

```python
__init__(
    *,
    connection_config: OracleConnectionConfig,
    table_name: str = "haystack_documents",
    embedding_dim: int,
    distance_metric: Literal["COSINE", "EUCLIDEAN", "DOT"] = "COSINE",
    create_table_if_not_exists: bool = True,
    create_index: bool = False,
    hnsw_neighbors: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_accuracy: int = 95,
    hnsw_parallel: int = 4
) -> None
```

Initialise the document store and optionally create the backing table and indexes.

**Parameters:**

- **connection_config** (<code>OracleConnectionConfig</code>) – Oracle connection settings (user, password, DSN, optional wallet).
- **table_name** (<code>str</code>) – Name of the Oracle table used to store documents. Must be a valid Oracle
  identifier (letters, digits, `_`, `$`, `#`; max 128 chars; cannot start with a digit).
- **embedding_dim** (<code>int</code>) – Dimensionality of the embedding vectors. Must match the model producing them.
- **distance_metric** (<code>Literal['COSINE', 'EUCLIDEAN', 'DOT']</code>) – Vector distance function used for similarity search.
  One of `"COSINE"`, `"EUCLIDEAN"`, or `"DOT"`.
- **create_table_if_not_exists** (<code>bool</code>) – When `True` (default), creates the table and the DBMS_SEARCH
  keyword index on first use if they do not already exist. Set to `False` when connecting to a
  pre-existing table.
- **create_index** (<code>bool</code>) – When `True`, creates an HNSW vector index on initialisation. Equivalent to
  calling :meth:`create_hnsw_index` manually. Defaults to `False`.
- **hnsw_neighbors** (<code>int</code>) – Number of neighbours in the HNSW graph. Higher values improve recall at the
  cost of index size and build time. Defaults to `32`.
- **hnsw_ef_construction** (<code>int</code>) – Size of the dynamic candidate list during HNSW index construction.
  Higher values improve recall at the cost of build time. Defaults to `200`.
- **hnsw_accuracy** (<code>int</code>) – Target recall accuracy percentage for the HNSW index (0-100).
  Defaults to `95`.
- **hnsw_parallel** (<code>int</code>) – Degree of parallelism used when building the HNSW index. Defaults to `4`.

**Raises:**

- <code>ValueError</code> – If `table_name` is not a valid Oracle identifier or `embedding_dim` is not
  a positive integer.

#### create_keyword_index

```python
create_keyword_index() -> None
```

Create the DBMS_SEARCH keyword index on this table.

Safe to call multiple times — silently skips if the index already exists.
Required for keyword retrieval. Called automatically when
`create_table_if_not_exists=True`, but must be called explicitly
when connecting to a pre-existing table.

#### create_hnsw_index

```python
create_hnsw_index() -> None
```

Create an HNSW vector index on the embedding column.

Safe to call multiple times — uses IF NOT EXISTS.

#### create_hnsw_index_async

```python
create_hnsw_index_async() -> None
```

Asynchronously creates an HNSW vector index on the embedding column.

Safe to call multiple times — uses `IF NOT EXISTS`.

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Writes documents to the document store.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to write to the document store.
- **policy** (<code>DuplicatePolicy</code>) – The duplicate policy to use when writing documents.

**Returns:**

- <code>int</code> – The number of documents written to the document store.

**Raises:**

- <code>DuplicateDocumentError</code> – If a document with the same id already exists in the document store
  and the policy is set to `DuplicatePolicy.FAIL` or `DuplicatePolicy.NONE`.

#### write_documents_async

```python
write_documents_async(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Asynchronously writes documents to the document store.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to write to the document store.
- **policy** (<code>DuplicatePolicy</code>) – The duplicate policy to use when writing documents.

**Returns:**

- <code>int</code> – The number of documents written to the document store.

**Raises:**

- <code>DuplicateDocumentError</code> – If a document with the same id already exists in the document store
  and the policy is set to `DuplicatePolicy.FAIL` or `DuplicatePolicy.NONE`.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – The filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

#### filter_documents_async

```python
filter_documents_async(filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – The filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes documents that match the provided `document_ids` from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete

#### delete_documents_async

```python
delete_documents_async(document_ids: list[str]) -> None
```

Asynchronously deletes documents that match the provided `document_ids` from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete

#### count_documents

```python
count_documents() -> int
```

Returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – Number of documents in the document store.

#### count_documents_async

```python
count_documents_async() -> int
```

Asynchronously returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – Number of documents in the document store.

#### delete_table

```python
delete_table() -> None
```

Permanently drops the document store table and its associated DBMS_SEARCH keyword index.

Uses `DROP TABLE ... PURGE` which bypasses the Oracle recycle bin — the operation is
irreversible. The keyword index is dropped after the table; if either operation fails a
:class:`DocumentStoreError` is raised.

**Raises:**

- <code>DocumentStoreError</code> – If the table or keyword index cannot be dropped.

#### delete_table_async

```python
delete_table_async() -> None
```

Asynchronously permanently drops the document store table and its DBMS_SEARCH keyword index.

Uses `DROP TABLE ... PURGE` which bypasses the Oracle recycle bin — the operation is
irreversible.

**Raises:**

- <code>DocumentStoreError</code> – If the table or keyword index cannot be dropped.

#### delete_all_documents

```python
delete_all_documents() -> None
```

Removes all documents from the table using `TRUNCATE`.

`TRUNCATE` is non-recoverable — it cannot be rolled back and bypasses row-level triggers.
The table structure and indexes are preserved.

#### delete_all_documents_async

```python
delete_all_documents_async() -> None
```

Asynchronously removes all documents from the table using `TRUNCATE`.

`TRUNCATE` is non-recoverable — it cannot be rolled back and bypasses row-level triggers.
The table structure and indexes are preserved.

#### count_documents_by_filter

```python
count_documents_by_filter(filters: dict[str, Any]) -> int
```

Returns the number of documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack filter dict. An empty dict matches all documents.
  See the `metadata filtering docs <https://docs.haystack.deepset.ai/docs/metadata-filtering>`\_.

**Returns:**

- <code>int</code> – Count of matching documents.

#### count_documents_by_filter_async

```python
count_documents_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously returns the number of documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack filter dict. An empty dict matches all documents.
  See the `metadata filtering docs <https://docs.haystack.deepset.ai/docs/metadata-filtering>`\_.

**Returns:**

- <code>int</code> – Count of matching documents.

#### delete_by_filter

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack filter dict. An empty dict is treated as a no-op and returns `0`
  without touching the table.
  See the `metadata filtering docs <https://docs.haystack.deepset.ai/docs/metadata-filtering>`\_.

**Returns:**

- <code>int</code> – Number of deleted documents.

#### delete_by_filter_async

```python
delete_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously deletes all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack filter dict. An empty dict is treated as a no-op and returns `0`
  without touching the table.
  See the `metadata filtering docs <https://docs.haystack.deepset.ai/docs/metadata-filtering>`\_.

**Returns:**

- <code>int</code> – Number of deleted documents.

#### update_by_filter

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Merges `meta` into the metadata of all documents that match the provided filters.

Uses Oracle's `JSON_MERGEPATCH` — existing keys are updated, new keys are added,
and keys set to `null` in `meta` are removed.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack filter dict that selects which documents to update.
  See the `metadata filtering docs <https://docs.haystack.deepset.ai/docs/metadata-filtering>`\_.
- **meta** (<code>dict\[str, Any\]</code>) – Metadata patch to apply. Must be a non-empty dictionary.

**Returns:**

- <code>int</code> – Number of updated documents.

**Raises:**

- <code>ValueError</code> – If `meta` is empty.

#### update_by_filter_async

```python
update_by_filter_async(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Asynchronously merges `meta` into the metadata of all documents matching the provided filters.

Uses Oracle's `JSON_MERGEPATCH` — existing keys are updated, new keys are added,
and keys set to `null` in `meta` are removed.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack filter dict that selects which documents to update.
  See the `metadata filtering docs <https://docs.haystack.deepset.ai/docs/metadata-filtering>`\_.
- **meta** (<code>dict\[str, Any\]</code>) – Metadata patch to apply. Must be a non-empty dictionary.

**Returns:**

- <code>int</code> – Number of updated documents.

**Raises:**

- <code>ValueError</code> – If `meta` is empty.

#### count_unique_metadata_by_filter

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Returns the number of distinct values for each requested metadata field among matching documents.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack filter dict that scopes the document set.
  See the `metadata filtering docs <https://docs.haystack.deepset.ai/docs/metadata-filtering>`\_.
- **metadata_fields** (<code>list\[str\]</code>) – List of metadata field names to count distinct values for.
  Fields may be prefixed with `"meta."` (e.g. `"meta.lang"` or `"lang"`).
  Must be a non-empty list.

**Returns:**

- <code>dict\[str, int\]</code> – Dict mapping each field name to its distinct-value count.

**Raises:**

- <code>ValueError</code> – If `metadata_fields` is empty.
- <code>ValueError</code> – If any field name contains characters outside `[A-Za-z0-9_.]`.

#### count_unique_metadata_by_filter_async

```python
count_unique_metadata_by_filter_async(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Asynchronously returns the number of distinct values for each metadata field among matching documents.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack filter dict that scopes the document set.
  See the `metadata filtering docs <https://docs.haystack.deepset.ai/docs/metadata-filtering>`\_.
- **metadata_fields** (<code>list\[str\]</code>) – List of metadata field names to count distinct values for.
  Fields may be prefixed with `"meta."` (e.g. `"meta.lang"` or `"lang"`).
  Must be a non-empty list.

**Returns:**

- <code>dict\[str, int\]</code> – Dict mapping each field name to its distinct-value count.

**Raises:**

- <code>ValueError</code> – If `metadata_fields` is empty.
- <code>ValueError</code> – If any field name contains characters outside `[A-Za-z0-9_.]`.

#### get_metadata_fields_info

```python
get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Return a mapping of metadata field names to their detected types.

Uses Oracle's `JSON_DATAGUIDE` aggregate to introspect the stored metadata column.
Returns an empty dict when the table has no documents.

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – Dict of the form `{"field_name": {"type": "<type>"}, ...}` where `<type>`
  is one of `"text"`, `"number"`, or `"boolean"`.

#### get_metadata_field_min_max

```python
get_metadata_field_min_max(metadata_field: str) -> dict[str, Any]
```

Return the minimum and maximum values of a metadata field across all documents.

First attempts numeric comparison via `TO_NUMBER` so that `MAX(1, 5, 10)` returns `10`
rather than `"5"` (which would win under lexicographic ordering). Falls back to plain string
comparison when the field contains non-numeric values. Numeric strings are automatically
converted to `int` or `float` in the result.

**Parameters:**

- **metadata_field** (<code>str</code>) – Metadata field name. May be prefixed with `"meta."`
  (e.g. `"meta.year"` or `"year"`).

**Returns:**

- <code>dict\[str, Any\]</code> – `{"min": <value>, "max": <value>}`. Both values are `None` when the table is
  empty or the field does not exist.

**Raises:**

- <code>ValueError</code> – If `metadata_field` contains characters outside `[A-Za-z0-9_.]`.

#### get_metadata_field_unique_values

```python
get_metadata_field_unique_values(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int | None = None,
) -> tuple[list[str], int]
```

Return a paginated list of distinct values for a metadata field, plus the total distinct count.

**Parameters:**

- **metadata_field** (<code>str</code>) – Metadata field name. May be prefixed with `"meta."`
  (e.g. `"meta.lang"` or `"lang"`).
- **search_term** (<code>str | None</code>) – Optional substring filter applied to both the document text and the field value.
- **from\_** (<code>int</code>) – Zero-based offset for pagination. Defaults to `0`.
- **size** (<code>int | None</code>) – Maximum number of values to return. When `None` all values from `from_` onward
  are returned.

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – A tuple `(values, total)` where `values` is the paginated list of distinct field
  values as strings and `total` is the overall distinct count (before pagination).

**Raises:**

- <code>ValueError</code> – If `metadata_field` contains characters outside `[A-Za-z0-9_.]`.

#### get_metadata_fields_info_async

```python
get_metadata_fields_info_async() -> dict[str, dict[str, str]]
```

Asynchronously returns a mapping of metadata field names to their detected types.

Uses Oracle's `JSON_DATAGUIDE` aggregate to introspect the stored metadata column.
Returns an empty dict when the table has no documents.

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – Dict of the form `{"field_name": {"type": "<type>"}, ...}` where `<type>`
  is one of `"text"`, `"number"`, or `"boolean"`.

#### get_metadata_field_min_max_async

```python
get_metadata_field_min_max_async(metadata_field: str) -> dict[str, Any]
```

Asynchronously returns the minimum and maximum values of a metadata field across all documents.

First attempts numeric comparison via `TO_NUMBER`, falling back to string comparison for
non-numeric fields. Numeric strings are automatically converted to `int` or `float`.

**Parameters:**

- **metadata_field** (<code>str</code>) – Metadata field name. May be prefixed with `"meta."`
  (e.g. `"meta.year"` or `"year"`).

**Returns:**

- <code>dict\[str, Any\]</code> – `{"min": <value>, "max": <value>}`. Both values are `None` when the table is
  empty or the field does not exist.

**Raises:**

- <code>ValueError</code> – If `metadata_field` contains characters outside `[A-Za-z0-9_.]`.

#### get_metadata_field_unique_values_async

```python
get_metadata_field_unique_values_async(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int | None = None,
) -> tuple[list[str], int]
```

Asynchronously returns a paginated list of distinct values for a metadata field, plus the total count.

**Parameters:**

- **metadata_field** (<code>str</code>) – Metadata field name. May be prefixed with `"meta."`
  (e.g. `"meta.lang"` or `"lang"`).
- **search_term** (<code>str | None</code>) – Optional substring filter applied to both the document text and the field value.
- **from\_** (<code>int</code>) – Zero-based offset for pagination. Defaults to `0`.
- **size** (<code>int | None</code>) – Maximum number of values to return. When `None` all values from `from_` onward
  are returned.

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – A tuple `(values, total)` where `values` is the paginated list of distinct field
  values as strings and `total` is the overall distinct count (before pagination).

**Raises:**

- <code>ValueError</code> – If `metadata_field` contains characters outside `[A-Za-z0-9_.]`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OracleDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>OracleDocumentStore</code> – Deserialized component.
