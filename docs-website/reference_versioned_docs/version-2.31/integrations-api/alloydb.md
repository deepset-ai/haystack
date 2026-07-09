---
title: "AlloyDB"
id: integrations-alloydb
description: "AlloyDB integration for Haystack"
slug: "/integrations-alloydb"
---


## haystack_integrations.components.retrievers.alloydb.embedding_retriever

### AlloyDBEmbeddingRetriever

Retrieves documents from the `AlloyDBDocumentStore` by embedding similarity.

Must be connected to the `AlloyDBDocumentStore`.

#### __init__

```python
__init__(
    *,
    document_store: AlloyDBDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    vector_function: (
        Literal["cosine_similarity", "inner_product", "l2_distance"] | None
    ) = None,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Create the `AlloyDBEmbeddingRetriever` component.

**Parameters:**

- **document_store** (<code>AlloyDBDocumentStore</code>) – An instance of `AlloyDBDocumentStore` to use as the document store.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved documents.
- **top_k** (<code>int</code>) – Maximum number of documents to return.
- **vector_function** (<code>Literal['cosine_similarity', 'inner_product', 'l2_distance'] | None</code>) – The similarity function to use when searching for similar embeddings.
  Overrides the `vector_function` set in the `AlloyDBDocumentStore`.
  `"cosine_similarity"` and `"inner_product"` are similarity functions and
  higher scores indicate greater similarity between the documents.
  `"l2_distance"` returns the straight-line distance between vectors,
  and the most similar documents are the ones with the smallest score.
  **Important**: when using the `"hnsw"` search strategy, make sure to use the same
  vector function as the one used when the HNSW index was created.
  If not specified, the `vector_function` of the `AlloyDBDocumentStore` is used.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied at query time.
  `FilterPolicy.REPLACE` (default) replaces the init filters with the run-time filters.
  `FilterPolicy.MERGE` merges the init filters with the run-time filters.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of `AlloyDBDocumentStore`.

#### run

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    vector_function: (
        Literal["cosine_similarity", "inner_product", "l2_distance"] | None
    ) = None,
) -> dict[str, list[Document]]
```

Retrieve documents from the `AlloyDBDocumentStore` by embedding similarity.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – A vector representation of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved documents.
  The `filter_policy` set at initialization determines how these are combined with the init filters.
- **top_k** (<code>int | None</code>) – Maximum number of documents to return. Overrides the `top_k` set at initialization.
- **vector_function** (<code>Literal['cosine_similarity', 'inner_product', 'l2_distance'] | None</code>) – The similarity function to use when searching for similar embeddings.
  Overrides the `vector_function` set at initialization.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary containing the `documents` retrieved from the document store.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AlloyDBEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AlloyDBEmbeddingRetriever</code> – Deserialized component.

## haystack_integrations.components.retrievers.alloydb.keyword_retriever

### AlloyDBKeywordRetriever

Retrieves documents from the `AlloyDBDocumentStore` by keyword search.

Uses PostgreSQL full-text search (`to_tsvector` / `plainto_tsquery`) to find documents.
Must be connected to the `AlloyDBDocumentStore`.

#### __init__

```python
__init__(
    *,
    document_store: AlloyDBDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Create the `AlloyDBKeywordRetriever` component.

**Parameters:**

- **document_store** (<code>AlloyDBDocumentStore</code>) – An instance of `AlloyDBDocumentStore` to use as the document store.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved documents.
- **top_k** (<code>int</code>) – Maximum number of documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied at query time.
  `FilterPolicy.REPLACE` (default) replaces the init filters with the run-time filters.
  `FilterPolicy.MERGE` merges the init filters with the run-time filters.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of `AlloyDBDocumentStore`.

#### run

```python
run(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, list[Document]]
```

Retrieve documents from the `AlloyDBDocumentStore` by keyword search.

**Parameters:**

- **query** (<code>str</code>) – A keyword query to search for.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved documents.
  The `filter_policy` set at initialization determines how these are combined with the init filters.
- **top_k** (<code>int | None</code>) – Maximum number of documents to return. Overrides the `top_k` set at initialization.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary containing the `documents` retrieved from the document store.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AlloyDBKeywordRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AlloyDBKeywordRetriever</code> – Deserialized component.

## haystack_integrations.document_stores.alloydb.document_store

### AlloyDBDocumentStore

Bases: <code>DocumentStore</code>

A Document Store backed by [Google Cloud AlloyDB](https://cloud.google.com/alloydb).

Uses the [pgvector extension](https://cloud.google.com/alloydb/docs/ai/work-with-embeddings) for vector search.

AlloyDB is a fully managed, PostgreSQL-compatible database service on Google Cloud.
Connection is handled securely via the
[AlloyDB Python Connector](https://github.com/GoogleCloudPlatform/alloydb-python-connector),
which provides TLS encryption and IAM-based authorization without requiring manual SSL certificate
management, firewall rules, or IP allowlisting.

**Filter limitations**: the `NOT` logical operator is not supported. Use `!=` or `not in`
comparison operators to express negation.

Usage example:

```python
import os
from haystack_integrations.document_stores.alloydb import AlloyDBDocumentStore

# Set required environment variables:
# ALLOYDB_INSTANCE_URI = "projects/MY_PROJECT/locations/MY_REGION/clusters/MY_CLUSTER/instances/MY_INSTANCE"
# ALLOYDB_USER = "my-db-user"
# ALLOYDB_PASSWORD = "my-db-password"

document_store = AlloyDBDocumentStore(
    db="my-database",
    embedding_dimension=768,
    recreate_table=True,
)
```

#### __init__

```python
__init__(
    *,
    instance_uri: Secret = Secret.from_env_var("ALLOYDB_INSTANCE_URI"),
    user: Secret = Secret.from_env_var("ALLOYDB_USER"),
    password: Secret = Secret.from_env_var("ALLOYDB_PASSWORD", strict=False),
    db: str = "postgres",
    enable_iam_auth: bool = False,
    ip_type: Literal["PRIVATE", "PUBLIC", "PSC"] = "PRIVATE",
    create_extension: bool = True,
    schema_name: str = "public",
    table_name: str = "haystack_documents",
    language: str = "english",
    embedding_dimension: int = 768,
    vector_function: Literal[
        "cosine_similarity", "inner_product", "l2_distance"
    ] = "cosine_similarity",
    recreate_table: bool = False,
    search_strategy: Literal[
        "exact_nearest_neighbor", "hnsw"
    ] = "exact_nearest_neighbor",
    hnsw_recreate_index_if_exists: bool = False,
    hnsw_index_creation_kwargs: dict[str, int] | None = None,
    hnsw_index_name: str = "haystack_hnsw_index",
    hnsw_ef_search: int | None = None,
    keyword_index_name: str = "haystack_keyword_index"
) -> None
```

Creates a new AlloyDBDocumentStore instance.

Connection to AlloyDB is established lazily on first use via the AlloyDB Python Connector.
A specific table to store Haystack documents will be created if it doesn't exist yet.

**Parameters:**

- **instance_uri** (<code>Secret</code>) – The AlloyDB instance URI in the format
  `"projects/PROJECT/locations/REGION/clusters/CLUSTER/instances/INSTANCE"`.
  Read from the `ALLOYDB_INSTANCE_URI` environment variable by default.
- **user** (<code>Secret</code>) – The database user. Read from the `ALLOYDB_USER` environment variable by default.
  When using IAM database authentication, use the service account email (omitting
  `.gserviceaccount.com`) or the full IAM user email.
- **password** (<code>Secret</code>) – The database password. Read from the `ALLOYDB_PASSWORD` environment variable by default.
  Not required when `enable_iam_auth=True`.
- **db** (<code>str</code>) – The name of the database to connect to. Defaults to `"postgres"`.
- **enable_iam_auth** (<code>bool</code>) – Whether to use IAM database authentication instead of a password.
  When `True`, `password` is ignored. The IAM principal must be granted the
  AlloyDB Client role and have an IAM database user created.
  See the [AlloyDB documentation](https://cloud.google.com/alloydb/docs/manage-iam-authn) for details.
- **ip_type** (<code>Literal['PRIVATE', 'PUBLIC', 'PSC']</code>) – The IP address type to use for the connection.
  `"PRIVATE"` (default) connects over a private VPC IP.
  `"PUBLIC"` connects over a public IP.
  `"PSC"` connects via Private Service Connect.
- **create_extension** (<code>bool</code>) – Whether to create the pgvector extension if it doesn't exist.
  Set this to `True` (default) to automatically create the extension if it is missing.
  Creating the extension may require superuser privileges.
  If set to `False`, ensure the extension is already installed; otherwise, an error will be raised.
- **schema_name** (<code>str</code>) – The name of the schema the table is created in. The schema must already exist.
- **table_name** (<code>str</code>) – The name of the table to use to store Haystack documents.
- **language** (<code>str</code>) – The language to be used to parse query and document content in keyword retrieval.
  To see the list of available languages, you can run the following SQL query in your PostgreSQL database:
  `SELECT cfgname FROM pg_ts_config;`.
- **embedding_dimension** (<code>int</code>) – The dimension of the embedding.
- **vector_function** (<code>Literal['cosine_similarity', 'inner_product', 'l2_distance']</code>) – The similarity function to use when searching for similar embeddings.
  `"cosine_similarity"` and `"inner_product"` are similarity functions and
  higher scores indicate greater similarity between the documents.
  `"l2_distance"` returns the straight-line distance between vectors,
  and the most similar documents are the ones with the smallest score.
  **Important**: when using the `"hnsw"` search strategy, an index will be created that depends on the
  `vector_function` passed here. Make sure subsequent queries will keep using the same
  vector similarity function in order to take advantage of the index.
- **recreate_table** (<code>bool</code>) – Whether to recreate the table if it already exists.
- **search_strategy** (<code>Literal['exact_nearest_neighbor', 'hnsw']</code>) – The search strategy to use when searching for similar embeddings.
  `"exact_nearest_neighbor"` provides perfect recall but can be slow for large numbers of documents.
  `"hnsw"` is an approximate nearest neighbor search strategy,
  which trades off some accuracy for speed; it is recommended for large numbers of documents.
  **Important**: when using the `"hnsw"` search strategy, an index will be created that depends on the
  `vector_function` passed here. Make sure subsequent queries will keep using the same
  vector similarity function in order to take advantage of the index.
- **hnsw_recreate_index_if_exists** (<code>bool</code>) – Whether to recreate the HNSW index if it already exists.
  Only used if search_strategy is set to `"hnsw"`.
- **hnsw_index_creation_kwargs** (<code>dict\[str, int\] | None</code>) – Additional keyword arguments to pass to the HNSW index creation.
  Only used if search_strategy is set to `"hnsw"`. Valid arguments are `m` and `ef_construction`.
  See the [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw) for details.
- **hnsw_index_name** (<code>str</code>) – Index name for the HNSW index.
- **hnsw_ef_search** (<code>int | None</code>) – The `ef_search` parameter to use at query time. Only used if search_strategy is set to
  `"hnsw"`. See the [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw).
- **keyword_index_name** (<code>str</code>) – Index name for the keyword GIN index.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AlloyDBDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AlloyDBDocumentStore</code> – Deserialized component.

#### close

```python
close() -> None
```

Closes the database connection and the AlloyDB connector.

Call this when you are done using the document store to release resources.
For long-lived applications the connector runs a background refresh thread;
calling `close()` ensures that thread is stopped cleanly.

#### delete_table

```python
delete_table() -> None
```

Deletes the table used to store Haystack documents.

The name of the schema (`schema_name`) and the name of the table (`table_name`)
are defined when initializing the `AlloyDBDocumentStore`.

#### count_documents

```python
count_documents() -> int
```

Returns how many documents are in the document store.

**Returns:**

- <code>int</code> – The number of documents in the document store.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Filter operator support**: comparison operators (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`,
`not in`, `like`, `not like`) and logical operators `AND` and `OR` are fully supported.
The `NOT` logical operator is **not** supported — use `!=` or `not in` comparison
operators instead.

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
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL
) -> int
```

Writes documents to the document store.

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

Deletes documents that match the provided `document_ids` from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete

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

Returns the number of documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to count documents.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns:**

- <code>int</code> – The number of documents that match the filters.

#### count_unique_metadata_by_filter

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Returns the count of unique values for each specified metadata field.

Considers only documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **metadata_fields** (<code>list\[str\]</code>) – List of metadata field names to count unique values for.
  Field names can include or omit the "meta." prefix.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary mapping field names to their unique value counts.

#### get_metadata_fields_info

```python
get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Returns information about the metadata fields in the document store.

Since metadata is stored in a JSONB field, this method analyzes actual data
to infer field types.

Example return:

```python
{
    'category': {'type': 'text'},
    'priority': {'type': 'integer'},
}
```

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – A dictionary mapping field names to their type information.

#### get_metadata_field_min_max

```python
get_metadata_field_min_max(field: str) -> dict[str, Any]
```

Returns the minimum and maximum values for a metadata field.

For numeric fields (integer, real), returns numeric min/max.
For text and other non-numeric fields, returns lexicographic min/max
using the `"C"` collation.

**Parameters:**

- **field** (<code>str</code>) – The metadata field name (with or without the "meta." prefix).

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with `min` and `max` keys. Returns
  `{"min": None, "max": None}` when the field has no values or the
  store is empty.

#### get_metadata_field_unique_values

```python
get_metadata_field_unique_values(
    field: str, filters: dict[str, Any] | None = None
) -> list[Any]
```

Returns a list of unique values for a metadata field.

**Parameters:**

- **field** (<code>str</code>) – The metadata field name (with or without the "meta." prefix).
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional filters to restrict the documents considered.

**Returns:**

- <code>list\[Any\]</code> – A list of unique values for the given field.
