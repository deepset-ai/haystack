---
title: "Pgvector"
id: integrations-pgvector
description: "Pgvector integration for Haystack"
slug: "/integrations-pgvector"
---


## `haystack_integrations.components.retrievers.pgvector.embedding_retriever`

### `PgvectorEmbeddingRetriever`

Retrieves documents from the `PgvectorDocumentStore`, based on their dense embeddings.

Example usage:

```python
from haystack.document_stores import DuplicatePolicy
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever

# Set an environment variable `PG_CONN_STR` with the connection string to your PostgreSQL database.
# e.g., "postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"

document_store = PgvectorDocumentStore(
    embedding_dimension=768,
    vector_function="cosine_similarity",
    recreate_table=True,
)

documents = [Document(content="There are over 7,000 languages spoken around the world today."),
             Document(content="Elephants have been observed to behave in a way that indicates..."),
             Document(content="In certain places, you can witness the phenomenon of bioluminescent waves.")]

document_embedder = SentenceTransformersDocumentEmbedder()
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)

document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
query_pipeline.add_component("retriever", PgvectorEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "How many languages are there?"

res = query_pipeline.run({"text_embedder": {"text": query}})

assert res['retriever']['documents'][0].content == "There are over 7,000 languages spoken around the world today."
```

#### `__init__`

```python
__init__(
    *,
    document_store: PgvectorDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    vector_function: (
        Literal["cosine_similarity", "inner_product", "l2_distance"] | None
    ) = None,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
)
```

**Parameters:**

- **document_store** (<code>PgvectorDocumentStore</code>) – An instance of `PgvectorDocumentStore`.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents.
- **top_k** (<code>int</code>) – Maximum number of Documents to return.
- **vector_function** (<code>Literal['cosine_similarity', 'inner_product', 'l2_distance'] | None</code>) – The similarity function to use when searching for similar embeddings.
  Defaults to the one set in the `document_store` instance.
  `"cosine_similarity"` and `"inner_product"` are similarity functions and
  higher scores indicate greater similarity between the documents.
  `"l2_distance"` returns the straight-line distance between vectors,
  and the most similar documents are the ones with the smallest score.
  **Important**: if the document store is using the `"hnsw"` search strategy, the vector function
  should match the one utilized during index creation to take advantage of the index.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of `PgvectorDocumentStore` or if `vector_function`
  is not one of the valid options.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> PgvectorEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>PgvectorEmbeddingRetriever</code> – Deserialized component.

#### `run`

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

Retrieve documents from the `PgvectorDocumentStore`, based on their embeddings.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – Maximum number of Documents to return.
- **vector_function** (<code>Literal['cosine_similarity', 'inner_product', 'l2_distance'] | None</code>) – The similarity function to use when searching for similar embeddings.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of `Document`s that are similar to `query_embedding`.

#### `run_async`

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    vector_function: (
        Literal["cosine_similarity", "inner_product", "l2_distance"] | None
    ) = None,
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents from the `PgvectorDocumentStore`, based on their embeddings.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – Maximum number of Documents to return.
- **vector_function** (<code>Literal['cosine_similarity', 'inner_product', 'l2_distance'] | None</code>) – The similarity function to use when searching for similar embeddings.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of `Document`s that are similar to `query_embedding`.

## `haystack_integrations.components.retrievers.pgvector.keyword_retriever`

### `PgvectorKeywordRetriever`

Retrieve documents from the `PgvectorDocumentStore`, based on keywords.

To rank the documents, the `ts_rank_cd` function of PostgreSQL is used.
It considers how often the query terms appear in the document, how close together the terms are in the document,
and how important is the part of the document where they occur.
For more details, see
[Postgres documentation](https://www.postgresql.org/docs/current/textsearch-controls.html#TEXTSEARCH-RANKING).

Usage example:

````python
from haystack.document_stores import DuplicatePolicy
from haystack import Document

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorKeywordRetriever

# Set an environment variable `PG_CONN_STR` with the connection string to your PostgreSQL database.
# e.g., "postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"

document_store = PgvectorDocumentStore(language="english", recreate_table=True)

documents = [Document(content="There are over 7,000 languages spoken around the world today."),
    Document(content="Elephants have been observed to behave in a way that indicates..."),
    Document(content="In certain places, you can witness the phenomenon of bioluminescent waves.")]

document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)

retriever = PgvectorKeywordRetriever(document_store=document_store)

result = retriever.run(query="languages")

assert res['retriever']['documents'][0].content == "There are over 7,000 languages spoken around the world today."












#### `__init__`

```python
__init__(
    *,
    document_store: PgvectorDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
)
````

**Parameters:**

- **document_store** (<code>PgvectorDocumentStore</code>) – An instance of `PgvectorDocumentStore`.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents.
- **top_k** (<code>int</code>) – Maximum number of Documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of `PgvectorDocumentStore`.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> PgvectorKeywordRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>PgvectorKeywordRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, list[Document]]
```

Retrieve documents from the `PgvectorDocumentStore`, based on keywords.

**Parameters:**

- **query** (<code>str</code>) – String to search in `Document`s' content.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – Maximum number of Documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of `Document`s that match the query.

#### `run_async`

```python
run_async(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents from the `PgvectorDocumentStore`, based on keywords.

**Parameters:**

- **query** (<code>str</code>) – String to search in `Document`s' content.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – Maximum number of Documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of `Document`s that match the query.

## `haystack_integrations.document_stores.pgvector.document_store`

### `PgvectorDocumentStore`

A Document Store using PostgreSQL with the [pgvector extension](https://github.com/pgvector/pgvector) installed.

#### `__init__`

```python
__init__(
    *,
    connection_string: Secret = Secret.from_env_var("PG_CONN_STR"),
    create_extension: bool = True,
    schema_name: str = "public",
    table_name: str = "haystack_documents",
    language: str = "english",
    embedding_dimension: int = 768,
    vector_type: Literal["vector", "halfvec"] = "vector",
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
)
```

Creates a new PgvectorDocumentStore instance.
It is meant to be connected to a PostgreSQL database with the pgvector extension installed.
A specific table to store Haystack documents will be created if it doesn't exist yet.

**Parameters:**

- **connection_string** (<code>Secret</code>) – The connection string to use to connect to the PostgreSQL database, defined as an
  environment variable. Supported formats:
- URI, e.g. `PG_CONN_STR="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"` (use percent-encoding for special
  characters)
- keyword/value format, e.g. `PG_CONN_STR="host=HOST port=PORT dbname=DBNAME user=USER password=PASSWORD"`
  See [PostgreSQL Documentation](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING)
  for more details.
- **create_extension** (<code>bool</code>) – Whether to create the pgvector extension if it doesn't exist.
  Set this to `True` (default) to automatically create the extension if it is missing.
  Creating the extension may require superuser privileges.
  If set to `False`, ensure the extension is already installed; otherwise, an error will be raised.
- **schema_name** (<code>str</code>) – The name of the schema the table is created in. The schema must already exist.
- **table_name** (<code>str</code>) – The name of the table to use to store Haystack documents.
- **language** (<code>str</code>) – The language to be used to parse query and document content in keyword retrieval.
  To see the list of available languages, you can run the following SQL query in your PostgreSQL database:
  `SELECT cfgname FROM pg_ts_config;`.
  More information can be found in this [StackOverflow answer](https://stackoverflow.com/a/39752553).
- **embedding_dimension** (<code>int</code>) – The dimension of the embedding.
- **vector_type** (<code>Literal['vector', 'halfvec']</code>) – The type of vector used for embedding storage.
  "vector" is the default.
  "halfvec" stores embeddings in half-precision, which is particularly useful for high-dimensional embeddings
  (dimension greater than 2,000 and up to 4,000). Requires pgvector versions 0.7.0 or later. For more
  information, see the [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file).
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
  Only used if search_strategy is set to `"hnsw"`. You can find the list of valid arguments in the
  [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw)
- **hnsw_index_name** (<code>str</code>) – Index name for the HNSW index.
- **hnsw_ef_search** (<code>int | None</code>) – The `ef_search` parameter to use at query time. Only used if search_strategy is set to
  `"hnsw"`. You can find more information about this parameter in the
  [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw).
- **keyword_index_name** (<code>str</code>) – Index name for the Keyword index.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> PgvectorDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>PgvectorDocumentStore</code> – Deserialized component.

#### `delete_table`

```python
delete_table()
```

Deletes the table used to store Haystack documents.
The name of the schema (`schema_name`) and the name of the table (`table_name`)
are defined when initializing the `PgvectorDocumentStore`.

#### `delete_table_async`

```python
delete_table_async()
```

Async method to delete the table used to store Haystack documents.

#### `count_documents`

```python
count_documents() -> int
```

Returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – Number of documents in the document store.

#### `count_documents_async`

```python
count_documents_async() -> int
```

Returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – Number of documents in the document store.

#### `filter_documents`

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

**Raises:**

- <code>TypeError</code> – If `filters` is not a dictionary.
- <code>ValueError</code> – If `filters` syntax is invalid.

#### `filter_documents_async`

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

**Raises:**

- <code>TypeError</code> – If `filters` is not a dictionary.
- <code>ValueError</code> – If `filters` syntax is invalid.

#### `write_documents`

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

- <code>ValueError</code> – If `documents` contains objects that are not of type `Document`.
- <code>DuplicateDocumentError</code> – If a document with the same id already exists in the document store
  and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
- <code>DocumentStoreError</code> – If the write operation fails for any other reason.

#### `write_documents_async`

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

- <code>ValueError</code> – If `documents` contains objects that are not of type `Document`.
- <code>DuplicateDocumentError</code> – If a document with the same id already exists in the document store
  and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
- <code>DocumentStoreError</code> – If the write operation fails for any other reason.

#### `delete_documents`

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes documents that match the provided `document_ids` from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete

#### `delete_documents_async`

```python
delete_documents_async(document_ids: list[str]) -> None
```

Asynchronously deletes documents that match the provided `document_ids` from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete

#### `delete_all_documents`

```python
delete_all_documents() -> None
```

Deletes all documents in the document store.

#### `delete_all_documents_async`

```python
delete_all_documents_async() -> None
```

Asynchronously deletes all documents in the document store.

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

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update.

**Returns:**

- <code>int</code> – The number of documents updated.

#### `update_by_filter_async`

```python
update_by_filter_async(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Asynchronously updates the metadata of all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update.

**Returns:**

- <code>int</code> – The number of documents updated.

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

Returns the count of unique values for each specified metadata field,
considering only documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **metadata_fields** (<code>list\[str\]</code>) – List of metadata field names to count unique values for.
  Field names can include or omit the "meta." prefix.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary mapping field names to their unique value counts.

#### `count_unique_metadata_by_filter_async`

```python
count_unique_metadata_by_filter_async(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Asynchronously returns the count of unique values for each specified metadata field,
considering only documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **metadata_fields** (<code>list\[str\]</code>) – List of metadata field names to count unique values for.
  Field names can include or omit the "meta." prefix.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary mapping field names to their unique value counts.

#### `get_metadata_fields_info`

```python
get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Returns the information about the metadata fields in the document store.

Since metadata is stored in a JSONB field, this method analyzes actual data
to infer field types.

Example return:

```python
{
    'content': {'type': 'text'},
    'category': {'type': 'text'},
    'status': {'type': 'text'},
    'priority': {'type': 'integer'},
}
```

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – A dictionary mapping field names to their type information.

#### `get_metadata_fields_info_async`

```python
get_metadata_fields_info_async() -> dict[str, dict[str, str]]
```

Asynchronously returns the information about the metadata fields in the document store.

Since metadata is stored in a JSONB field, this method analyzes actual data
to infer field types.

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – A dictionary mapping field names to their type information.

#### `get_metadata_field_min_max`

```python
get_metadata_field_min_max(metadata_field: str) -> dict[str, Any]
```

Returns the minimum and maximum values for a given metadata field.

**Parameters:**

- **metadata_field** (<code>str</code>) – The name of the metadata field. Can include or omit the "meta." prefix.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with 'min' and 'max' keys containing the minimum and maximum values.
  For numeric fields (integer, real), returns numeric min/max.
  For text fields, returns lexicographic min/max based on database collation.

**Raises:**

- <code>ValueError</code> – If the field doesn't exist or has no values.

#### `get_metadata_field_min_max_async`

```python
get_metadata_field_min_max_async(metadata_field: str) -> dict[str, Any]
```

Asynchronously returns the minimum and maximum values for a given metadata field.

**Parameters:**

- **metadata_field** (<code>str</code>) – The name of the metadata field. Can include or omit the "meta." prefix.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with 'min' and 'max' keys containing the minimum and maximum values.
  For numeric fields (integer, real), returns numeric min/max.
  For text fields, returns lexicographic min/max based on database collation.

**Raises:**

- <code>ValueError</code> – If the field doesn't exist or has no values.

#### `get_metadata_field_unique_values`

```python
get_metadata_field_unique_values(
    metadata_field: str, search_term: str | None, from_: int, size: int
) -> tuple[list[str], int]
```

Returns unique values for a given metadata field, optionally filtered by a search term.

**Parameters:**

- **metadata_field** (<code>str</code>) – The name of the metadata field. Can include or omit the "meta." prefix.
- **search_term** (<code>str | None</code>) – Optional search term to filter documents by content before extracting unique values.
  If None, all documents are considered.
- **from\_** (<code>int</code>) – The offset for pagination (0-based).
- **size** (<code>int</code>) – The number of unique values to return.

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – A tuple containing:
- A list of unique values (as strings)
- The total count of unique values

#### `get_metadata_field_unique_values_async`

```python
get_metadata_field_unique_values_async(
    metadata_field: str, search_term: str | None, from_: int, size: int
) -> tuple[list[str], int]
```

Asynchronously returns unique values for a given metadata field, optionally filtered by a search term.

**Parameters:**

- **metadata_field** (<code>str</code>) – The name of the metadata field. Can include or omit the "meta." prefix.
- **search_term** (<code>str | None</code>) – Optional search term to filter documents by content before extracting unique values.
  If None, all documents are considered.
- **from\_** (<code>int</code>) – The offset for pagination (0-based).
- **size** (<code>int</code>) – The number of unique values to return.

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – A tuple containing:
- A list of unique values (as strings)
- The total count of unique values
