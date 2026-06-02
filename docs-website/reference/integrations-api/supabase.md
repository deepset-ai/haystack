---
title: "Supabase"
id: integrations-supabase
description: "Supabase integration for Haystack"
slug: "/integrations-supabase"
---


## haystack_integrations.components.downloaders.supabase.supabase_bucket_downloader

### SupabaseBucketDownloader

Downloads files from a Supabase Storage bucket and returns them as ByteStream objects.

Files are downloaded in-memory and returned as `ByteStream` objects ready for further
processing in indexing pipelines (e.g. passing to a `DocumentConverter`).

Example usage:

```python
from haystack_integrations.components.downloaders.supabase import SupabaseBucketDownloader
from haystack.utils import Secret

downloader = SupabaseBucketDownloader(
    supabase_url="https://<project-ref>.supabase.co",
    supabase_key=Secret.from_env_var("SUPABASE_SERVICE_KEY"),
    bucket_name="my-documents",
)
result = downloader.run(sources=["reports/report.pdf", "data/notes.txt"])
streams = result["streams"]
```

#### __init__

```python
__init__(
    *,
    supabase_url: str,
    supabase_key: Secret = Secret.from_env_var("SUPABASE_SERVICE_KEY"),
    bucket_name: str,
    file_extensions: list[str] | None = None
) -> None
```

Creates a new SupabaseBucketDownloader instance.

**Parameters:**

- **supabase_url** (<code>str</code>) – The URL of your Supabase project, e.g. `https://<project-ref>.supabase.co`.
- **supabase_key** (<code>Secret</code>) – The Supabase API key used to authenticate requests. Defaults to the
  `SUPABASE_SERVICE_KEY` environment variable. Use the service role key for private buckets.
- **bucket_name** (<code>str</code>) – The name of the Supabase Storage bucket to download files from.
- **file_extensions** (<code>list\[str\] | None</code>) – Optional list of file extensions to filter downloads (e.g. `[".pdf", ".txt"]`).
  If `None`, all files are downloaded. Extensions are matched case-insensitively.

#### warm_up

```python
warm_up() -> None
```

Initializes the Supabase client.

Called automatically on the first run(), or can be called explicitly in a pipeline.

#### run

```python
run(sources: list[str]) -> dict[str, list[ByteStream]]
```

Downloads files from the Supabase Storage bucket.

**Parameters:**

- **sources** (<code>list\[str\]</code>) – List of file paths within the bucket to download,
  e.g. `["folder/file.pdf", "notes.txt"]`.

**Returns:**

- <code>dict\[str, list\[ByteStream\]\]</code> – A dictionary with:
- `streams`: list of `ByteStream` objects, one per successfully downloaded file.
  Each `ByteStream` has `meta["file_path"]` and `meta["bucket_name"]` set.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SupabaseBucketDownloader
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SupabaseBucketDownloader</code> – Deserialized component.

## haystack_integrations.components.retrievers.supabase.embedding_retriever

### SupabasePgvectorEmbeddingRetriever

Bases: <code>PgvectorEmbeddingRetriever</code>

Retrieves documents from the `SupabasePgvectorDocumentStore`, based on their dense embeddings.

This is a thin wrapper around `PgvectorEmbeddingRetriever`, adapted for use with
`SupabasePgvectorDocumentStore`.

Example usage:

# Set an environment variable `SUPABASE_DB_URL` with the connection string to your Supabase database.

```bash
export SUPABASE_DB_URL=postgresql://postgres:postgres@localhost:5432/postgres
```

```python
from haystack import Document, Pipeline
from haystack.document_stores.types.policy import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder

from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore
from haystack_integrations.components.retrievers.supabase import SupabasePgvectorEmbeddingRetriever

document_store = SupabasePgvectorDocumentStore(
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
query_pipeline.add_component("retriever", SupabasePgvectorEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "How many languages are there?"

res = query_pipeline.run({"text_embedder": {"text": query}})
print(res['retriever']['documents'][0].content)
# >> "There are over 7,000 languages spoken around the world today."
```

#### __init__

```python
__init__(
    *,
    document_store: SupabasePgvectorDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    vector_function: (
        Literal["cosine_similarity", "inner_product", "l2_distance"] | None
    ) = None,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Initialize the SupabasePgvectorEmbeddingRetriever.

**Parameters:**

- **document_store** (<code>SupabasePgvectorDocumentStore</code>) – An instance of `SupabasePgvectorDocumentStore`.
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

- <code>ValueError</code> – If `document_store` is not an instance of `SupabasePgvectorDocumentStore` or if
  `vector_function` is not one of the valid options.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SupabasePgvectorEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SupabasePgvectorEmbeddingRetriever</code> – Deserialized component.

## haystack_integrations.components.retrievers.supabase.groonga_bm25_retriever

### SupabaseGroongaBM25Retriever

Retrieves documents from SupabaseGroongaDocumentStore using PGroonga full-text search.

This retriever works without embeddings — it searches documents using plain text queries.
It can be used alongside SupabasePgvectorEmbeddingRetriever in hybrid search pipelines.

Note: async operations are not supported as the supabase-py sync client does not expose
awaitable query methods. Use the sync run() method instead.

Example usage:

```python
from haystack_integrations.document_stores.supabase import SupabaseGroongaDocumentStore
from haystack_integrations.components.retrievers.supabase import SupabaseGroongaBM25Retriever
from haystack.utils import Secret

document_store = SupabaseGroongaDocumentStore(
    supabase_url="https://<project>.supabase.co",
    supabase_key=Secret.from_env_var("SUPABASE_SERVICE_KEY"),
    table_name="haystack_fts_documents",
)
document_store.warm_up()

retriever = SupabaseGroongaBM25Retriever(document_store=document_store, top_k=10)
result = retriever.run(query="python programming")
print(result["documents"])
```

#### __init__

```python
__init__(
    *,
    document_store: SupabaseGroongaDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Initialize the SupabaseGroongaBM25Retriever.

**Parameters:**

- **document_store** (<code>SupabaseGroongaDocumentStore</code>) – An instance of SupabaseGroongaDocumentStore.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional filters applied to retrieved Documents.
- **top_k** (<code>int</code>) – Maximum number of Documents to return. Defaults to 10.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

**Raises:**

- <code>ValueError</code> – If document_store is not an instance of SupabaseGroongaDocumentStore.

#### run

```python
run(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, list[Document]]
```

Runs the retriever on the given query.

**Parameters:**

- **query** (<code>str</code>) – The text query to search for.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional runtime filters. Merged or replaced based on filter_policy.
- **top_k** (<code>int | None</code>) – Optional override for maximum number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – Dictionary with key "documents" containing list of matching Documents.

#### run_async

```python
run_async(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, list[Document]]
```

Async version of run().

Note: supabase-py's sync client does not support native async queries.
This method runs the synchronous retrieval and returns the result.
For fully async support, consider using acreate_client() from supabase-py
and refactoring the document store accordingly.

**Parameters:**

- **query** (<code>str</code>) – The text query to search for.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional runtime filters. Merged or replaced based on filter_policy.
- **top_k** (<code>int | None</code>) – Optional override for maximum number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – Dictionary with key "documents" containing list of matching Documents.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SupabaseGroongaBM25Retriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SupabaseGroongaBM25Retriever</code> – Deserialized component.

## haystack_integrations.components.retrievers.supabase.keyword_retriever

### SupabasePgvectorKeywordRetriever

Bases: <code>PgvectorKeywordRetriever</code>

Retrieves documents from the `SupabasePgvectorDocumentStore`, based on keywords.

This is a thin wrapper around `PgvectorKeywordRetriever`, adapted for use with
`SupabasePgvectorDocumentStore`.

To rank the documents, the `ts_rank_cd` function of PostgreSQL is used.
It considers how often the query terms appear in the document, how close together the terms are in the document,
and how important is the part of the document where they occur.

Example usage:

# Set an environment variable `SUPABASE_DB_URL` with the connection string to your Supabase database.

```bash
export SUPABASE_DB_URL=postgresql://postgres:postgres@localhost:5432/postgres
```

```python
from haystack import Document, Pipeline
from haystack.document_stores.types.policy import DuplicatePolicy

from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore
from haystack_integrations.components.retrievers.supabase import SupabasePgvectorKeywordRetriever

document_store = SupabasePgvectorDocumentStore(
    embedding_dimension=768,
    recreate_table=True,
)

documents = [Document(content="There are over 7,000 languages spoken around the world today."),
             Document(content="Elephants have been observed to behave in a way that indicates..."),
             Document(content="In certain places, you can witness the phenomenon of bioluminescent waves.")]

document_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
retriever = SupabasePgvectorKeywordRetriever(document_store=document_store)
result = retriever.run(query="languages")

print(result['documents'][0].content)
# >> "There are over 7,000 languages spoken around the world today."
```

#### __init__

```python
__init__(
    *,
    document_store: SupabasePgvectorDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Initialize the SupabasePgvectorKeywordRetriever.

**Parameters:**

- **document_store** (<code>SupabasePgvectorDocumentStore</code>) – An instance of `SupabasePgvectorDocumentStore`.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents.
- **top_k** (<code>int</code>) – Maximum number of Documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of `SupabasePgvectorDocumentStore`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SupabasePgvectorKeywordRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SupabasePgvectorKeywordRetriever</code> – Deserialized component.

## haystack_integrations.document_stores.supabase.document_store

### SupabasePgvectorDocumentStore

Bases: <code>PgvectorDocumentStore</code>

A Document Store for Supabase, using PostgreSQL with the pgvector extension.

It should be used with Supabase installed.

This is a thin wrapper around `PgvectorDocumentStore` with Supabase-specific defaults:

- Reads the connection string from the `SUPABASE_DB_URL` environment variable.
- Defaults `create_extension` to `False` since pgvector is pre-installed on Supabase.

**Connection notes:** Supabase offers two pooler ports — transaction mode (6543) and session mode (5432).
For best compatibility with pgvector operations, use session mode (port 5432) or a direct connection.

Example usage:

# Set an environment variable `SUPABASE_DB_URL` with the connection string to your Supabase database.

```bash
export SUPABASE_DB_URL=postgresql://postgres:postgres@localhost:5432/postgres
```

```python
from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore

document_store = SupabasePgvectorDocumentStore(
    embedding_dimension=768,
    vector_function="cosine_similarity",
    recreate_table=True,
)
```

#### __init__

```python
__init__(
    *,
    connection_string: Secret = Secret.from_env_var("SUPABASE_DB_URL"),
    create_extension: bool = False,
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
) -> None
```

Creates a new SupabasePgvectorDocumentStore instance.

**Parameters:**

- **connection_string** (<code>Secret</code>) – The connection string for the Supabase PostgreSQL database, defined as an
  environment variable. Default: `SUPABASE_DB_URL`. Format:
  `postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres`
- **create_extension** (<code>bool</code>) – Whether to create the pgvector extension if it doesn't exist.
  Defaults to `False` since Supabase has pgvector pre-installed.
- **schema_name** (<code>str</code>) – The name of the schema the table is created in.
- **table_name** (<code>str</code>) – The name of the table to use to store Haystack documents.
- **language** (<code>str</code>) – The language to be used to parse query and document content in keyword retrieval.
- **embedding_dimension** (<code>int</code>) – The dimension of the embedding.
- **vector_type** (<code>Literal['vector', 'halfvec']</code>) – The type of vector used for embedding storage. `"vector"` or `"halfvec"`.
- **vector_function** (<code>Literal['cosine_similarity', 'inner_product', 'l2_distance']</code>) – The similarity function to use when searching for similar embeddings.
- **recreate_table** (<code>bool</code>) – Whether to recreate the table if it already exists.
- **search_strategy** (<code>Literal['exact_nearest_neighbor', 'hnsw']</code>) – The search strategy to use: `"exact_nearest_neighbor"` or `"hnsw"`.
- **hnsw_recreate_index_if_exists** (<code>bool</code>) – Whether to recreate the HNSW index if it already exists.
- **hnsw_index_creation_kwargs** (<code>dict\[str, int\] | None</code>) – Additional keyword arguments for HNSW index creation.
- **hnsw_index_name** (<code>str</code>) – Index name for the HNSW index.
- **hnsw_ef_search** (<code>int | None</code>) – The `ef_search` parameter to use at query time for HNSW.
- **keyword_index_name** (<code>str</code>) – Index name for the Keyword index.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SupabasePgvectorDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SupabasePgvectorDocumentStore</code> – Deserialized component.

## haystack_integrations.document_stores.supabase.groonga_document_store

### SupabaseGroongaDocumentStore

Bases: <code>DocumentStore</code>

A Document Store for Supabase using PGroonga for full-text search.

PGroonga is a PostgreSQL extension for fast, multilingual full-text search.
Unlike vector search, this store works with plain text queries — no embeddings needed.

Prerequisites:

- A Supabase project with PGroonga extension enabled.
- Enable PGroonga in your Supabase project by running:
  `CREATE EXTENSION IF NOT EXISTS pgroonga;`

Example usage:

```python
from haystack_integrations.document_stores.supabase import SupabaseGroongaDocumentStore
from haystack.utils import Secret

document_store = SupabaseGroongaDocumentStore(
    supabase_url="https://<project>.supabase.co",
    supabase_key=Secret.from_env_var("SUPABASE_SERVICE_KEY"),
    table_name="haystack_fts_documents",
)
document_store.warm_up()
```

#### __init__

```python
__init__(
    *,
    supabase_url: str,
    supabase_key: Secret = Secret.from_env_var(
        "SUPABASE_SERVICE_KEY", strict=False
    ),
    table_name: str = "haystack_groonga_documents",
    recreate_table: bool = False
) -> None
```

Creates a new SupabaseGroongaDocumentStore instance.

Note: Call warm_up() before using the store to initialize the client and table.

**Parameters:**

- **supabase_url** (<code>str</code>) – The URL of your Supabase project.
  Format: `https://<project-ref>.supabase.co`
- **supabase_key** (<code>Secret</code>) – The service role key for your Supabase project.
  Defaults to reading from the `SUPABASE_SERVICE_KEY` environment variable.
- **table_name** (<code>str</code>) – The name of the table to store documents in.
  Defaults to `haystack_groonga_documents`.
- **recreate_table** (<code>bool</code>) – Whether to drop and recreate the table on startup.
  Defaults to `False`.

#### warm_up

```python
warm_up() -> None
```

Initializes the Supabase client and sets up the table.

Must be called before using the document store.

#### count_documents

```python
count_documents() -> int
```

Returns the number of documents in the store.

**Returns:**

- <code>int</code> – Number of documents.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns documents matching the given filters.

Supports the standard Haystack filter syntax with the following operators:

- Comparison: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`
- Logical: `AND`, `OR`, `NOT` (`OR` and `NOT` support simple conditions
  only — no nested logical operators inside them)

**Known limitation:** For `!=` and `not in` on `meta.*` fields, documents
where the field is absent are included in the result (matching Python `None != value`
semantics). For `>` / `>=` / `<` / `<=`, documents where the field is absent
are excluded (SQL `NULL` comparison semantics).

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Optional Haystack filter dict.
  Simple comparison: `{"field": "meta.language", "operator": "==", "value": "en"}`
  Logical: `{"operator": "AND", "conditions": [...]}`

**Returns:**

- <code>list\[Document\]</code> – List of matching Document objects.

**Raises:**

- <code>FilterError</code> – If the filter structure is malformed or uses an unsupported operator.

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL
) -> int
```

Writes documents to the store.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Haystack Document objects to write.
- **policy** (<code>DuplicatePolicy</code>) – How to handle duplicate documents. Defaults to DuplicatePolicy.FAIL.

**Returns:**

- <code>int</code> – Number of documents written.

#### delete_by_filter

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes documents matching the given filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Filters to select documents for deletion.

**Returns:**

- <code>int</code> – Number of documents deleted.

#### update_by_filter

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Updates the metadata of documents matching the given filters.

Provided meta fields are merged into the existing document metadata.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Filters to select documents to update.
- **meta** (<code>dict\[str, Any\]</code>) – Metadata fields to set on matching documents.

**Returns:**

- <code>int</code> – Number of documents updated.

#### delete_all_documents

```python
delete_all_documents() -> None
```

Deletes all documents from the store.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes documents with the given IDs.

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
from_dict(data: dict[str, Any]) -> SupabaseGroongaDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SupabaseGroongaDocumentStore</code> – Deserialized component.
