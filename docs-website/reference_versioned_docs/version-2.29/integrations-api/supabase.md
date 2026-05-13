---
title: "Supabase"
id: integrations-supabase
description: "Supabase integration for Haystack"
slug: "/integrations-supabase"
---


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
