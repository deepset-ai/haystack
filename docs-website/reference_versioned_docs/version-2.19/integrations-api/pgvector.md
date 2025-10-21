---
title: "Pgvector"
id: integrations-pgvector
description: "Pgvector integration for Haystack"
slug: "/integrations-pgvector"
---

<a id="haystack_integrations.components.retrievers.pgvector.embedding_retriever"></a>

# Module haystack\_integrations.components.retrievers.pgvector.embedding\_retriever

<a id="haystack_integrations.components.retrievers.pgvector.embedding_retriever.PgvectorEmbeddingRetriever"></a>

## PgvectorEmbeddingRetriever

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

<a id="haystack_integrations.components.retrievers.pgvector.embedding_retriever.PgvectorEmbeddingRetriever.__init__"></a>

#### PgvectorEmbeddingRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: PgvectorDocumentStore,
             filters: Optional[Dict[str, Any]] = None,
             top_k: int = 10,
             vector_function: Optional[Literal["cosine_similarity",
                                               "inner_product",
                                               "l2_distance"]] = None,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE)
```

**Arguments**:

- `document_store`: An instance of `PgvectorDocumentStore`.
- `filters`: Filters applied to the retrieved Documents.
- `top_k`: Maximum number of Documents to return.
- `vector_function`: The similarity function to use when searching for similar embeddings.
Defaults to the one set in the `document_store` instance.
`"cosine_similarity"` and `"inner_product"` are similarity functions and
higher scores indicate greater similarity between the documents.
`"l2_distance"` returns the straight-line distance between vectors,
and the most similar documents are the ones with the smallest score.
**Important**: if the document store is using the `"hnsw"` search strategy, the vector function
should match the one utilized during index creation to take advantage of the index.
- `filter_policy`: Policy to determine how filters are applied.

**Raises**:

- `ValueError`: If `document_store` is not an instance of `PgvectorDocumentStore` or if `vector_function`
is not one of the valid options.

<a id="haystack_integrations.components.retrievers.pgvector.embedding_retriever.PgvectorEmbeddingRetriever.to_dict"></a>

#### PgvectorEmbeddingRetriever.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.pgvector.embedding_retriever.PgvectorEmbeddingRetriever.from_dict"></a>

#### PgvectorEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "PgvectorEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.pgvector.embedding_retriever.PgvectorEmbeddingRetriever.run"></a>

#### PgvectorEmbeddingRetriever.run

```python
@component.output_types(documents=List[Document])
def run(
    query_embedding: List[float],
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    vector_function: Optional[Literal["cosine_similarity", "inner_product",
                                      "l2_distance"]] = None
) -> Dict[str, List[Document]]
```

Retrieve documents from the `PgvectorDocumentStore`, based on their embeddings.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of Documents to return.
- `vector_function`: The similarity function to use when searching for similar embeddings.

**Returns**:

A dictionary with the following keys:
- `documents`: List of `Document`s that are similar to `query_embedding`.

<a id="haystack_integrations.components.retrievers.pgvector.embedding_retriever.PgvectorEmbeddingRetriever.run_async"></a>

#### PgvectorEmbeddingRetriever.run\_async

```python
@component.output_types(documents=List[Document])
async def run_async(
    query_embedding: List[float],
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    vector_function: Optional[Literal["cosine_similarity", "inner_product",
                                      "l2_distance"]] = None
) -> Dict[str, List[Document]]
```

Asynchronously retrieve documents from the `PgvectorDocumentStore`, based on their embeddings.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of Documents to return.
- `vector_function`: The similarity function to use when searching for similar embeddings.

**Returns**:

A dictionary with the following keys:
- `documents`: List of `Document`s that are similar to `query_embedding`.

<a id="haystack_integrations.components.retrievers.pgvector.keyword_retriever"></a>

# Module haystack\_integrations.components.retrievers.pgvector.keyword\_retriever

<a id="haystack_integrations.components.retrievers.pgvector.keyword_retriever.PgvectorKeywordRetriever"></a>

## PgvectorKeywordRetriever

Retrieve documents from the `PgvectorDocumentStore`, based on keywords.

To rank the documents, the `ts_rank_cd` function of PostgreSQL is used.
It considers how often the query terms appear in the document, how close together the terms are in the document,
and how important is the part of the document where they occur.
For more details, see
[Postgres documentation](https://www.postgresql.org/docs/current/textsearch-controls.html#TEXTSEARCH-RANKING).

Usage example:
```python
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

<a id="haystack_integrations.components.retrievers.pgvector.keyword_retriever.PgvectorKeywordRetriever.__init__"></a>

#### PgvectorKeywordRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: PgvectorDocumentStore,
             filters: Optional[Dict[str, Any]] = None,
             top_k: int = 10,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE)
```

**Arguments**:

- `document_store`: An instance of `PgvectorDocumentStore`.
- `filters`: Filters applied to the retrieved Documents.
- `top_k`: Maximum number of Documents to return.
- `filter_policy`: Policy to determine how filters are applied.

**Raises**:

- `ValueError`: If `document_store` is not an instance of `PgvectorDocumentStore`.

<a id="haystack_integrations.components.retrievers.pgvector.keyword_retriever.PgvectorKeywordRetriever.to_dict"></a>

#### PgvectorKeywordRetriever.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.pgvector.keyword_retriever.PgvectorKeywordRetriever.from_dict"></a>

#### PgvectorKeywordRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "PgvectorKeywordRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.pgvector.keyword_retriever.PgvectorKeywordRetriever.run"></a>

#### PgvectorKeywordRetriever.run

```python
@component.output_types(documents=List[Document])
def run(query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None) -> Dict[str, List[Document]]
```

Retrieve documents from the `PgvectorDocumentStore`, based on keywords.

**Arguments**:

- `query`: String to search in `Document`s' content.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of Documents to return.

**Returns**:

A dictionary with the following keys:
- `documents`: List of `Document`s that match the query.

<a id="haystack_integrations.components.retrievers.pgvector.keyword_retriever.PgvectorKeywordRetriever.run_async"></a>

#### PgvectorKeywordRetriever.run\_async

```python
@component.output_types(documents=List[Document])
async def run_async(query: str,
                    filters: Optional[Dict[str, Any]] = None,
                    top_k: Optional[int] = None) -> Dict[str, List[Document]]
```

Asynchronously retrieve documents from the `PgvectorDocumentStore`, based on keywords.

**Arguments**:

- `query`: String to search in `Document`s' content.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of Documents to return.

**Returns**:

A dictionary with the following keys:
- `documents`: List of `Document`s that match the query.

<a id="haystack_integrations.document_stores.pgvector.document_store"></a>

# Module haystack\_integrations.document\_stores.pgvector.document\_store

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore"></a>

## PgvectorDocumentStore

A Document Store using PostgreSQL with the [pgvector extension](https://github.com/pgvector/pgvector) installed.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.__init__"></a>

#### PgvectorDocumentStore.\_\_init\_\_

```python
def __init__(*,
             connection_string: Secret = Secret.from_env_var("PG_CONN_STR"),
             create_extension: bool = True,
             schema_name: str = "public",
             table_name: str = "haystack_documents",
             language: str = "english",
             embedding_dimension: int = 768,
             vector_type: Literal["vector", "halfvec"] = "vector",
             vector_function: Literal["cosine_similarity", "inner_product",
                                      "l2_distance"] = "cosine_similarity",
             recreate_table: bool = False,
             search_strategy: Literal["exact_nearest_neighbor",
                                      "hnsw"] = "exact_nearest_neighbor",
             hnsw_recreate_index_if_exists: bool = False,
             hnsw_index_creation_kwargs: Optional[Dict[str, int]] = None,
             hnsw_index_name: str = "haystack_hnsw_index",
             hnsw_ef_search: Optional[int] = None,
             keyword_index_name: str = "haystack_keyword_index")
```

Creates a new PgvectorDocumentStore instance.

It is meant to be connected to a PostgreSQL database with the pgvector extension installed.
A specific table to store Haystack documents will be created if it doesn't exist yet.

**Arguments**:

- `connection_string`: The connection string to use to connect to the PostgreSQL database, defined as an
environment variable. It can be provided in either URI format
e.g.: `PG_CONN_STR="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"`, or keyword/value format
e.g.: `PG_CONN_STR="host=HOST port=PORT dbname=DBNAME user=USER password=PASSWORD"`
See [PostgreSQL Documentation](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING)
for more details.
- `create_extension`: Whether to create the pgvector extension if it doesn't exist.
Set this to `True` (default) to automatically create the extension if it is missing.
Creating the extension may require superuser privileges.
If set to `False`, ensure the extension is already installed; otherwise, an error will be raised.
- `schema_name`: The name of the schema the table is created in. The schema must already exist.
- `table_name`: The name of the table to use to store Haystack documents.
- `language`: The language to be used to parse query and document content in keyword retrieval.
To see the list of available languages, you can run the following SQL query in your PostgreSQL database:
`SELECT cfgname FROM pg_ts_config;`.
More information can be found in this [StackOverflow answer](https://stackoverflow.com/a/39752553).
- `embedding_dimension`: The dimension of the embedding.
- `vector_type`: The type of vector used for embedding storage.
"vector" is the default.
"halfvec" stores embeddings in half-precision, which is particularly useful for high-dimensional embeddings
(dimension greater than 2,000 and up to 4,000). Requires pgvector versions 0.7.0 or later. For more
information, see the [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file).
- `vector_function`: The similarity function to use when searching for similar embeddings.
`"cosine_similarity"` and `"inner_product"` are similarity functions and
higher scores indicate greater similarity between the documents.
`"l2_distance"` returns the straight-line distance between vectors,
and the most similar documents are the ones with the smallest score.
**Important**: when using the `"hnsw"` search strategy, an index will be created that depends on the
`vector_function` passed here. Make sure subsequent queries will keep using the same
vector similarity function in order to take advantage of the index.
- `recreate_table`: Whether to recreate the table if it already exists.
- `search_strategy`: The search strategy to use when searching for similar embeddings.
`"exact_nearest_neighbor"` provides perfect recall but can be slow for large numbers of documents.
`"hnsw"` is an approximate nearest neighbor search strategy,
which trades off some accuracy for speed; it is recommended for large numbers of documents.
**Important**: when using the `"hnsw"` search strategy, an index will be created that depends on the
`vector_function` passed here. Make sure subsequent queries will keep using the same
vector similarity function in order to take advantage of the index.
- `hnsw_recreate_index_if_exists`: Whether to recreate the HNSW index if it already exists.
Only used if search_strategy is set to `"hnsw"`.
- `hnsw_index_creation_kwargs`: Additional keyword arguments to pass to the HNSW index creation.
Only used if search_strategy is set to `"hnsw"`. You can find the list of valid arguments in the
[pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw)
- `hnsw_index_name`: Index name for the HNSW index.
- `hnsw_ef_search`: The `ef_search` parameter to use at query time. Only used if search_strategy is set to
`"hnsw"`. You can find more information about this parameter in the
[pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw).
- `keyword_index_name`: Index name for the Keyword index.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.to_dict"></a>

#### PgvectorDocumentStore.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.from_dict"></a>

#### PgvectorDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "PgvectorDocumentStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.delete_table"></a>

#### PgvectorDocumentStore.delete\_table

```python
def delete_table()
```

Deletes the table used to store Haystack documents.
The name of the schema (`schema_name`) and the name of the table (`table_name`)
are defined when initializing the `PgvectorDocumentStore`.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.delete_table_async"></a>

#### PgvectorDocumentStore.delete\_table\_async

```python
async def delete_table_async()
```

Async method to delete the table used to store Haystack documents.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.count_documents"></a>

#### PgvectorDocumentStore.count\_documents

```python
def count_documents() -> int
```

Returns how many documents are present in the document store.

**Returns**:

Number of documents in the document store.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.count_documents_async"></a>

#### PgvectorDocumentStore.count\_documents\_async

```python
async def count_documents_async() -> int
```

Returns how many documents are present in the document store.

**Returns**:

Number of documents in the document store.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.filter_documents"></a>

#### PgvectorDocumentStore.filter\_documents

```python
def filter_documents(
        filters: Optional[Dict[str, Any]] = None) -> List[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

**Arguments**:

- `filters`: The filters to apply to the document list.

**Raises**:

- `TypeError`: If `filters` is not a dictionary.
- `ValueError`: If `filters` syntax is invalid.

**Returns**:

A list of Documents that match the given filters.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.filter_documents_async"></a>

#### PgvectorDocumentStore.filter\_documents\_async

```python
async def filter_documents_async(
        filters: Optional[Dict[str, Any]] = None) -> List[Document]
```

Asynchronously returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

**Arguments**:

- `filters`: The filters to apply to the document list.

**Raises**:

- `TypeError`: If `filters` is not a dictionary.
- `ValueError`: If `filters` syntax is invalid.

**Returns**:

A list of Documents that match the given filters.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.write_documents"></a>

#### PgvectorDocumentStore.write\_documents

```python
def write_documents(documents: List[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Writes documents to the document store.

**Arguments**:

- `documents`: A list of Documents to write to the document store.
- `policy`: The duplicate policy to use when writing documents.

**Raises**:

- `ValueError`: If `documents` contains objects that are not of type `Document`.
- `DuplicateDocumentError`: If a document with the same id already exists in the document store
and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
- `DocumentStoreError`: If the write operation fails for any other reason.

**Returns**:

The number of documents written to the document store.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.write_documents_async"></a>

#### PgvectorDocumentStore.write\_documents\_async

```python
async def write_documents_async(
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Asynchronously writes documents to the document store.

**Arguments**:

- `documents`: A list of Documents to write to the document store.
- `policy`: The duplicate policy to use when writing documents.

**Raises**:

- `ValueError`: If `documents` contains objects that are not of type `Document`.
- `DuplicateDocumentError`: If a document with the same id already exists in the document store
and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
- `DocumentStoreError`: If the write operation fails for any other reason.

**Returns**:

The number of documents written to the document store.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.delete_documents"></a>

#### PgvectorDocumentStore.delete\_documents

```python
def delete_documents(document_ids: List[str]) -> None
```

Deletes documents that match the provided `document_ids` from the document store.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.delete_documents_async"></a>

#### PgvectorDocumentStore.delete\_documents\_async

```python
async def delete_documents_async(document_ids: List[str]) -> None
```

Asynchronously deletes documents that match the provided `document_ids` from the document store.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.delete_all_documents"></a>

#### PgvectorDocumentStore.delete\_all\_documents

```python
def delete_all_documents() -> None
```

Deletes all documents in the document store.

<a id="haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.delete_all_documents_async"></a>

#### PgvectorDocumentStore.delete\_all\_documents\_async

```python
async def delete_all_documents_async() -> None
```

Asynchronously deletes all documents in the document store.
