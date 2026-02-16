---
title: "Elasticsearch"
id: integrations-elasticsearch
description: "Elasticsearch integration for Haystack"
slug: "/integrations-elasticsearch"
---

<a id="haystack_integrations.components.retrievers.elasticsearch.bm25_retriever"></a>

## Module haystack\_integrations.components.retrievers.elasticsearch.bm25\_retriever

<a id="haystack_integrations.components.retrievers.elasticsearch.bm25_retriever.ElasticsearchBM25Retriever"></a>

### ElasticsearchBM25Retriever

ElasticsearchBM25Retriever retrieves documents from the ElasticsearchDocumentStore using BM25 algorithm to find the
most similar documents to a user's query.

This retriever is only compatible with ElasticsearchDocumentStore.

Usage example:
```python
from haystack import Document
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever

document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")
retriever = ElasticsearchBM25Retriever(document_store=document_store)

# Add documents to DocumentStore
documents = [
    Document(text="My name is Carla and I live in Berlin"),
    Document(text="My name is Paul and I live in New York"),
    Document(text="My name is Silvano and I live in Matera"),
    Document(text="My name is Usagi Tsukino and I live in Tokyo"),
]
document_store.write_documents(documents)

result = retriever.run(query="Who lives in Berlin?")
for doc in result["documents"]:
    print(doc.content)
```

<a id="haystack_integrations.components.retrievers.elasticsearch.bm25_retriever.ElasticsearchBM25Retriever.__init__"></a>

#### ElasticsearchBM25Retriever.\_\_init\_\_

```python
def __init__(*,
             document_store: ElasticsearchDocumentStore,
             filters: dict[str, Any] | None = None,
             fuzziness: str = "AUTO",
             top_k: int = 10,
             scale_score: bool = False,
             filter_policy: str | FilterPolicy = FilterPolicy.REPLACE)
```

Initialize ElasticsearchBM25Retriever with an instance ElasticsearchDocumentStore.

**Arguments**:

- `document_store`: An instance of ElasticsearchDocumentStore.
- `filters`: Filters applied to the retrieved Documents, for more info
see `ElasticsearchDocumentStore.filter_documents`.
- `fuzziness`: Fuzziness parameter passed to Elasticsearch. See the official
[documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/common-options.html#fuzziness)
for more details.
- `top_k`: Maximum number of Documents to return.
- `scale_score`: If `True` scales the Document`s scores between 0 and 1.
- `filter_policy`: Policy to determine how filters are applied.

**Raises**:

- `ValueError`: If `document_store` is not an instance of `ElasticsearchDocumentStore`.

<a id="haystack_integrations.components.retrievers.elasticsearch.bm25_retriever.ElasticsearchBM25Retriever.to_dict"></a>

#### ElasticsearchBM25Retriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.elasticsearch.bm25_retriever.ElasticsearchBM25Retriever.from_dict"></a>

#### ElasticsearchBM25Retriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ElasticsearchBM25Retriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.elasticsearch.bm25_retriever.ElasticsearchBM25Retriever.run"></a>

#### ElasticsearchBM25Retriever.run

```python
@component.output_types(documents=list[Document])
def run(query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None) -> dict[str, list[Document]]
```

Retrieve documents using the BM25 keyword-based algorithm.

**Arguments**:

- `query`: String to search in the `Document`s text.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of `Document` to return.

**Returns**:

A dictionary with the following keys:
- `documents`: List of `Document`s that match the query.

<a id="haystack_integrations.components.retrievers.elasticsearch.bm25_retriever.ElasticsearchBM25Retriever.run_async"></a>

#### ElasticsearchBM25Retriever.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(query: str,
                    filters: dict[str, Any] | None = None,
                    top_k: int | None = None) -> dict[str, list[Document]]
```

Asynchronously retrieve documents using the BM25 keyword-based algorithm.

**Arguments**:

- `query`: String to search in the `Document` text.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of `Document` to return.

**Returns**:

A dictionary with the following keys:
- `documents`: List of `Document`s that match the query.

<a id="haystack_integrations.components.retrievers.elasticsearch.embedding_retriever"></a>

## Module haystack\_integrations.components.retrievers.elasticsearch.embedding\_retriever

<a id="haystack_integrations.components.retrievers.elasticsearch.embedding_retriever.ElasticsearchEmbeddingRetriever"></a>

### ElasticsearchEmbeddingRetriever

ElasticsearchEmbeddingRetriever retrieves documents from the ElasticsearchDocumentStore using vector similarity.

Usage example:
```python
from haystack import Document
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever

document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")
retriever = ElasticsearchEmbeddingRetriever(document_store=document_store)

# Add documents to DocumentStore
documents = [
    Document(text="My name is Carla and I live in Berlin"),
    Document(text="My name is Paul and I live in New York"),
    Document(text="My name is Silvano and I live in Matera"),
    Document(text="My name is Usagi Tsukino and I live in Tokyo"),
]
document_store.write_documents(documents)

te = SentenceTransformersTextEmbedder()
te.warm_up()
query_embeddings = te.run("Who lives in Berlin?")["embedding"]

result = retriever.run(query=query_embeddings)
for doc in result["documents"]:
    print(doc.content)
```

<a id="haystack_integrations.components.retrievers.elasticsearch.embedding_retriever.ElasticsearchEmbeddingRetriever.__init__"></a>

#### ElasticsearchEmbeddingRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: ElasticsearchDocumentStore,
             filters: dict[str, Any] | None = None,
             top_k: int = 10,
             num_candidates: int | None = None,
             filter_policy: str | FilterPolicy = FilterPolicy.REPLACE)
```

Create the ElasticsearchEmbeddingRetriever component.

**Arguments**:

- `document_store`: An instance of ElasticsearchDocumentStore.
- `filters`: Filters applied to the retrieved Documents.
Filters are applied during the approximate KNN search to ensure that top_k matching documents are returned.
- `top_k`: Maximum number of Documents to return.
- `num_candidates`: Number of approximate nearest neighbor candidates on each shard. Defaults to top_k * 10.
Increasing this value will improve search accuracy at the cost of slower search speeds.
You can read more about it in the Elasticsearch
[documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#tune-approximate-knn-for-speed-accuracy)
- `filter_policy`: Policy to determine how filters are applied.

**Raises**:

- `ValueError`: If `document_store` is not an instance of ElasticsearchDocumentStore.

<a id="haystack_integrations.components.retrievers.elasticsearch.embedding_retriever.ElasticsearchEmbeddingRetriever.to_dict"></a>

#### ElasticsearchEmbeddingRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.elasticsearch.embedding_retriever.ElasticsearchEmbeddingRetriever.from_dict"></a>

#### ElasticsearchEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ElasticsearchEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.elasticsearch.embedding_retriever.ElasticsearchEmbeddingRetriever.run"></a>

#### ElasticsearchEmbeddingRetriever.run

```python
@component.output_types(documents=list[Document])
def run(query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None) -> dict[str, list[Document]]
```

Retrieve documents using a vector similarity metric.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied when fetching documents from the Document Store.
Filters are applied during the approximate kNN search to ensure the Retriever returns
  `top_k` matching documents.
The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
- `top_k`: Maximum number of documents to return.

**Returns**:

A dictionary with the following keys:
- `documents`: List of `Document`s most similar to the given `query_embedding`

<a id="haystack_integrations.components.retrievers.elasticsearch.embedding_retriever.ElasticsearchEmbeddingRetriever.run_async"></a>

#### ElasticsearchEmbeddingRetriever.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(query_embedding: list[float],
                    filters: dict[str, Any] | None = None,
                    top_k: int | None = None) -> dict[str, list[Document]]
```

Asynchronously retrieve documents using a vector similarity metric.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied when fetching documents from the Document Store.
Filters are applied during the approximate kNN search to ensure the Retriever returns
  `top_k` matching documents.
The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
- `top_k`: Maximum number of documents to return.

**Returns**:

A dictionary with the following keys:
- `documents`: List of `Document`s that match the query.

<a id="haystack_integrations.components.retrievers.elasticsearch.sql_retriever"></a>

## Module haystack\_integrations.components.retrievers.elasticsearch.sql\_retriever

<a id="haystack_integrations.components.retrievers.elasticsearch.sql_retriever.ElasticsearchSQLRetriever"></a>

### ElasticsearchSQLRetriever

Executes raw Elasticsearch SQL queries against an ElasticsearchDocumentStore.

This component allows you to execute SQL queries directly against the Elasticsearch index,
which is useful for fetching metadata, aggregations, and other structured data at runtime.

Returns the raw JSON response from the Elasticsearch SQL API.

Usage example:
```python
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchSQLRetriever

document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")
retriever = ElasticsearchSQLRetriever(document_store=document_store)

result = retriever.run(
    query="SELECT content, category FROM \"my_index\" WHERE category = 'A'"
)
# result["result"] contains the raw Elasticsearch JSON response
```

<a id="haystack_integrations.components.retrievers.elasticsearch.sql_retriever.ElasticsearchSQLRetriever.__init__"></a>

#### ElasticsearchSQLRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: ElasticsearchDocumentStore,
             raise_on_failure: bool = True,
             fetch_size: int | None = None)
```

Creates the ElasticsearchSQLRetriever component.

**Arguments**:

- `document_store`: An instance of ElasticsearchDocumentStore to use with the Retriever.
- `raise_on_failure`: Whether to raise an exception if the API call fails. Otherwise, log a warning and return an empty dict.
- `fetch_size`: Optional number of results to fetch per page. If not provided, the default
fetch size set in Elasticsearch is used.

**Raises**:

- `ValueError`: If `document_store` is not an instance of ElasticsearchDocumentStore.

<a id="haystack_integrations.components.retrievers.elasticsearch.sql_retriever.ElasticsearchSQLRetriever.to_dict"></a>

#### ElasticsearchSQLRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.elasticsearch.sql_retriever.ElasticsearchSQLRetriever.from_dict"></a>

#### ElasticsearchSQLRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ElasticsearchSQLRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.elasticsearch.sql_retriever.ElasticsearchSQLRetriever.run"></a>

#### ElasticsearchSQLRetriever.run

```python
@component.output_types(result=dict[str, Any])
def run(query: str,
        document_store: ElasticsearchDocumentStore | None = None,
        fetch_size: int | None = None) -> dict[str, dict[str, Any]]
```

Execute a raw Elasticsearch SQL query against the index.

**Arguments**:

- `query`: The Elasticsearch SQL query to execute.
- `document_store`: Optionally, an instance of ElasticsearchDocumentStore to use with the Retriever.
- `fetch_size`: Optional number of results to fetch per page. If not provided, uses the value
specified during initialization, or the default fetch size set in Elasticsearch.

**Returns**:

A dictionary containing the raw JSON response from Elasticsearch SQL API:
- result: The raw JSON response from Elasticsearch (dict) or empty dict on error.

Example:
    ```python
    retriever = ElasticsearchSQLRetriever(document_store=document_store)
    result = retriever.run(
        query="SELECT content, category FROM \"my_index\" WHERE category = 'A'"
    )
    # result["result"] contains the raw Elasticsearch JSON response
    # result["result"]["columns"] contains column metadata
    # result["result"]["rows"] contains the data rows
    ```

<a id="haystack_integrations.components.retrievers.elasticsearch.sql_retriever.ElasticsearchSQLRetriever.run_async"></a>

#### ElasticsearchSQLRetriever.run\_async

```python
@component.output_types(result=dict[str, Any])
async def run_async(
        query: str,
        document_store: ElasticsearchDocumentStore | None = None,
        fetch_size: int | None = None) -> dict[str, dict[str, Any]]
```

Asynchronously execute a raw Elasticsearch SQL query against the index.

**Arguments**:

- `query`: The Elasticsearch SQL query to execute.
- `document_store`: Optionally, an instance of ElasticsearchDocumentStore to use with the Retriever.
- `fetch_size`: Optional number of results to fetch per page. If not provided, uses the value
specified during initialization, or the default fetch size set in Elasticsearch.

**Returns**:

A dictionary containing the raw JSON response from Elasticsearch SQL API:
- result: The raw JSON response from Elasticsearch (dict) or empty dict on error.

Example:
    ```python
    retriever = ElasticsearchSQLRetriever(document_store=document_store)
    result = await retriever.run_async(
        query="SELECT content, category FROM \"my_index\" WHERE category = 'A'"
    )
    # result["result"] contains the raw Elasticsearch JSON response
    # result["result"]["columns"] contains column metadata
    # result["result"]["rows"] contains the data rows
    ```

<a id="haystack_integrations.document_stores.elasticsearch.document_store"></a>

## Module haystack\_integrations.document\_stores.elasticsearch.document\_store

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore"></a>

### ElasticsearchDocumentStore

An ElasticsearchDocumentStore instance that works with Elastic Cloud or your own
Elasticsearch cluster.

Usage example (Elastic Cloud):
```python
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
document_store = ElasticsearchDocumentStore(
    api_key_id=Secret.from_env_var("ELASTIC_API_KEY_ID", strict=False),
    api_key=Secret.from_env_var("ELASTIC_API_KEY", strict=False),
)
```

Usage example (self-hosted Elasticsearch instance):
```python
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")
```
In the above example we connect with security disabled just to show the basic usage.
We strongly recommend to enable security so that only authorized users can access your data.

For more details on how to connect to Elasticsearch and configure security,
see the official Elasticsearch
[documentation](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/connecting.html)

All extra keyword arguments will be passed to the Elasticsearch client.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.__init__"></a>

#### ElasticsearchDocumentStore.\_\_init\_\_

```python
def __init__(
        *,
        hosts: Hosts | None = None,
        custom_mapping: dict[str, Any] | None = None,
        index: str = "default",
        api_key: Secret = Secret.from_env_var("ELASTIC_API_KEY", strict=False),
        api_key_id: Secret = Secret.from_env_var("ELASTIC_API_KEY_ID",
                                                 strict=False),
        embedding_similarity_function: Literal["cosine", "dot_product",
                                               "l2_norm",
                                               "max_inner_product"] = "cosine",
        **kwargs: Any)
```

Creates a new ElasticsearchDocumentStore instance.

It will also try to create that index if it doesn't exist yet. Otherwise, it will use the existing one.

One can also set the similarity function used to compare Documents embeddings. This is mostly useful
when using the `ElasticsearchDocumentStore` in a Pipeline with an `ElasticsearchEmbeddingRetriever`.

For more information on connection parameters, see the official Elasticsearch
[documentation](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/connecting.html)

For the full list of supported kwargs, see the official Elasticsearch
[reference](https://elasticsearch-py.readthedocs.io/en/stable/api.html#module-elasticsearch)

Authentication is provided via Secret objects, which by default are loaded from environment variables.
You can either provide both `api_key_id` and `api_key`, or just `api_key` containing a base64-encoded string
of `id:secret`. Secret instances can also be loaded from a token using the `Secret.from_token()` method.

**Arguments**:

- `hosts`: List of hosts running the Elasticsearch client.
- `custom_mapping`: Custom mapping for the index. If not provided, a default mapping will be used.
- `index`: Name of index in Elasticsearch.
- `api_key`: A Secret object containing the API key for authenticating or base64-encoded with the
concatenated secret and id for authenticating with Elasticsearch (separated by “:”).
- `api_key_id`: A Secret object containing the API key ID for authenticating with Elasticsearch.
- `embedding_similarity_function`: The similarity function used to compare Documents embeddings.
This parameter only takes effect if the index does not yet exist and is created.
To choose the most appropriate function, look for information about your embedding model.
To understand how document scores are computed, see the Elasticsearch
[documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html#dense-vector-params)
- `**kwargs`: Optional arguments that `Elasticsearch` takes.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.client"></a>

#### ElasticsearchDocumentStore.client

```python
@property
def client() -> Elasticsearch
```

Returns the synchronous Elasticsearch client, initializing it if necessary.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.async_client"></a>

#### ElasticsearchDocumentStore.async\_client

```python
@property
def async_client() -> AsyncElasticsearch
```

Returns the asynchronous Elasticsearch client, initializing it if necessary.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.to_dict"></a>

#### ElasticsearchDocumentStore.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.from_dict"></a>

#### ElasticsearchDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ElasticsearchDocumentStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.count_documents"></a>

#### ElasticsearchDocumentStore.count\_documents

```python
def count_documents() -> int
```

Returns how many documents are present in the document store.

**Returns**:

Number of documents in the document store.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.count_documents_async"></a>

#### ElasticsearchDocumentStore.count\_documents\_async

```python
async def count_documents_async() -> int
```

Asynchronously returns how many documents are present in the document store.

**Returns**:

Number of documents in the document store.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.filter_documents"></a>

#### ElasticsearchDocumentStore.filter\_documents

```python
def filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

The main query method for the document store. It retrieves all documents that match the filters.

**Arguments**:

- `filters`: A dictionary of filters to apply. For more information on the structure of the filters,
see the official Elasticsearch
[documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html)

**Returns**:

List of `Document`s that match the filters.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.filter_documents_async"></a>

#### ElasticsearchDocumentStore.filter\_documents\_async

```python
async def filter_documents_async(
        filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously retrieves all documents that match the filters.

**Arguments**:

- `filters`: A dictionary of filters to apply. For more information on the structure of the filters,
see the official Elasticsearch
[documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html)

**Returns**:

List of `Document`s that match the filters.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.write_documents"></a>

#### ElasticsearchDocumentStore.write\_documents

```python
def write_documents(
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
        refresh: Literal["wait_for", True, False] = "wait_for") -> int
```

Writes `Document`s to Elasticsearch.

**Arguments**:

- `documents`: List of Documents to write to the document store.
- `policy`: DuplicatePolicy to apply when a document with the same ID already exists in the document store.
- `refresh`: Controls when changes are made visible to search operations.
- `True`: Force refresh immediately after the operation.
- `False`: Do not refresh (better performance for bulk operations).
- `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
For more details, see the [Elasticsearch refresh documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter).

**Raises**:

- `ValueError`: If `documents` is not a list of `Document`s.
- `DuplicateDocumentError`: If a document with the same ID already exists in the document store and
`policy` is set to `DuplicatePolicy.FAIL` or `DuplicatePolicy.NONE`.
- `DocumentStoreError`: If an error occurs while writing the documents to the document store.

**Returns**:

Number of documents written to the document store.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.write_documents_async"></a>

#### ElasticsearchDocumentStore.write\_documents\_async

```python
async def write_documents_async(
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
        refresh: Literal["wait_for", True, False] = "wait_for") -> int
```

Asynchronously writes `Document`s to Elasticsearch.

**Arguments**:

- `documents`: List of Documents to write to the document store.
- `policy`: DuplicatePolicy to apply when a document with the same ID already exists in the document store.
- `refresh`: Controls when changes are made visible to search operations.
- `True`: Force refresh immediately after the operation.
- `False`: Do not refresh (better performance for bulk operations).
- `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
For more details, see the [Elasticsearch refresh documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter).

**Raises**:

- `ValueError`: If `documents` is not a list of `Document`s.
- `DuplicateDocumentError`: If a document with the same ID already exists in the document store and
`policy` is set to `DuplicatePolicy.FAIL` or `DuplicatePolicy.NONE`.
- `DocumentStoreError`: If an error occurs while writing the documents to the document store.

**Returns**:

Number of documents written to the document store.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.delete_documents"></a>

#### ElasticsearchDocumentStore.delete\_documents

```python
def delete_documents(
        document_ids: list[str],
        refresh: Literal["wait_for", True, False] = "wait_for") -> None
```

Deletes all documents with a matching document_ids from the document store.

**Arguments**:

- `document_ids`: the document ids to delete
- `refresh`: Controls when changes are made visible to search operations.
- `True`: Force refresh immediately after the operation.
- `False`: Do not refresh (better performance for bulk operations).
- `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
For more details, see the [Elasticsearch refresh documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter).

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.delete_documents_async"></a>

#### ElasticsearchDocumentStore.delete\_documents\_async

```python
async def delete_documents_async(
        document_ids: list[str],
        refresh: Literal["wait_for", True, False] = "wait_for") -> None
```

Asynchronously deletes all documents with a matching document_ids from the document store.

**Arguments**:

- `document_ids`: the document ids to delete
- `refresh`: Controls when changes are made visible to search operations.
- `True`: Force refresh immediately after the operation.
- `False`: Do not refresh (better performance for bulk operations).
- `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
For more details, see the [Elasticsearch refresh documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter).

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.delete_all_documents"></a>

#### ElasticsearchDocumentStore.delete\_all\_documents

```python
def delete_all_documents(recreate_index: bool = False,
                         refresh: bool = True) -> None
```

Deletes all documents in the document store.

A fast way to clear all documents from the document store while preserving any index settings and mappings.

**Arguments**:

- `recreate_index`: If True, the index will be deleted and recreated with the original mappings and
settings. If False, all documents will be deleted using the `delete_by_query` API.
- `refresh`: If True, Elasticsearch refreshes all shards involved in the delete by query after the request
completes. If False, no refresh is performed. For more details, see the
[Elasticsearch delete_by_query refresh documentation](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-delete-by-query#operation-delete-by-query-refresh).

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.delete_all_documents_async"></a>

#### ElasticsearchDocumentStore.delete\_all\_documents\_async

```python
async def delete_all_documents_async(recreate_index: bool = False,
                                     refresh: bool = True) -> None
```

Asynchronously deletes all documents in the document store.

A fast way to clear all documents from the document store while preserving any index settings and mappings.

**Arguments**:

- `recreate_index`: If True, the index will be deleted and recreated with the original mappings and
settings. If False, all documents will be deleted using the `delete_by_query` API.
- `refresh`: If True, Elasticsearch refreshes all shards involved in the delete by query after the request
completes. If False, no refresh is performed. For more details, see the
[Elasticsearch delete_by_query refresh documentation](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-delete-by-query#operation-delete-by-query-refresh).

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.delete_by_filter"></a>

#### ElasticsearchDocumentStore.delete\_by\_filter

```python
def delete_by_filter(filters: dict[str, Any], refresh: bool = False) -> int
```

Deletes all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for deletion.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `refresh`: If True, Elasticsearch refreshes all shards involved in the delete by query after the request
completes. If False, no refresh is performed. For more details, see the
[Elasticsearch delete_by_query refresh documentation](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-delete-by-query#operation-delete-by-query-refresh).

**Returns**:

The number of documents deleted.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.delete_by_filter_async"></a>

#### ElasticsearchDocumentStore.delete\_by\_filter\_async

```python
async def delete_by_filter_async(filters: dict[str, Any],
                                 refresh: bool = False) -> int
```

Asynchronously deletes all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for deletion.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `refresh`: If True, Elasticsearch refreshes all shards involved in the delete by query after the request
completes. If False, no refresh is performed. For more details, see the
[Elasticsearch refresh documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter).

**Returns**:

The number of documents deleted.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.update_by_filter"></a>

#### ElasticsearchDocumentStore.update\_by\_filter

```python
def update_by_filter(filters: dict[str, Any],
                     meta: dict[str, Any],
                     refresh: bool = False) -> int
```

Updates the metadata of all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for updating.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `meta`: The metadata fields to update.
- `refresh`: If True, Elasticsearch refreshes all shards involved in the update by query after the request
completes. If False, no refresh is performed. For more details, see the
[Elasticsearch update_by_query refresh documentation](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-update-by-query#operation-update-by-query-refresh).

**Returns**:

The number of documents updated.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.update_by_filter_async"></a>

#### ElasticsearchDocumentStore.update\_by\_filter\_async

```python
async def update_by_filter_async(filters: dict[str, Any],
                                 meta: dict[str, Any],
                                 refresh: bool = False) -> int
```

Asynchronously updates the metadata of all documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to select documents for updating.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `meta`: The metadata fields to update.
- `refresh`: If True, Elasticsearch refreshes all shards involved in the update by query after the request
completes. If False, no refresh is performed. For more details, see the
[Elasticsearch update_by_query refresh documentation](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-update-by-query#operation-update-by-query-refresh).

**Returns**:

The number of documents updated.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.count_documents_by_filter"></a>

#### ElasticsearchDocumentStore.count\_documents\_by\_filter

```python
def count_documents_by_filter(filters: dict[str, Any]) -> int
```

Returns the number of documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to count documents.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns**:

The number of documents that match the filters.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.count_documents_by_filter_async"></a>

#### ElasticsearchDocumentStore.count\_documents\_by\_filter\_async

```python
async def count_documents_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously returns the number of documents that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to count documents.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns**:

The number of documents that match the filters.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.count_unique_metadata_by_filter"></a>

#### ElasticsearchDocumentStore.count\_unique\_metadata\_by\_filter

```python
def count_unique_metadata_by_filter(
        filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]
```

Returns the number of unique values for each specified metadata field of the documents

that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to count documents.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `metadata_fields`: List of field names to calculate unique values for.
Field names can include or omit the "meta." prefix.

**Raises**:

- `ValueError`: If any of the requested fields don't exist in the index mapping.

**Returns**:

A dictionary mapping each metadata field name to the count of its unique values among the filtered
documents.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.count_unique_metadata_by_filter_async"></a>

#### ElasticsearchDocumentStore.count\_unique\_metadata\_by\_filter\_async

```python
async def count_unique_metadata_by_filter_async(
        filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]
```

Asynchronously returns the number of unique values for each specified metadata field of the documents

that match the provided filters.

**Arguments**:

- `filters`: The filters to apply to count documents.
For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- `metadata_fields`: List of field names to calculate unique values for.
Field names can include or omit the "meta." prefix.

**Raises**:

- `ValueError`: If any of the requested fields don't exist in the index mapping.

**Returns**:

A dictionary mapping each metadata field name to the count of its unique values among the filtered
documents.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.get_metadata_fields_info"></a>

#### ElasticsearchDocumentStore.get\_metadata\_fields\_info

```python
def get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Returns the information about the fields in the index.

If we populated the index with documents like:

```python
    Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1})
    Document(content="Doc 2", meta={"category": "B", "status": "inactive"})
```

This method would return:

```python
    {
        'content': {'type': 'text'},
        'category': {'type': 'keyword'},
        'status': {'type': 'keyword'},
        'priority': {'type': 'long'},
    }
```

**Returns**:

The information about the fields in the index.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.get_metadata_fields_info_async"></a>

#### ElasticsearchDocumentStore.get\_metadata\_fields\_info\_async

```python
async def get_metadata_fields_info_async() -> dict[str, dict[str, str]]
```

Asynchronously returns the information about the fields in the index.

If we populated the index with documents like:

```python
    Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1})
    Document(content="Doc 2", meta={"category": "B", "status": "inactive"})
```

This method would return:

```python
    {
        'content': {'type': 'text'},
        'category': {'type': 'keyword'},
        'status': {'type': 'keyword'},
        'priority': {'type': 'long'},
    }
```

**Returns**:

The information about the fields in the index.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.get_metadata_field_min_max"></a>

#### ElasticsearchDocumentStore.get\_metadata\_field\_min\_max

```python
def get_metadata_field_min_max(metadata_field: str) -> dict[str, int | None]
```

Returns the minimum and maximum values for the given metadata field.

**Arguments**:

- `metadata_field`: The metadata field to get the minimum and maximum values for.

**Returns**:

A dictionary with the keys "min" and "max", where each value is the minimum or maximum value of the
metadata field across all documents.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.get_metadata_field_min_max_async"></a>

#### ElasticsearchDocumentStore.get\_metadata\_field\_min\_max\_async

```python
async def get_metadata_field_min_max_async(
        metadata_field: str) -> dict[str, int | None]
```

Asynchronously returns the minimum and maximum values for the given metadata field.

**Arguments**:

- `metadata_field`: The metadata field to get the minimum and maximum values for.

**Returns**:

A dictionary with the keys "min" and "max", where each value is the minimum or maximum value of the
metadata field across all documents.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.get_metadata_field_unique_values"></a>

#### ElasticsearchDocumentStore.get\_metadata\_field\_unique\_values

```python
def get_metadata_field_unique_values(
    metadata_field: str,
    search_term: str | None = None,
    size: int | None = 10000,
    after: dict[str, Any] | None = None
) -> tuple[list[str], dict[str, Any] | None]
```

Returns unique values for a metadata field, optionally filtered by a search term in the content.

Uses composite aggregations for proper pagination beyond 10k results.

See: https://www.elastic.co/docs/reference/aggregations/search-aggregations-bucket-composite-aggregation

**Arguments**:

- `metadata_field`: The metadata field to get unique values for.
- `search_term`: Optional search term to filter documents by matching in the content field.
- `size`: The number of unique values to return per page. Defaults to 10000.
- `after`: Optional pagination key from the previous response. Use None for the first page.
For subsequent pages, pass the `after_key` from the previous response.

**Returns**:

A tuple containing (list of unique values, after_key for pagination).
The after_key is None when there are no more results. Use it in the `after` parameter
for the next page.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.get_metadata_field_unique_values_async"></a>

#### ElasticsearchDocumentStore.get\_metadata\_field\_unique\_values\_async

```python
async def get_metadata_field_unique_values_async(
    metadata_field: str,
    search_term: str | None = None,
    size: int | None = 10000,
    after: dict[str, Any] | None = None
) -> tuple[list[str], dict[str, Any] | None]
```

Asynchronously returns unique values for a metadata field, optionally filtered by a search term in the content.

Uses composite aggregations for proper pagination beyond 10k results.

See: https://www.elastic.co/docs/reference/aggregations/search-aggregations-bucket-composite-aggregation

**Arguments**:

- `metadata_field`: The metadata field to get unique values for.
- `search_term`: Optional search term to filter documents by matching in the content field.
- `size`: The number of unique values to return per page. Defaults to 10000.
- `after`: Optional pagination key from the previous response. Use None for the first page.
For subsequent pages, pass the `after_key` from the previous response.

**Returns**:

A tuple containing (list of unique values, after_key for pagination).
The after_key is None when there are no more results. Use it in the `after` parameter
for the next page.

