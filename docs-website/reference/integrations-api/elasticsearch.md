---
title: "Elasticsearch"
id: integrations-elasticsearch
description: "Elasticsearch integration for Haystack"
slug: "/integrations-elasticsearch"
---


## `haystack_integrations.components.retrievers.elasticsearch.bm25_retriever`

### `ElasticsearchBM25Retriever`

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

#### `__init__`

```python
__init__(
    *,
    document_store: ElasticsearchDocumentStore,
    filters: dict[str, Any] | None = None,
    fuzziness: str = "AUTO",
    top_k: int = 10,
    scale_score: bool = False,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
)
```

Initialize ElasticsearchBM25Retriever with an instance ElasticsearchDocumentStore.

**Parameters:**

- **document_store** (<code>ElasticsearchDocumentStore</code>) – An instance of ElasticsearchDocumentStore.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents, for more info
  see `ElasticsearchDocumentStore.filter_documents`.
- **fuzziness** (<code>str</code>) – Fuzziness parameter passed to Elasticsearch. See the official
  [documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/common-options.html#fuzziness)
  for more details.
- **top_k** (<code>int</code>) – Maximum number of Documents to return.
- **scale_score** (<code>bool</code>) – If `True` scales the Document\`s scores between 0 and 1.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of `ElasticsearchDocumentStore`.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ElasticsearchBM25Retriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ElasticsearchBM25Retriever</code> – Deserialized component.

#### `run`

```python
run(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, list[Document]]
```

Retrieve documents using the BM25 keyword-based algorithm.

**Parameters:**

- **query** (<code>str</code>) – String to search in the `Document`s text.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – Maximum number of `Document` to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of `Document`s that match the query.

#### `run_async`

```python
run_async(
    query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents using the BM25 keyword-based algorithm.

**Parameters:**

- **query** (<code>str</code>) – String to search in the `Document` text.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – Maximum number of `Document` to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of `Document`s that match the query.

## `haystack_integrations.components.retrievers.elasticsearch.embedding_retriever`

### `ElasticsearchEmbeddingRetriever`

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

#### `__init__`

```python
__init__(
    *,
    document_store: ElasticsearchDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    num_candidates: int | None = None,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
)
```

Create the ElasticsearchEmbeddingRetriever component.

**Parameters:**

- **document_store** (<code>ElasticsearchDocumentStore</code>) – An instance of ElasticsearchDocumentStore.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents.
  Filters are applied during the approximate KNN search to ensure that top_k matching documents are returned.
- **top_k** (<code>int</code>) – Maximum number of Documents to return.
- **num_candidates** (<code>int | None</code>) – Number of approximate nearest neighbor candidates on each shard. Defaults to top_k * 10.
  Increasing this value will improve search accuracy at the cost of slower search speeds.
  You can read more about it in the Elasticsearch
  [documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#tune-approximate-knn-for-speed-accuracy)
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of ElasticsearchDocumentStore.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ElasticsearchEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ElasticsearchEmbeddingRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents using a vector similarity metric.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied when fetching documents from the Document Store.
  Filters are applied during the approximate kNN search to ensure the Retriever returns
  `top_k` matching documents.
  The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
- **top_k** (<code>int | None</code>) – Maximum number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of `Document`s most similar to the given `query_embedding`

#### `run_async`

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents using a vector similarity metric.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied when fetching documents from the Document Store.
  Filters are applied during the approximate kNN search to ensure the Retriever returns
  `top_k` matching documents.
  The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
- **top_k** (<code>int | None</code>) – Maximum number of documents to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: List of `Document`s that match the query.

## `haystack_integrations.components.retrievers.elasticsearch.sql_retriever`

### `ElasticsearchSQLRetriever`

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

#### `__init__`

```python
__init__(
    *,
    document_store: ElasticsearchDocumentStore,
    raise_on_failure: bool = True,
    fetch_size: int | None = None
)
```

Creates the ElasticsearchSQLRetriever component.

**Parameters:**

- **document_store** (<code>ElasticsearchDocumentStore</code>) – An instance of ElasticsearchDocumentStore to use with the Retriever.
- **raise_on_failure** (<code>bool</code>) – Whether to raise an exception if the API call fails. Otherwise, log a warning and return an empty dict.
- **fetch_size** (<code>int | None</code>) – Optional number of results to fetch per page. If not provided, the default
  fetch size set in Elasticsearch is used.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of ElasticsearchDocumentStore.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ElasticsearchSQLRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ElasticsearchSQLRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query: str,
    document_store: ElasticsearchDocumentStore | None = None,
    fetch_size: int | None = None,
) -> dict[str, dict[str, Any]]
```

Execute a raw Elasticsearch SQL query against the index.

**Parameters:**

- **query** (<code>str</code>) – The Elasticsearch SQL query to execute.
- **document_store** (<code>ElasticsearchDocumentStore | None</code>) – Optionally, an instance of ElasticsearchDocumentStore to use with the Retriever.
- **fetch_size** (<code>int | None</code>) – Optional number of results to fetch per page. If not provided, uses the value
  specified during initialization, or the default fetch size set in Elasticsearch.

**Returns:**

- <code>dict\[str, dict\[str, Any\]\]</code> – A dictionary containing the raw JSON response from Elasticsearch SQL API:
  - result: The raw JSON response from Elasticsearch (dict) or empty dict on error.

Example:
`python     retriever = ElasticsearchSQLRetriever(document_store=document_store)     result = retriever.run(         query="SELECT content, category FROM \"my_index\" WHERE category = 'A'"     )     # result["result"] contains the raw Elasticsearch JSON response     # result["result"]["columns"] contains column metadata     # result["result"]["rows"] contains the data rows     `

#### `run_async`

```python
run_async(
    query: str,
    document_store: ElasticsearchDocumentStore | None = None,
    fetch_size: int | None = None,
) -> dict[str, dict[str, Any]]
```

Asynchronously execute a raw Elasticsearch SQL query against the index.

**Parameters:**

- **query** (<code>str</code>) – The Elasticsearch SQL query to execute.
- **document_store** (<code>ElasticsearchDocumentStore | None</code>) – Optionally, an instance of ElasticsearchDocumentStore to use with the Retriever.
- **fetch_size** (<code>int | None</code>) – Optional number of results to fetch per page. If not provided, uses the value
  specified during initialization, or the default fetch size set in Elasticsearch.

**Returns:**

- <code>dict\[str, dict\[str, Any\]\]</code> – A dictionary containing the raw JSON response from Elasticsearch SQL API:
  - result: The raw JSON response from Elasticsearch (dict) or empty dict on error.

Example:
`python     retriever = ElasticsearchSQLRetriever(document_store=document_store)     result = await retriever.run_async(         query="SELECT content, category FROM \"my_index\" WHERE category = 'A'"     )     # result["result"] contains the raw Elasticsearch JSON response     # result["result"]["columns"] contains column metadata     # result["result"]["rows"] contains the data rows     `

## `haystack_integrations.document_stores.elasticsearch.document_store`

### `ElasticsearchDocumentStore`

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

#### `__init__`

```python
__init__(
    *,
    hosts: Hosts | None = None,
    custom_mapping: dict[str, Any] | None = None,
    index: str = "default",
    api_key: Secret = Secret.from_env_var("ELASTIC_API_KEY", strict=False),
    api_key_id: Secret = Secret.from_env_var(
        "ELASTIC_API_KEY_ID", strict=False
    ),
    embedding_similarity_function: Literal[
        "cosine", "dot_product", "l2_norm", "max_inner_product"
    ] = "cosine",
    **kwargs: Any
)
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

**Parameters:**

- **hosts** (<code>Hosts | None</code>) – List of hosts running the Elasticsearch client.
- **custom_mapping** (<code>dict\[str, Any\] | None</code>) – Custom mapping for the index. If not provided, a default mapping will be used.
- **index** (<code>str</code>) – Name of index in Elasticsearch.
- **api_key** (<code>Secret</code>) – A Secret object containing the API key for authenticating or base64-encoded with the
  concatenated secret and id for authenticating with Elasticsearch (separated by “:”).
- **api_key_id** (<code>Secret</code>) – A Secret object containing the API key ID for authenticating with Elasticsearch.
- **embedding_similarity_function** (<code>Literal['cosine', 'dot_product', 'l2_norm', 'max_inner_product']</code>) – The similarity function used to compare Documents embeddings.
  This parameter only takes effect if the index does not yet exist and is created.
  To choose the most appropriate function, look for information about your embedding model.
  To understand how document scores are computed, see the Elasticsearch
  [documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html#dense-vector-params)
- \*\***kwargs** (<code>Any</code>) – Optional arguments that `Elasticsearch` takes.

#### `client`

```python
client: Elasticsearch
```

Returns the synchronous Elasticsearch client, initializing it if necessary.

#### `async_client`

```python
async_client: AsyncElasticsearch
```

Returns the asynchronous Elasticsearch client, initializing it if necessary.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ElasticsearchDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ElasticsearchDocumentStore</code> – Deserialized component.

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

Asynchronously returns how many documents are present in the document store.

**Returns:**

- <code>int</code> – Number of documents in the document store.

#### `filter_documents`

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

The main query method for the document store. It retrieves all documents that match the filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary of filters to apply. For more information on the structure of the filters,
  see the official Elasticsearch
  [documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html)

**Returns:**

- <code>list\[Document\]</code> – List of `Document`s that match the filters.

#### `filter_documents_async`

```python
filter_documents_async(filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously retrieves all documents that match the filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary of filters to apply. For more information on the structure of the filters,
  see the official Elasticsearch
  [documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html)

**Returns:**

- <code>list\[Document\]</code> – List of `Document`s that match the filters.

#### `write_documents`

```python
write_documents(
    documents: list[Document],
    policy: DuplicatePolicy = DuplicatePolicy.NONE,
    refresh: Literal["wait_for", True, False] = "wait_for",
) -> int
```

Writes `Document`s to Elasticsearch.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Documents to write to the document store.
- **policy** (<code>DuplicatePolicy</code>) – DuplicatePolicy to apply when a document with the same ID already exists in the document store.
- **refresh** (<code>Literal['wait_for', True, False]</code>) – Controls when changes are made visible to search operations.
- `True`: Force refresh immediately after the operation.
- `False`: Do not refresh (better performance for bulk operations).
- `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
  For more details, see the [Elasticsearch refresh documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter).

**Returns:**

- <code>int</code> – Number of documents written to the document store.

**Raises:**

- <code>ValueError</code> – If `documents` is not a list of `Document`s.
- <code>DuplicateDocumentError</code> – If a document with the same ID already exists in the document store and
  `policy` is set to `DuplicatePolicy.FAIL` or `DuplicatePolicy.NONE`.
- <code>DocumentStoreError</code> – If an error occurs while writing the documents to the document store.

#### `write_documents_async`

```python
write_documents_async(
    documents: list[Document],
    policy: DuplicatePolicy = DuplicatePolicy.NONE,
    refresh: Literal["wait_for", True, False] = "wait_for",
) -> int
```

Asynchronously writes `Document`s to Elasticsearch.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Documents to write to the document store.
- **policy** (<code>DuplicatePolicy</code>) – DuplicatePolicy to apply when a document with the same ID already exists in the document store.
- **refresh** (<code>Literal['wait_for', True, False]</code>) – Controls when changes are made visible to search operations.
- `True`: Force refresh immediately after the operation.
- `False`: Do not refresh (better performance for bulk operations).
- `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
  For more details, see the [Elasticsearch refresh documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter).

**Returns:**

- <code>int</code> – Number of documents written to the document store.

**Raises:**

- <code>ValueError</code> – If `documents` is not a list of `Document`s.
- <code>DuplicateDocumentError</code> – If a document with the same ID already exists in the document store and
  `policy` is set to `DuplicatePolicy.FAIL` or `DuplicatePolicy.NONE`.
- <code>DocumentStoreError</code> – If an error occurs while writing the documents to the document store.

#### `delete_documents`

```python
delete_documents(
    document_ids: list[str],
    refresh: Literal["wait_for", True, False] = "wait_for",
) -> None
```

Deletes all documents with a matching document_ids from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete
- **refresh** (<code>Literal['wait_for', True, False]</code>) – Controls when changes are made visible to search operations.
- `True`: Force refresh immediately after the operation.
- `False`: Do not refresh (better performance for bulk operations).
- `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
  For more details, see the [Elasticsearch refresh documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter).

#### `delete_documents_async`

```python
delete_documents_async(
    document_ids: list[str],
    refresh: Literal["wait_for", True, False] = "wait_for",
) -> None
```

Asynchronously deletes all documents with a matching document_ids from the document store.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – the document ids to delete
- **refresh** (<code>Literal['wait_for', True, False]</code>) – Controls when changes are made visible to search operations.
- `True`: Force refresh immediately after the operation.
- `False`: Do not refresh (better performance for bulk operations).
- `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
  For more details, see the [Elasticsearch refresh documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter).

#### `delete_all_documents`

```python
delete_all_documents(
    recreate_index: bool = False, refresh: bool = True
) -> None
```

Deletes all documents in the document store.

A fast way to clear all documents from the document store while preserving any index settings and mappings.

**Parameters:**

- **recreate_index** (<code>bool</code>) – If True, the index will be deleted and recreated with the original mappings and
  settings. If False, all documents will be deleted using the `delete_by_query` API.
- **refresh** (<code>bool</code>) – If True, Elasticsearch refreshes all shards involved in the delete by query after the request
  completes. If False, no refresh is performed. For more details, see the
  [Elasticsearch delete_by_query refresh documentation](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-delete-by-query#operation-delete-by-query-refresh).

#### `delete_all_documents_async`

```python
delete_all_documents_async(
    recreate_index: bool = False, refresh: bool = True
) -> None
```

Asynchronously deletes all documents in the document store.

A fast way to clear all documents from the document store while preserving any index settings and mappings.

**Parameters:**

- **recreate_index** (<code>bool</code>) – If True, the index will be deleted and recreated with the original mappings and
  settings. If False, all documents will be deleted using the `delete_by_query` API.
- **refresh** (<code>bool</code>) – If True, Elasticsearch refreshes all shards involved in the delete by query after the request
  completes. If False, no refresh is performed. For more details, see the
  [Elasticsearch delete_by_query refresh documentation](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-delete-by-query#operation-delete-by-query-refresh).

#### `delete_by_filter`

```python
delete_by_filter(filters: dict[str, Any], refresh: bool = False) -> int
```

Deletes all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for deletion.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **refresh** (<code>bool</code>) – If True, Elasticsearch refreshes all shards involved in the delete by query after the request
  completes. If False, no refresh is performed. For more details, see the
  [Elasticsearch delete_by_query refresh documentation](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-delete-by-query#operation-delete-by-query-refresh).

**Returns:**

- <code>int</code> – The number of documents deleted.

#### `delete_by_filter_async`

```python
delete_by_filter_async(filters: dict[str, Any], refresh: bool = False) -> int
```

Asynchronously deletes all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for deletion.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **refresh** (<code>bool</code>) – If True, Elasticsearch refreshes all shards involved in the delete by query after the request
  completes. If False, no refresh is performed. For more details, see the
  [Elasticsearch refresh documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter).

**Returns:**

- <code>int</code> – The number of documents deleted.

#### `update_by_filter`

```python
update_by_filter(
    filters: dict[str, Any], meta: dict[str, Any], refresh: bool = False
) -> int
```

Updates the metadata of all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update.
- **refresh** (<code>bool</code>) – If True, Elasticsearch refreshes all shards involved in the update by query after the request
  completes. If False, no refresh is performed. For more details, see the
  [Elasticsearch update_by_query refresh documentation](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-update-by-query#operation-update-by-query-refresh).

**Returns:**

- <code>int</code> – The number of documents updated.

#### `update_by_filter_async`

```python
update_by_filter_async(
    filters: dict[str, Any], meta: dict[str, Any], refresh: bool = False
) -> int
```

Asynchronously updates the metadata of all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update.
- **refresh** (<code>bool</code>) – If True, Elasticsearch refreshes all shards involved in the update by query after the request
  completes. If False, no refresh is performed. For more details, see the
  [Elasticsearch update_by_query refresh documentation](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-update-by-query#operation-update-by-query-refresh).

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

Returns the number of unique values for each specified metadata field of the documents
that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to count documents.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **metadata_fields** (<code>list\[str\]</code>) – List of field names to calculate unique values for.
  Field names can include or omit the "meta." prefix.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary mapping each metadata field name to the count of its unique values among the filtered
  documents.

**Raises:**

- <code>ValueError</code> – If any of the requested fields don't exist in the index mapping.

#### `count_unique_metadata_by_filter_async`

```python
count_unique_metadata_by_filter_async(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Asynchronously returns the number of unique values for each specified metadata field of the documents
that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to count documents.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **metadata_fields** (<code>list\[str\]</code>) – List of field names to calculate unique values for.
  Field names can include or omit the "meta." prefix.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary mapping each metadata field name to the count of its unique values among the filtered
  documents.

**Raises:**

- <code>ValueError</code> – If any of the requested fields don't exist in the index mapping.

#### `get_metadata_fields_info`

```python
get_metadata_fields_info() -> dict[str, dict[str, str]]
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

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – The information about the fields in the index.

#### `get_metadata_fields_info_async`

```python
get_metadata_fields_info_async() -> dict[str, dict[str, str]]
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

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – The information about the fields in the index.

#### `get_metadata_field_min_max`

```python
get_metadata_field_min_max(metadata_field: str) -> dict[str, int | None]
```

Returns the minimum and maximum values for the given metadata field.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to get the minimum and maximum values for.

**Returns:**

- <code>dict\[str, int | None\]</code> – A dictionary with the keys "min" and "max", where each value is the minimum or maximum value of the
  metadata field across all documents.

#### `get_metadata_field_min_max_async`

```python
get_metadata_field_min_max_async(metadata_field: str) -> dict[str, int | None]
```

Asynchronously returns the minimum and maximum values for the given metadata field.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to get the minimum and maximum values for.

**Returns:**

- <code>dict\[str, int | None\]</code> – A dictionary with the keys "min" and "max", where each value is the minimum or maximum value of the
  metadata field across all documents.

#### `get_metadata_field_unique_values`

```python
get_metadata_field_unique_values(
    metadata_field: str,
    search_term: str | None = None,
    size: int | None = 10000,
    after: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any] | None]
```

Returns unique values for a metadata field, optionally filtered by a search term in the content.
Uses composite aggregations for proper pagination beyond 10k results.

See: https://www.elastic.co/docs/reference/aggregations/search-aggregations-bucket-composite-aggregation

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to get unique values for.
- **search_term** (<code>str | None</code>) – Optional search term to filter documents by matching in the content field.
- **size** (<code>int | None</code>) – The number of unique values to return per page. Defaults to 10000.
- **after** (<code>dict\[str, Any\] | None</code>) – Optional pagination key from the previous response. Use None for the first page.
  For subsequent pages, pass the `after_key` from the previous response.

**Returns:**

- <code>tuple\[list\[str\], dict\[str, Any\] | None\]</code> – A tuple containing (list of unique values, after_key for pagination).
  The after_key is None when there are no more results. Use it in the `after` parameter
  for the next page.

#### `get_metadata_field_unique_values_async`

```python
get_metadata_field_unique_values_async(
    metadata_field: str,
    search_term: str | None = None,
    size: int | None = 10000,
    after: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any] | None]
```

Asynchronously returns unique values for a metadata field, optionally filtered by a search term in the content.
Uses composite aggregations for proper pagination beyond 10k results.

See: https://www.elastic.co/docs/reference/aggregations/search-aggregations-bucket-composite-aggregation

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field to get unique values for.
- **search_term** (<code>str | None</code>) – Optional search term to filter documents by matching in the content field.
- **size** (<code>int | None</code>) – The number of unique values to return per page. Defaults to 10000.
- **after** (<code>dict\[str, Any\] | None</code>) – Optional pagination key from the previous response. Use None for the first page.
  For subsequent pages, pass the `after_key` from the previous response.

**Returns:**

- <code>tuple\[list\[str\], dict\[str, Any\] | None\]</code> – A tuple containing (list of unique values, after_key for pagination).
  The after_key is None when there are no more results. Use it in the `after` parameter
  for the next page.

## `haystack_integrations.document_stores.elasticsearch.filters`
