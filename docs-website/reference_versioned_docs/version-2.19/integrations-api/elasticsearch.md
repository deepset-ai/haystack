---
title: "Elasticsearch"
id: integrations-elasticsearch
description: "Elasticsearch integration for Haystack"
slug: "/integrations-elasticsearch"
---

<a id="haystack_integrations.components.retrievers.elasticsearch.bm25_retriever"></a>

# Module haystack\_integrations.components.retrievers.elasticsearch.bm25\_retriever

<a id="haystack_integrations.components.retrievers.elasticsearch.bm25_retriever.ElasticsearchBM25Retriever"></a>

## ElasticsearchBM25Retriever

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
             filters: Optional[Dict[str, Any]] = None,
             fuzziness: str = "AUTO",
             top_k: int = 10,
             scale_score: bool = False,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE)
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
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.elasticsearch.bm25_retriever.ElasticsearchBM25Retriever.from_dict"></a>

#### ElasticsearchBM25Retriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "ElasticsearchBM25Retriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.elasticsearch.bm25_retriever.ElasticsearchBM25Retriever.run"></a>

#### ElasticsearchBM25Retriever.run

```python
@component.output_types(documents=List[Document])
def run(query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None) -> Dict[str, List[Document]]
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
@component.output_types(documents=List[Document])
async def run_async(query: str,
                    filters: Optional[Dict[str, Any]] = None,
                    top_k: Optional[int] = None) -> Dict[str, List[Document]]
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

# Module haystack\_integrations.components.retrievers.elasticsearch.embedding\_retriever

<a id="haystack_integrations.components.retrievers.elasticsearch.embedding_retriever.ElasticsearchEmbeddingRetriever"></a>

## ElasticsearchEmbeddingRetriever

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
             filters: Optional[Dict[str, Any]] = None,
             top_k: int = 10,
             num_candidates: Optional[int] = None,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE)
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
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.elasticsearch.embedding_retriever.ElasticsearchEmbeddingRetriever.from_dict"></a>

#### ElasticsearchEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "ElasticsearchEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.elasticsearch.embedding_retriever.ElasticsearchEmbeddingRetriever.run"></a>

#### ElasticsearchEmbeddingRetriever.run

```python
@component.output_types(documents=List[Document])
def run(query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None) -> Dict[str, List[Document]]
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
@component.output_types(documents=List[Document])
async def run_async(query_embedding: List[float],
                    filters: Optional[Dict[str, Any]] = None,
                    top_k: Optional[int] = None) -> Dict[str, List[Document]]
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

<a id="haystack_integrations.document_stores.elasticsearch.document_store"></a>

# Module haystack\_integrations.document\_stores.elasticsearch.document\_store

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore"></a>

## ElasticsearchDocumentStore

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
        hosts: Optional[Hosts] = None,
        custom_mapping: Optional[Dict[str, Any]] = None,
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
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.from_dict"></a>

#### ElasticsearchDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "ElasticsearchDocumentStore"
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
def filter_documents(
        filters: Optional[Dict[str, Any]] = None) -> List[Document]
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
        filters: Optional[Dict[str, Any]] = None) -> List[Document]
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
def write_documents(documents: List[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Writes `Document`s to Elasticsearch.

**Arguments**:

- `documents`: List of Documents to write to the document store.
- `policy`: DuplicatePolicy to apply when a document with the same ID already exists in the document store.

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
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Asynchronously writes `Document`s to Elasticsearch.

**Arguments**:

- `documents`: List of Documents to write to the document store.
- `policy`: DuplicatePolicy to apply when a document with the same ID already exists in the document store.

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
def delete_documents(document_ids: List[str]) -> None
```

Deletes all documents with a matching document_ids from the document store.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.delete_documents_async"></a>

#### ElasticsearchDocumentStore.delete\_documents\_async

```python
async def delete_documents_async(document_ids: List[str]) -> None
```

Asynchronously deletes all documents with a matching document_ids from the document store.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.delete_all_documents"></a>

#### ElasticsearchDocumentStore.delete\_all\_documents

```python
def delete_all_documents(recreate_index: bool = False) -> None
```

Deletes all documents in the document store.

A fast way to clear all documents from the document store while preserving any index settings and mappings.

**Arguments**:

- `recreate_index`: If True, the index will be deleted and recreated with the original mappings and
settings. If False, all documents will be deleted using the `delete_by_query` API.

<a id="haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore.delete_all_documents_async"></a>

#### ElasticsearchDocumentStore.delete\_all\_documents\_async

```python
async def delete_all_documents_async(recreate_index: bool = False) -> None
```

Asynchronously deletes all documents in the document store.

A fast way to clear all documents from the document store while preserving any index settings and mappings.

**Arguments**:

- `recreate_index`: If True, the index will be deleted and recreated with the original mappings and
settings. If False, all documents will be deleted using the `delete_by_query` API.
