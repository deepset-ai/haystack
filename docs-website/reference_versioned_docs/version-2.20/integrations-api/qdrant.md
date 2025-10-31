---
title: "Qdrant"
id: integrations-qdrant
description: "Qdrant integration for Haystack"
slug: "/integrations-qdrant"
---

<a id="haystack_integrations.components.retrievers.qdrant.retriever"></a>

## Module haystack\_integrations.components.retrievers.qdrant.retriever

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantEmbeddingRetriever"></a>

### QdrantEmbeddingRetriever

A component for retrieving documents from an QdrantDocumentStore using dense vectors.

Usage example:
```python
from haystack.dataclasses import Document
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

document_store = QdrantDocumentStore(
    ":memory:",
    recreate_index=True,
    return_embedding=True,
)

document_store.write_documents([Document(content="test", embedding=[0.5]*768)])

retriever = QdrantEmbeddingRetriever(document_store=document_store)

# using a fake vector to keep the example simple
retriever.run(query_embedding=[0.1]*768)
```

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantEmbeddingRetriever.__init__"></a>

#### QdrantEmbeddingRetriever.\_\_init\_\_

```python
def __init__(document_store: QdrantDocumentStore,
             filters: Optional[Union[Dict[str, Any], models.Filter]] = None,
             top_k: int = 10,
             scale_score: bool = False,
             return_embedding: bool = False,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
             score_threshold: Optional[float] = None,
             group_by: Optional[str] = None,
             group_size: Optional[int] = None) -> None
```

Create a QdrantEmbeddingRetriever component.

**Arguments**:

- `document_store`: An instance of QdrantDocumentStore.
- `filters`: A dictionary with filters to narrow down the search space.
- `top_k`: The maximum number of documents to retrieve. If using `group_by` parameters, maximum number of
groups to return.
- `scale_score`: Whether to scale the scores of the retrieved documents or not.
- `return_embedding`: Whether to return the embedding of the retrieved Documents.
- `filter_policy`: Policy to determine how filters are applied.
- `score_threshold`: A minimal score threshold for the result.
Score of the returned result might be higher or smaller than the threshold
 depending on the `similarity` function specified in the Document Store.
E.g. for cosine similarity only higher scores will be returned.
- `group_by`: Payload field to group by, must be a string or number field. If the field contains more than 1
value, all values will be used for grouping. One point can be in multiple groups.
- `group_size`: Maximum amount of points to return per group. Default is 3.

**Raises**:

- `ValueError`: If `document_store` is not an instance of `QdrantDocumentStore`.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantEmbeddingRetriever.to_dict"></a>

#### QdrantEmbeddingRetriever.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantEmbeddingRetriever.from_dict"></a>

#### QdrantEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "QdrantEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantEmbeddingRetriever.run"></a>

#### QdrantEmbeddingRetriever.run

```python
@component.output_types(documents=List[Document])
def run(query_embedding: List[float],
        filters: Optional[Union[Dict[str, Any], models.Filter]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        group_by: Optional[str] = None,
        group_size: Optional[int] = None) -> Dict[str, List[Document]]
```

Run the Embedding Retriever on the given input data.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: A dictionary with filters to narrow down the search space.
- `top_k`: The maximum number of documents to return. If using `group_by` parameters, maximum number of
groups to return.
- `scale_score`: Whether to scale the scores of the retrieved documents or not.
- `return_embedding`: Whether to return the embedding of the retrieved Documents.
- `score_threshold`: A minimal score threshold for the result.
- `group_by`: Payload field to group by, must be a string or number field. If the field contains more than 1
value, all values will be used for grouping. One point can be in multiple groups.
- `group_size`: Maximum amount of points to return per group. Default is 3.

**Raises**:

- `ValueError`: If 'filter_policy' is set to 'MERGE' and 'filters' is a native Qdrant filter.

**Returns**:

The retrieved documents.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantEmbeddingRetriever.run_async"></a>

#### QdrantEmbeddingRetriever.run\_async

```python
@component.output_types(documents=List[Document])
async def run_async(
        query_embedding: List[float],
        filters: Optional[Union[Dict[str, Any], models.Filter]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        group_by: Optional[str] = None,
        group_size: Optional[int] = None) -> Dict[str, List[Document]]
```

Asynchronously run the Embedding Retriever on the given input data.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: A dictionary with filters to narrow down the search space.
- `top_k`: The maximum number of documents to return. If using `group_by` parameters, maximum number of
groups to return.
- `scale_score`: Whether to scale the scores of the retrieved documents or not.
- `return_embedding`: Whether to return the embedding of the retrieved Documents.
- `score_threshold`: A minimal score threshold for the result.
- `group_by`: Payload field to group by, must be a string or number field. If the field contains more than 1
value, all values will be used for grouping. One point can be in multiple groups.
- `group_size`: Maximum amount of points to return per group. Default is 3.

**Raises**:

- `ValueError`: If 'filter_policy' is set to 'MERGE' and 'filters' is a native Qdrant filter.

**Returns**:

The retrieved documents.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantSparseEmbeddingRetriever"></a>

### QdrantSparseEmbeddingRetriever

A component for retrieving documents from an QdrantDocumentStore using sparse vectors.

Usage example:
```python
from haystack_integrations.components.retrievers.qdrant import QdrantSparseEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.dataclasses import Document, SparseEmbedding

document_store = QdrantDocumentStore(
    ":memory:",
    use_sparse_embeddings=True,
    recreate_index=True,
    return_embedding=True,
)

doc = Document(content="test", sparse_embedding=SparseEmbedding(indices=[0, 3, 5], values=[0.1, 0.5, 0.12]))
document_store.write_documents([doc])

retriever = QdrantSparseEmbeddingRetriever(document_store=document_store)
sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
retriever.run(query_sparse_embedding=sparse_embedding)
```

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantSparseEmbeddingRetriever.__init__"></a>

#### QdrantSparseEmbeddingRetriever.\_\_init\_\_

```python
def __init__(document_store: QdrantDocumentStore,
             filters: Optional[Union[Dict[str, Any], models.Filter]] = None,
             top_k: int = 10,
             scale_score: bool = False,
             return_embedding: bool = False,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
             score_threshold: Optional[float] = None,
             group_by: Optional[str] = None,
             group_size: Optional[int] = None) -> None
```

Create a QdrantSparseEmbeddingRetriever component.

**Arguments**:

- `document_store`: An instance of QdrantDocumentStore.
- `filters`: A dictionary with filters to narrow down the search space.
- `top_k`: The maximum number of documents to retrieve. If using `group_by` parameters, maximum number of
groups to return.
- `scale_score`: Whether to scale the scores of the retrieved documents or not.
- `return_embedding`: Whether to return the sparse embedding of the retrieved Documents.
- `filter_policy`: Policy to determine how filters are applied. Defaults to "replace".
- `score_threshold`: A minimal score threshold for the result.
Score of the returned result might be higher or smaller than the threshold
 depending on the Distance function used.
E.g. for cosine similarity only higher scores will be returned.
- `group_by`: Payload field to group by, must be a string or number field. If the field contains more than 1
value, all values will be used for grouping. One point can be in multiple groups.
- `group_size`: Maximum amount of points to return per group. Default is 3.

**Raises**:

- `ValueError`: If `document_store` is not an instance of `QdrantDocumentStore`.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantSparseEmbeddingRetriever.to_dict"></a>

#### QdrantSparseEmbeddingRetriever.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantSparseEmbeddingRetriever.from_dict"></a>

#### QdrantSparseEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "QdrantSparseEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantSparseEmbeddingRetriever.run"></a>

#### QdrantSparseEmbeddingRetriever.run

```python
@component.output_types(documents=List[Document])
def run(query_sparse_embedding: SparseEmbedding,
        filters: Optional[Union[Dict[str, Any], models.Filter]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        group_by: Optional[str] = None,
        group_size: Optional[int] = None) -> Dict[str, List[Document]]
```

Run the Sparse Embedding Retriever on the given input data.

**Arguments**:

- `query_sparse_embedding`: Sparse Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: The maximum number of documents to return. If using `group_by` parameters, maximum number of
groups to return.
- `scale_score`: Whether to scale the scores of the retrieved documents or not.
- `return_embedding`: Whether to return the embedding of the retrieved Documents.
- `score_threshold`: A minimal score threshold for the result.
Score of the returned result might be higher or smaller than the threshold
 depending on the Distance function used.
E.g. for cosine similarity only higher scores will be returned.
- `group_by`: Payload field to group by, must be a string or number field. If the field contains more than 1
value, all values will be used for grouping. One point can be in multiple groups.
- `group_size`: Maximum amount of points to return per group. Default is 3.

**Raises**:

- `ValueError`: If 'filter_policy' is set to 'MERGE' and 'filters' is a native Qdrant filter.

**Returns**:

The retrieved documents.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantSparseEmbeddingRetriever.run_async"></a>

#### QdrantSparseEmbeddingRetriever.run\_async

```python
@component.output_types(documents=List[Document])
async def run_async(
        query_sparse_embedding: SparseEmbedding,
        filters: Optional[Union[Dict[str, Any], models.Filter]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        group_by: Optional[str] = None,
        group_size: Optional[int] = None) -> Dict[str, List[Document]]
```

Asynchronously run the Sparse Embedding Retriever on the given input data.

**Arguments**:

- `query_sparse_embedding`: Sparse Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: The maximum number of documents to return. If using `group_by` parameters, maximum number of
groups to return.
- `scale_score`: Whether to scale the scores of the retrieved documents or not.
- `return_embedding`: Whether to return the embedding of the retrieved Documents.
- `score_threshold`: A minimal score threshold for the result.
Score of the returned result might be higher or smaller than the threshold
 depending on the Distance function used.
E.g. for cosine similarity only higher scores will be returned.
- `group_by`: Payload field to group by, must be a string or number field. If the field contains more than 1
value, all values will be used for grouping. One point can be in multiple groups.
- `group_size`: Maximum amount of points to return per group. Default is 3.

**Raises**:

- `ValueError`: If 'filter_policy' is set to 'MERGE' and 'filters' is a native Qdrant filter.

**Returns**:

The retrieved documents.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantHybridRetriever"></a>

### QdrantHybridRetriever

A component for retrieving documents from an QdrantDocumentStore using both dense and sparse vectors
and fusing the results using Reciprocal Rank Fusion.

Usage example:
```python
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.dataclasses import Document, SparseEmbedding

document_store = QdrantDocumentStore(
    ":memory:",
    use_sparse_embeddings=True,
    recreate_index=True,
    return_embedding=True,
    wait_result_from_api=True,
)

doc = Document(content="test",
               embedding=[0.5]*768,
               sparse_embedding=SparseEmbedding(indices=[0, 3, 5], values=[0.1, 0.5, 0.12]))

document_store.write_documents([doc])

retriever = QdrantHybridRetriever(document_store=document_store)
embedding = [0.1]*768
sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
retriever.run(query_embedding=embedding, query_sparse_embedding=sparse_embedding)
```

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantHybridRetriever.__init__"></a>

#### QdrantHybridRetriever.\_\_init\_\_

```python
def __init__(document_store: QdrantDocumentStore,
             filters: Optional[Union[Dict[str, Any], models.Filter]] = None,
             top_k: int = 10,
             return_embedding: bool = False,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
             score_threshold: Optional[float] = None,
             group_by: Optional[str] = None,
             group_size: Optional[int] = None) -> None
```

Create a QdrantHybridRetriever component.

**Arguments**:

- `document_store`: An instance of QdrantDocumentStore.
- `filters`: A dictionary with filters to narrow down the search space.
- `top_k`: The maximum number of documents to retrieve. If using `group_by` parameters, maximum number of
groups to return.
- `return_embedding`: Whether to return the embeddings of the retrieved Documents.
- `filter_policy`: Policy to determine how filters are applied.
- `score_threshold`: A minimal score threshold for the result.
Score of the returned result might be higher or smaller than the threshold
 depending on the Distance function used.
E.g. for cosine similarity only higher scores will be returned.
- `group_by`: Payload field to group by, must be a string or number field. If the field contains more than 1
value, all values will be used for grouping. One point can be in multiple groups.
- `group_size`: Maximum amount of points to return per group. Default is 3.

**Raises**:

- `ValueError`: If 'document_store' is not an instance of QdrantDocumentStore.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantHybridRetriever.to_dict"></a>

#### QdrantHybridRetriever.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantHybridRetriever.from_dict"></a>

#### QdrantHybridRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "QdrantHybridRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantHybridRetriever.run"></a>

#### QdrantHybridRetriever.run

```python
@component.output_types(documents=List[Document])
def run(query_embedding: List[float],
        query_sparse_embedding: SparseEmbedding,
        filters: Optional[Union[Dict[str, Any], models.Filter]] = None,
        top_k: Optional[int] = None,
        return_embedding: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        group_by: Optional[str] = None,
        group_size: Optional[int] = None) -> Dict[str, List[Document]]
```

Run the Sparse Embedding Retriever on the given input data.

**Arguments**:

- `query_embedding`: Dense embedding of the query.
- `query_sparse_embedding`: Sparse embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: The maximum number of documents to return. If using `group_by` parameters, maximum number of
groups to return.
- `return_embedding`: Whether to return the embedding of the retrieved Documents.
- `score_threshold`: A minimal score threshold for the result.
Score of the returned result might be higher or smaller than the threshold
 depending on the Distance function used.
E.g. for cosine similarity only higher scores will be returned.
- `group_by`: Payload field to group by, must be a string or number field. If the field contains more than 1
value, all values will be used for grouping. One point can be in multiple groups.
- `group_size`: Maximum amount of points to return per group. Default is 3.

**Raises**:

- `ValueError`: If 'filter_policy' is set to 'MERGE' and 'filters' is a native Qdrant filter.

**Returns**:

The retrieved documents.

<a id="haystack_integrations.components.retrievers.qdrant.retriever.QdrantHybridRetriever.run_async"></a>

#### QdrantHybridRetriever.run\_async

```python
@component.output_types(documents=List[Document])
async def run_async(
        query_embedding: List[float],
        query_sparse_embedding: SparseEmbedding,
        filters: Optional[Union[Dict[str, Any], models.Filter]] = None,
        top_k: Optional[int] = None,
        return_embedding: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        group_by: Optional[str] = None,
        group_size: Optional[int] = None) -> Dict[str, List[Document]]
```

Asynchronously run the Sparse Embedding Retriever on the given input data.

**Arguments**:

- `query_embedding`: Dense embedding of the query.
- `query_sparse_embedding`: Sparse embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: The maximum number of documents to return. If using `group_by` parameters, maximum number of
groups to return.
- `return_embedding`: Whether to return the embedding of the retrieved Documents.
- `score_threshold`: A minimal score threshold for the result.
Score of the returned result might be higher or smaller than the threshold
 depending on the Distance function used.
E.g. for cosine similarity only higher scores will be returned.
- `group_by`: Payload field to group by, must be a string or number field. If the field contains more than 1
value, all values will be used for grouping. One point can be in multiple groups.
- `group_size`: Maximum amount of points to return per group. Default is 3.

**Raises**:

- `ValueError`: If 'filter_policy' is set to 'MERGE' and 'filters' is a native Qdrant filter.

**Returns**:

The retrieved documents.

<a id="haystack_integrations.document_stores.qdrant.document_store"></a>

## Module haystack\_integrations.document\_stores.qdrant.document\_store

<a id="haystack_integrations.document_stores.qdrant.document_store.get_batches_from_generator"></a>

#### get\_batches\_from\_generator

```python
def get_batches_from_generator(iterable: List, n: int) -> Generator
```

Batch elements of an iterable into fixed-length chunks or blocks.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore"></a>

### QdrantDocumentStore

A QdrantDocumentStore implementation that you
can use with any Qdrant instance: in-memory, disk-persisted, Docker-based,
and Qdrant Cloud Cluster deployments.

Usage example by creating an in-memory instance:

```python
from haystack.dataclasses.document import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

document_store = QdrantDocumentStore(
    ":memory:",
    recreate_index=True
)
document_store.write_documents([
    Document(content="This is first", embedding=[0.0]*5),
    Document(content="This is second", embedding=[0.1, 0.2, 0.3, 0.4, 0.5])
])
```

Usage example with Qdrant Cloud:

```python
from haystack.dataclasses.document import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

document_store = QdrantDocumentStore(
        url="https://xxxxxx-xxxxx-xxxxx-xxxx-xxxxxxxxx.us-east.aws.cloud.qdrant.io:6333",
    api_key="<your-api-key>",
)
document_store.write_documents([
    Document(content="This is first", embedding=[0.0]*5),
    Document(content="This is second", embedding=[0.1, 0.2, 0.3, 0.4, 0.5])
])
```

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.__init__"></a>

#### QdrantDocumentStore.\_\_init\_\_

```python
def __init__(location: Optional[str] = None,
             url: Optional[str] = None,
             port: int = 6333,
             grpc_port: int = 6334,
             prefer_grpc: bool = False,
             https: Optional[bool] = None,
             api_key: Optional[Secret] = None,
             prefix: Optional[str] = None,
             timeout: Optional[int] = None,
             host: Optional[str] = None,
             path: Optional[str] = None,
             force_disable_check_same_thread: bool = False,
             index: str = "Document",
             embedding_dim: int = 768,
             on_disk: bool = False,
             use_sparse_embeddings: bool = False,
             sparse_idf: bool = False,
             similarity: str = "cosine",
             return_embedding: bool = False,
             progress_bar: bool = True,
             recreate_index: bool = False,
             shard_number: Optional[int] = None,
             replication_factor: Optional[int] = None,
             write_consistency_factor: Optional[int] = None,
             on_disk_payload: Optional[bool] = None,
             hnsw_config: Optional[dict] = None,
             optimizers_config: Optional[dict] = None,
             wal_config: Optional[dict] = None,
             quantization_config: Optional[dict] = None,
             init_from: Optional[dict] = None,
             wait_result_from_api: bool = True,
             metadata: Optional[dict] = None,
             write_batch_size: int = 100,
             scroll_size: int = 10_000,
             payload_fields_to_index: Optional[List[dict]] = None) -> None
```

**Arguments**:

- `location`: If `":memory:"` - use in-memory Qdrant instance.
If `str` - use it as a URL parameter.
If `None` - use default values for host and port.
- `url`: Either host or str of `Optional[scheme], host, Optional[port], Optional[prefix]`.
- `port`: Port of the REST API interface.
- `grpc_port`: Port of the gRPC interface.
- `prefer_grpc`: If `True` - use gRPC interface whenever possible in custom methods.
- `https`: If `True` - use HTTPS(SSL) protocol.
- `api_key`: API key for authentication in Qdrant Cloud.
- `prefix`: If not `None` - add prefix to the REST URL path.
Example: service/v1 will result in http://localhost:6333/service/v1/{qdrant-endpoint}
for REST API.
- `timeout`: Timeout for REST and gRPC API requests.
- `host`: Host name of Qdrant service. If ùrl` and `host` are `None`, set to `localhost`.
- `path`: Persistence path for QdrantLocal.
- `force_disable_check_same_thread`: For QdrantLocal, force disable check_same_thread.
Only use this if you can guarantee that you can resolve the thread safety outside QdrantClient.
- `index`: Name of the index.
- `embedding_dim`: Dimension of the embeddings.
- `on_disk`: Whether to store the collection on disk.
- `use_sparse_embeddings`: If set to `True`, enables support for sparse embeddings.
- `sparse_idf`: If set to `True`, computes the Inverse Document Frequency (IDF) when using sparse embeddings.
It is required to use techniques like BM42. It is ignored if `use_sparse_embeddings` is `False`.
- `similarity`: The similarity metric to use.
- `return_embedding`: Whether to return embeddings in the search results.
- `progress_bar`: Whether to show a progress bar or not.
- `recreate_index`: Whether to recreate the index.
- `shard_number`: Number of shards in the collection.
- `replication_factor`: Replication factor for the collection.
Defines how many copies of each shard will be created. Effective only in distributed mode.
- `write_consistency_factor`: Write consistency factor for the collection. Minimum value is 1.
Defines how many replicas should apply to the operation for it to be considered successful.
Increasing this number makes the collection more resilient to inconsistencies
but will cause failures if not enough replicas are available.
Effective only in distributed mode.
- `on_disk_payload`: If `True`, the point's payload will not be stored in memory and
will be read from the disk every time it is requested.
This setting saves RAM by slightly increasing response time.
Note: indexed payload values remain in RAM.
- `hnsw_config`: Params for HNSW index.
- `optimizers_config`: Params for optimizer.
- `wal_config`: Params for Write-Ahead-Log.
- `quantization_config`: Params for quantization. If `None`, quantization will be disabled.
- `init_from`: Use data stored in another collection to initialize this collection.
- `wait_result_from_api`: Whether to wait for the result from the API after each request.
- `metadata`: Additional metadata to include with the documents.
- `write_batch_size`: The batch size for writing documents.
- `scroll_size`: The scroll size for reading documents.
- `payload_fields_to_index`: List of payload fields to index.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.count_documents"></a>

#### QdrantDocumentStore.count\_documents

```python
def count_documents() -> int
```

Returns the number of documents present in the Document Store.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.count_documents_async"></a>

#### QdrantDocumentStore.count\_documents\_async

```python
async def count_documents_async() -> int
```

Asynchronously returns the number of documents present in the document dtore.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.filter_documents"></a>

#### QdrantDocumentStore.filter\_documents

```python
def filter_documents(
    filters: Optional[Union[Dict[str, Any], rest.Filter]] = None
) -> List[Document]
```

Returns the documents that match the provided filters.

For a detailed specification of the filters, refer to the
[documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Arguments**:

- `filters`: The filters to apply to the document list.

**Returns**:

A list of documents that match the given filters.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.filter_documents_async"></a>

#### QdrantDocumentStore.filter\_documents\_async

```python
async def filter_documents_async(
    filters: Optional[Union[Dict[str, Any], rest.Filter]] = None
) -> List[Document]
```

Asynchronously returns the documents that match the provided filters.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.write_documents"></a>

#### QdrantDocumentStore.write\_documents

```python
def write_documents(documents: List[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int
```

Writes documents to Qdrant using the specified policy.

The QdrantDocumentStore can handle duplicate documents based on the given policy.
The available policies are:
- `FAIL`: The operation will raise an error if any document already exists.
- `OVERWRITE`: Existing documents will be overwritten with the new ones.
- `SKIP`: Existing documents will be skipped, and only new documents will be added.

**Arguments**:

- `documents`: A list of Document objects to write to Qdrant.
- `policy`: The policy for handling duplicate documents.

**Returns**:

The number of documents written to the document store.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.write_documents_async"></a>

#### QdrantDocumentStore.write\_documents\_async

```python
async def write_documents_async(
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int
```

Asynchronously writes documents to Qdrant using the specified policy.

The QdrantDocumentStore can handle duplicate documents based on the given policy.
The available policies are:
- `FAIL`: The operation will raise an error if any document already exists.
- `OVERWRITE`: Existing documents will be overwritten with the new ones.
- `SKIP`: Existing documents will be skipped, and only new documents will be added.

**Arguments**:

- `documents`: A list of Document objects to write to Qdrant.
- `policy`: The policy for handling duplicate documents.

**Returns**:

The number of documents written to the document store.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.delete_documents"></a>

#### QdrantDocumentStore.delete\_documents

```python
def delete_documents(document_ids: List[str]) -> None
```

Deletes documents that match the provided `document_ids` from the document store.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.delete_documents_async"></a>

#### QdrantDocumentStore.delete\_documents\_async

```python
async def delete_documents_async(document_ids: List[str]) -> None
```

Asynchronously deletes documents that match the provided `document_ids` from the document store.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.from_dict"></a>

#### QdrantDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "QdrantDocumentStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.to_dict"></a>

#### QdrantDocumentStore.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.get_documents_by_id"></a>

#### QdrantDocumentStore.get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str]) -> List[Document]
```

Retrieves documents from Qdrant by their IDs.

**Arguments**:

- `ids`: A list of document IDs to retrieve.

**Returns**:

A list of documents.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.get_documents_by_id_async"></a>

#### QdrantDocumentStore.get\_documents\_by\_id\_async

```python
async def get_documents_by_id_async(ids: List[str]) -> List[Document]
```

Retrieves documents from Qdrant by their IDs.

**Arguments**:

- `ids`: A list of document IDs to retrieve.

**Returns**:

A list of documents.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.get_distance"></a>

#### QdrantDocumentStore.get\_distance

```python
def get_distance(similarity: str) -> rest.Distance
```

Retrieves the distance metric for the specified similarity measure.

**Arguments**:

- `similarity`: The similarity measure to retrieve the distance.

**Raises**:

- `QdrantStoreError`: If the provided similarity measure is not supported.

**Returns**:

The corresponding rest.Distance object.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.recreate_collection"></a>

#### QdrantDocumentStore.recreate\_collection

```python
def recreate_collection(collection_name: str,
                        distance: rest.Distance,
                        embedding_dim: int,
                        on_disk: Optional[bool] = None,
                        use_sparse_embeddings: Optional[bool] = None,
                        sparse_idf: bool = False) -> None
```

Recreates the Qdrant collection with the specified parameters.

**Arguments**:

- `collection_name`: The name of the collection to recreate.
- `distance`: The distance metric to use for the collection.
- `embedding_dim`: The dimension of the embeddings.
- `on_disk`: Whether to store the collection on disk.
- `use_sparse_embeddings`: Whether to use sparse embeddings.
- `sparse_idf`: Whether to compute the Inverse Document Frequency (IDF) when using sparse embeddings. Required for BM42.

<a id="haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore.recreate_collection_async"></a>

#### QdrantDocumentStore.recreate\_collection\_async

```python
async def recreate_collection_async(
        collection_name: str,
        distance: rest.Distance,
        embedding_dim: int,
        on_disk: Optional[bool] = None,
        use_sparse_embeddings: Optional[bool] = None,
        sparse_idf: bool = False) -> None
```

Asynchronously recreates the Qdrant collection with the specified parameters.

**Arguments**:

- `collection_name`: The name of the collection to recreate.
- `distance`: The distance metric to use for the collection.
- `embedding_dim`: The dimension of the embeddings.
- `on_disk`: Whether to store the collection on disk.
- `use_sparse_embeddings`: Whether to use sparse embeddings.
- `sparse_idf`: Whether to compute the Inverse Document Frequency (IDF) when using sparse embeddings. Required for BM42.

<a id="haystack_integrations.document_stores.qdrant.migrate_to_sparse"></a>

## Module haystack\_integrations.document\_stores.qdrant.migrate\_to\_sparse

<a id="haystack_integrations.document_stores.qdrant.migrate_to_sparse.migrate_to_sparse_embeddings_support"></a>

#### migrate\_to\_sparse\_embeddings\_support

```python
def migrate_to_sparse_embeddings_support(
        old_document_store: QdrantDocumentStore, new_index: str) -> None
```

Utility function to migrate an existing `QdrantDocumentStore` to a new one with support for sparse embeddings.

With qdrant-hasytack v3.3.0, support for sparse embeddings has been added to `QdrantDocumentStore`.
This feature is disabled by default and can be enabled by setting `use_sparse_embeddings=True` in the init
parameters. To store sparse embeddings, Document stores/collections created with this feature disabled must be
migrated to a new collection with the feature enabled.

This utility function applies to on-premise and cloud instances of Qdrant.
It does not work for local in-memory/disk-persisted instances.

The utility function merely migrates the existing documents so that they are ready to store sparse embeddings.
It does not compute sparse embeddings. To do this, you need to use a Sparse Embedder component.

Example usage:
```python
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.document_stores.qdrant import migrate_to_sparse_embeddings_support

old_document_store = QdrantDocumentStore(url="http://localhost:6333",
                                         index="Document",
                                         use_sparse_embeddings=False)
new_index = "Document_sparse"

migrate_to_sparse_embeddings_support(old_document_store, new_index)

# now you can use the new document store with sparse embeddings support
new_document_store = QdrantDocumentStore(url="http://localhost:6333",
                                         index=new_index,
                                         use_sparse_embeddings=True)
```

**Arguments**:

- `old_document_store`: The existing QdrantDocumentStore instance to migrate from.
- `new_index`: The name of the new index/collection to create with sparse embeddings support.

