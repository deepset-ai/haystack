---
title: "Qdrant"
id: integrations-qdrant
description: "Qdrant integration for Haystack"
slug: "/integrations-qdrant"
---


## `haystack_integrations.components.retrievers.qdrant.retriever`

### `QdrantEmbeddingRetriever`

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

#### `__init__`

```python
__init__(
    document_store: QdrantDocumentStore,
    filters: dict[str, Any] | models.Filter | None = None,
    top_k: int = 10,
    scale_score: bool = False,
    return_embedding: bool = False,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    score_threshold: float | None = None,
    group_by: str | None = None,
    group_size: int | None = None,
) -> None
```

Create a QdrantEmbeddingRetriever component.

**Parameters:**

- **document_store** (<code>QdrantDocumentStore</code>) – An instance of QdrantDocumentStore.
- **filters** (<code>dict\[str, Any\] | Filter | None</code>) – A dictionary with filters to narrow down the search space.
- **top_k** (<code>int</code>) – The maximum number of documents to retrieve. If using `group_by` parameters, maximum number of
  groups to return.
- **scale_score** (<code>bool</code>) – Whether to scale the scores of the retrieved documents or not.
- **return_embedding** (<code>bool</code>) – Whether to return the embedding of the retrieved Documents.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.
- **score_threshold** (<code>float | None</code>) – A minimal score threshold for the result.
  Score of the returned result might be higher or smaller than the threshold
  depending on the `similarity` function specified in the Document Store.
  E.g. for cosine similarity only higher scores will be returned.
- **group_by** (<code>str | None</code>) – Payload field to group by, must be a string or number field. If the field contains more than 1
  value, all values will be used for grouping. One point can be in multiple groups.
- **group_size** (<code>int | None</code>) – Maximum amount of points to return per group. Default is 3.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of `QdrantDocumentStore`.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> QdrantEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>QdrantEmbeddingRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | models.Filter | None = None,
    top_k: int | None = None,
    scale_score: bool | None = None,
    return_embedding: bool | None = None,
    score_threshold: float | None = None,
    group_by: str | None = None,
    group_size: int | None = None,
) -> dict[str, list[Document]]
```

Run the Embedding Retriever on the given input data.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | Filter | None</code>) – A dictionary with filters to narrow down the search space.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return. If using `group_by` parameters, maximum number of
  groups to return.
- **scale_score** (<code>bool | None</code>) – Whether to scale the scores of the retrieved documents or not.
- **return_embedding** (<code>bool | None</code>) – Whether to return the embedding of the retrieved Documents.
- **score_threshold** (<code>float | None</code>) – A minimal score threshold for the result.
- **group_by** (<code>str | None</code>) – Payload field to group by, must be a string or number field. If the field contains more than 1
  value, all values will be used for grouping. One point can be in multiple groups.
- **group_size** (<code>int | None</code>) – Maximum amount of points to return per group. Default is 3.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – The retrieved documents.

**Raises:**

- <code>ValueError</code> – If 'filter_policy' is set to 'MERGE' and 'filters' is a native Qdrant filter.

#### `run_async`

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | models.Filter | None = None,
    top_k: int | None = None,
    scale_score: bool | None = None,
    return_embedding: bool | None = None,
    score_threshold: float | None = None,
    group_by: str | None = None,
    group_size: int | None = None,
) -> dict[str, list[Document]]
```

Asynchronously run the Embedding Retriever on the given input data.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | Filter | None</code>) – A dictionary with filters to narrow down the search space.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return. If using `group_by` parameters, maximum number of
  groups to return.
- **scale_score** (<code>bool | None</code>) – Whether to scale the scores of the retrieved documents or not.
- **return_embedding** (<code>bool | None</code>) – Whether to return the embedding of the retrieved Documents.
- **score_threshold** (<code>float | None</code>) – A minimal score threshold for the result.
- **group_by** (<code>str | None</code>) – Payload field to group by, must be a string or number field. If the field contains more than 1
  value, all values will be used for grouping. One point can be in multiple groups.
- **group_size** (<code>int | None</code>) – Maximum amount of points to return per group. Default is 3.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – The retrieved documents.

**Raises:**

- <code>ValueError</code> – If 'filter_policy' is set to 'MERGE' and 'filters' is a native Qdrant filter.

### `QdrantSparseEmbeddingRetriever`

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

#### `__init__`

```python
__init__(
    document_store: QdrantDocumentStore,
    filters: dict[str, Any] | models.Filter | None = None,
    top_k: int = 10,
    scale_score: bool = False,
    return_embedding: bool = False,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    score_threshold: float | None = None,
    group_by: str | None = None,
    group_size: int | None = None,
) -> None
```

Create a QdrantSparseEmbeddingRetriever component.

**Parameters:**

- **document_store** (<code>QdrantDocumentStore</code>) – An instance of QdrantDocumentStore.
- **filters** (<code>dict\[str, Any\] | Filter | None</code>) – A dictionary with filters to narrow down the search space.
- **top_k** (<code>int</code>) – The maximum number of documents to retrieve. If using `group_by` parameters, maximum number of
  groups to return.
- **scale_score** (<code>bool</code>) – Whether to scale the scores of the retrieved documents or not.
- **return_embedding** (<code>bool</code>) – Whether to return the sparse embedding of the retrieved Documents.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied. Defaults to "replace".
- **score_threshold** (<code>float | None</code>) – A minimal score threshold for the result.
  Score of the returned result might be higher or smaller than the threshold
  depending on the Distance function used.
  E.g. for cosine similarity only higher scores will be returned.
- **group_by** (<code>str | None</code>) – Payload field to group by, must be a string or number field. If the field contains more than 1
  value, all values will be used for grouping. One point can be in multiple groups.
- **group_size** (<code>int | None</code>) – Maximum amount of points to return per group. Default is 3.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of `QdrantDocumentStore`.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> QdrantSparseEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>QdrantSparseEmbeddingRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query_sparse_embedding: SparseEmbedding,
    filters: dict[str, Any] | models.Filter | None = None,
    top_k: int | None = None,
    scale_score: bool | None = None,
    return_embedding: bool | None = None,
    score_threshold: float | None = None,
    group_by: str | None = None,
    group_size: int | None = None,
) -> dict[str, list[Document]]
```

Run the Sparse Embedding Retriever on the given input data.

**Parameters:**

- **query_sparse_embedding** (<code>SparseEmbedding</code>) – Sparse Embedding of the query.
- **filters** (<code>dict\[str, Any\] | Filter | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return. If using `group_by` parameters, maximum number of
  groups to return.
- **scale_score** (<code>bool | None</code>) – Whether to scale the scores of the retrieved documents or not.
- **return_embedding** (<code>bool | None</code>) – Whether to return the embedding of the retrieved Documents.
- **score_threshold** (<code>float | None</code>) – A minimal score threshold for the result.
  Score of the returned result might be higher or smaller than the threshold
  depending on the Distance function used.
  E.g. for cosine similarity only higher scores will be returned.
- **group_by** (<code>str | None</code>) – Payload field to group by, must be a string or number field. If the field contains more than 1
  value, all values will be used for grouping. One point can be in multiple groups.
- **group_size** (<code>int | None</code>) – Maximum amount of points to return per group. Default is 3.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – The retrieved documents.

**Raises:**

- <code>ValueError</code> – If 'filter_policy' is set to 'MERGE' and 'filters' is a native Qdrant filter.

#### `run_async`

```python
run_async(
    query_sparse_embedding: SparseEmbedding,
    filters: dict[str, Any] | models.Filter | None = None,
    top_k: int | None = None,
    scale_score: bool | None = None,
    return_embedding: bool | None = None,
    score_threshold: float | None = None,
    group_by: str | None = None,
    group_size: int | None = None,
) -> dict[str, list[Document]]
```

Asynchronously run the Sparse Embedding Retriever on the given input data.

**Parameters:**

- **query_sparse_embedding** (<code>SparseEmbedding</code>) – Sparse Embedding of the query.
- **filters** (<code>dict\[str, Any\] | Filter | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return. If using `group_by` parameters, maximum number of
  groups to return.
- **scale_score** (<code>bool | None</code>) – Whether to scale the scores of the retrieved documents or not.
- **return_embedding** (<code>bool | None</code>) – Whether to return the embedding of the retrieved Documents.
- **score_threshold** (<code>float | None</code>) – A minimal score threshold for the result.
  Score of the returned result might be higher or smaller than the threshold
  depending on the Distance function used.
  E.g. for cosine similarity only higher scores will be returned.
- **group_by** (<code>str | None</code>) – Payload field to group by, must be a string or number field. If the field contains more than 1
  value, all values will be used for grouping. One point can be in multiple groups.
- **group_size** (<code>int | None</code>) – Maximum amount of points to return per group. Default is 3.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – The retrieved documents.

**Raises:**

- <code>ValueError</code> – If 'filter_policy' is set to 'MERGE' and 'filters' is a native Qdrant filter.

### `QdrantHybridRetriever`

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

#### `__init__`

```python
__init__(
    document_store: QdrantDocumentStore,
    filters: dict[str, Any] | models.Filter | None = None,
    top_k: int = 10,
    return_embedding: bool = False,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    score_threshold: float | None = None,
    group_by: str | None = None,
    group_size: int | None = None,
) -> None
```

Create a QdrantHybridRetriever component.

**Parameters:**

- **document_store** (<code>QdrantDocumentStore</code>) – An instance of QdrantDocumentStore.
- **filters** (<code>dict\[str, Any\] | Filter | None</code>) – A dictionary with filters to narrow down the search space.
- **top_k** (<code>int</code>) – The maximum number of documents to retrieve. If using `group_by` parameters, maximum number of
  groups to return.
- **return_embedding** (<code>bool</code>) – Whether to return the embeddings of the retrieved Documents.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.
- **score_threshold** (<code>float | None</code>) – A minimal score threshold for the result.
  Score of the returned result might be higher or smaller than the threshold
  depending on the Distance function used.
  E.g. for cosine similarity only higher scores will be returned.
- **group_by** (<code>str | None</code>) – Payload field to group by, must be a string or number field. If the field contains more than 1
  value, all values will be used for grouping. One point can be in multiple groups.
- **group_size** (<code>int | None</code>) – Maximum amount of points to return per group. Default is 3.

**Raises:**

- <code>ValueError</code> – If 'document_store' is not an instance of QdrantDocumentStore.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> QdrantHybridRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>QdrantHybridRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query_embedding: list[float],
    query_sparse_embedding: SparseEmbedding,
    filters: dict[str, Any] | models.Filter | None = None,
    top_k: int | None = None,
    return_embedding: bool | None = None,
    score_threshold: float | None = None,
    group_by: str | None = None,
    group_size: int | None = None,
) -> dict[str, list[Document]]
```

Run the Sparse Embedding Retriever on the given input data.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Dense embedding of the query.
- **query_sparse_embedding** (<code>SparseEmbedding</code>) – Sparse embedding of the query.
- **filters** (<code>dict\[str, Any\] | Filter | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return. If using `group_by` parameters, maximum number of
  groups to return.
- **return_embedding** (<code>bool | None</code>) – Whether to return the embedding of the retrieved Documents.
- **score_threshold** (<code>float | None</code>) – A minimal score threshold for the result.
  Score of the returned result might be higher or smaller than the threshold
  depending on the Distance function used.
  E.g. for cosine similarity only higher scores will be returned.
- **group_by** (<code>str | None</code>) – Payload field to group by, must be a string or number field. If the field contains more than 1
  value, all values will be used for grouping. One point can be in multiple groups.
- **group_size** (<code>int | None</code>) – Maximum amount of points to return per group. Default is 3.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – The retrieved documents.

**Raises:**

- <code>ValueError</code> – If 'filter_policy' is set to 'MERGE' and 'filters' is a native Qdrant filter.

#### `run_async`

```python
run_async(
    query_embedding: list[float],
    query_sparse_embedding: SparseEmbedding,
    filters: dict[str, Any] | models.Filter | None = None,
    top_k: int | None = None,
    return_embedding: bool | None = None,
    score_threshold: float | None = None,
    group_by: str | None = None,
    group_size: int | None = None,
) -> dict[str, list[Document]]
```

Asynchronously run the Sparse Embedding Retriever on the given input data.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Dense embedding of the query.
- **query_sparse_embedding** (<code>SparseEmbedding</code>) – Sparse embedding of the query.
- **filters** (<code>dict\[str, Any\] | Filter | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to return. If using `group_by` parameters, maximum number of
  groups to return.
- **return_embedding** (<code>bool | None</code>) – Whether to return the embedding of the retrieved Documents.
- **score_threshold** (<code>float | None</code>) – A minimal score threshold for the result.
  Score of the returned result might be higher or smaller than the threshold
  depending on the Distance function used.
  E.g. for cosine similarity only higher scores will be returned.
- **group_by** (<code>str | None</code>) – Payload field to group by, must be a string or number field. If the field contains more than 1
  value, all values will be used for grouping. One point can be in multiple groups.
- **group_size** (<code>int | None</code>) – Maximum amount of points to return per group. Default is 3.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – The retrieved documents.

**Raises:**

- <code>ValueError</code> – If 'filter_policy' is set to 'MERGE' and 'filters' is a native Qdrant filter.

## `haystack_integrations.document_stores.qdrant.document_store`

### `get_batches_from_generator`

```python
get_batches_from_generator(iterable: list, n: int) -> Generator
```

Batch elements of an iterable into fixed-length chunks or blocks.

### `QdrantDocumentStore`

A QdrantDocumentStore implementation that you can use with any Qdrant instance: in-memory, disk-persisted,
Docker-based, and Qdrant Cloud Cluster deployments.

Usage example by creating an in-memory instance:

```python
from haystack.dataclasses.document import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

document_store = QdrantDocumentStore(
    ":memory:",
    recreate_index=True,
    embedding_dim=5
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

#### `__init__`

```python
__init__(
    location: str | None = None,
    url: str | None = None,
    port: int = 6333,
    grpc_port: int = 6334,
    prefer_grpc: bool = False,
    https: bool | None = None,
    api_key: Secret | None = None,
    prefix: str | None = None,
    timeout: int | None = None,
    host: str | None = None,
    path: str | None = None,
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
    shard_number: int | None = None,
    replication_factor: int | None = None,
    write_consistency_factor: int | None = None,
    on_disk_payload: bool | None = None,
    hnsw_config: dict | None = None,
    optimizers_config: dict | None = None,
    wal_config: dict | None = None,
    quantization_config: dict | None = None,
    wait_result_from_api: bool = True,
    metadata: dict | None = None,
    write_batch_size: int = 100,
    scroll_size: int = 10000,
    payload_fields_to_index: list[dict] | None = None,
) -> None
```

Initializes a QdrantDocumentStore.

**Parameters:**

- **location** (<code>str | None</code>) – If `":memory:"` - use in-memory Qdrant instance.
  If `str` - use it as a URL parameter.
  If `None` - use default values for host and port.
- **url** (<code>str | None</code>) – Either host or str of `Optional[scheme], host, Optional[port], Optional[prefix]`.
- **port** (<code>int</code>) – Port of the REST API interface.
- **grpc_port** (<code>int</code>) – Port of the gRPC interface.
- **prefer_grpc** (<code>bool</code>) – If `True` - use gRPC interface whenever possible in custom methods.
- **https** (<code>bool | None</code>) – If `True` - use HTTPS(SSL) protocol.
- **api_key** (<code>Secret | None</code>) – API key for authentication in Qdrant Cloud.
- **prefix** (<code>str | None</code>) – If not `None` - add prefix to the REST URL path.
  Example: service/v1 will result in http://localhost:6333/service/v1/{qdrant-endpoint}
  for REST API.
- **timeout** (<code>int | None</code>) – Timeout for REST and gRPC API requests.
- **host** (<code>str | None</code>) – Host name of Qdrant service. If ùrl`and`host`are`None`, set to `localhost\`.
- **path** (<code>str | None</code>) – Persistence path for QdrantLocal.
- **force_disable_check_same_thread** (<code>bool</code>) – For QdrantLocal, force disable check_same_thread.
  Only use this if you can guarantee that you can resolve the thread safety outside QdrantClient.
- **index** (<code>str</code>) – Name of the index.
- **embedding_dim** (<code>int</code>) – Dimension of the embeddings.
- **on_disk** (<code>bool</code>) – Whether to store the collection on disk.
- **use_sparse_embeddings** (<code>bool</code>) – If set to `True`, enables support for sparse embeddings.
- **sparse_idf** (<code>bool</code>) – If set to `True`, computes the Inverse Document Frequency (IDF) when using sparse embeddings.
  It is required to use techniques like BM42. It is ignored if `use_sparse_embeddings` is `False`.
- **similarity** (<code>str</code>) – The similarity metric to use.
- **return_embedding** (<code>bool</code>) – Whether to return embeddings in the search results.
- **progress_bar** (<code>bool</code>) – Whether to show a progress bar or not.
- **recreate_index** (<code>bool</code>) – Whether to recreate the index.
- **shard_number** (<code>int | None</code>) – Number of shards in the collection.
- **replication_factor** (<code>int | None</code>) – Replication factor for the collection.
  Defines how many copies of each shard will be created. Effective only in distributed mode.
- **write_consistency_factor** (<code>int | None</code>) – Write consistency factor for the collection. Minimum value is 1.
  Defines how many replicas should apply to the operation for it to be considered successful.
  Increasing this number makes the collection more resilient to inconsistencies
  but will cause failures if not enough replicas are available.
  Effective only in distributed mode.
- **on_disk_payload** (<code>bool | None</code>) – If `True`, the point's payload will not be stored in memory and
  will be read from the disk every time it is requested.
  This setting saves RAM by slightly increasing response time.
  Note: indexed payload values remain in RAM.
- **hnsw_config** (<code>dict | None</code>) – Params for HNSW index.
- **optimizers_config** (<code>dict | None</code>) – Params for optimizer.
- **wal_config** (<code>dict | None</code>) – Params for Write-Ahead-Log.
- **quantization_config** (<code>dict | None</code>) – Params for quantization. If `None`, quantization will be disabled.
- **wait_result_from_api** (<code>bool</code>) – Whether to wait for the result from the API after each request.
- **metadata** (<code>dict | None</code>) – Additional metadata to include with the documents.
- **write_batch_size** (<code>int</code>) – The batch size for writing documents.
- **scroll_size** (<code>int</code>) – The scroll size for reading documents.
- **payload_fields_to_index** (<code>list\[dict\] | None</code>) – List of payload fields to index.

#### `count_documents`

```python
count_documents() -> int
```

Returns the number of documents present in the Document Store.

#### `count_documents_async`

```python
count_documents_async() -> int
```

Asynchronously returns the number of documents present in the document dtore.

#### `filter_documents`

```python
filter_documents(
    filters: dict[str, Any] | rest.Filter | None = None,
) -> list[Document]
```

Returns the documents that match the provided filters.

For a detailed specification of the filters, refer to the
[documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Parameters:**

- **filters** (<code>dict\[str, Any\] | Filter | None</code>) – The filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – A list of documents that match the given filters.

#### `filter_documents_async`

```python
filter_documents_async(
    filters: dict[str, Any] | rest.Filter | None = None,
) -> list[Document]
```

Asynchronously returns the documents that match the provided filters.

#### `write_documents`

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL
) -> int
```

Writes documents to Qdrant using the specified policy.
The QdrantDocumentStore can handle duplicate documents based on the given policy.
The available policies are:

- `FAIL`: The operation will raise an error if any document already exists.
- `OVERWRITE`: Existing documents will be overwritten with the new ones.
- `SKIP`: Existing documents will be skipped, and only new documents will be added.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Document objects to write to Qdrant.
- **policy** (<code>DuplicatePolicy</code>) – The policy for handling duplicate documents.

**Returns:**

- <code>int</code> – The number of documents written to the document store.

#### `write_documents_async`

```python
write_documents_async(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL
) -> int
```

Asynchronously writes documents to Qdrant using the specified policy.
The QdrantDocumentStore can handle duplicate documents based on the given policy.
The available policies are:

- `FAIL`: The operation will raise an error if any document already exists.
- `OVERWRITE`: Existing documents will be overwritten with the new ones.
- `SKIP`: Existing documents will be skipped, and only new documents will be added.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Document objects to write to Qdrant.
- **policy** (<code>DuplicatePolicy</code>) – The policy for handling duplicate documents.

**Returns:**

- <code>int</code> – The number of documents written to the document store.

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

**Note**: This operation is not atomic. Documents matching the filter are fetched first,
then updated. If documents are modified between the fetch and update operations,
those changes may be lost.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update. This will be merged with existing metadata.

**Returns:**

- <code>int</code> – The number of documents updated.

#### `update_by_filter_async`

```python
update_by_filter_async(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Asynchronously updates the metadata of all documents that match the provided filters.

**Note**: This operation is not atomic. Documents matching the filter are fetched first,
then updated. If documents are modified between the fetch and update operations,
those changes may be lost.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update. This will be merged with existing metadata.

**Returns:**

- <code>int</code> – The number of documents updated.

#### `delete_all_documents`

```python
delete_all_documents(recreate_index: bool = False) -> None
```

Deletes all documents from the document store.

**Parameters:**

- **recreate_index** (<code>bool</code>) – Whether to recreate the index after deleting all documents.

#### `delete_all_documents_async`

```python
delete_all_documents_async(recreate_index: bool = False) -> None
```

Asynchronously deletes all documents from the document store.

**Parameters:**

- **recreate_index** (<code>bool</code>) – Whether to recreate the index after deleting all documents.

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

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for counting.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns:**

- <code>int</code> – The number of documents that match the filters.

#### `get_metadata_fields_info`

```python
get_metadata_fields_info() -> dict[str, str]
```

Returns the information about the fields from the collection.

**Returns:**

- <code>dict\[str, str\]</code> – A dictionary mapping field names to their types e.g.:

```python
{"field_name": "integer"}
```

#### `get_metadata_fields_info_async`

```python
get_metadata_fields_info_async() -> dict[str, str]
```

Asynchronously returns the information about the fields from the collection.

**Returns:**

- <code>dict\[str, str\]</code> – A dictionary mapping field names to their types e.g.:

```python
{"field_name": "integer"}
```

#### `get_metadata_field_min_max`

```python
get_metadata_field_min_max(metadata_field: str) -> dict[str, Any]
```

Returns the minimum and maximum values for the given metadata field.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field key (inside `meta`) to get the minimum and maximum values for.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the keys "min" and "max", where each value is the minimum or maximum value of the
  metadata field across all documents. Returns an empty dict if no documents have the field.

#### `get_metadata_field_min_max_async`

```python
get_metadata_field_min_max_async(metadata_field: str) -> dict[str, Any]
```

Asynchronously returns the minimum and maximum values for the given metadata field.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field key (inside `meta`) to get the minimum and maximum values for.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the keys "min" and "max", where each value is the minimum or maximum value of the
  metadata field across all documents. Returns an empty dict if no documents have the field.

#### `count_unique_metadata_by_filter`

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Returns the number of unique values for each specified metadata field among documents that match the filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to restrict the documents considered.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **metadata_fields** (<code>list\[str\]</code>) – List of metadata field keys (inside `meta`) to count unique values for.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary mapping each metadata field name to the count of its unique values among the filtered
  documents.

#### `count_unique_metadata_by_filter_async`

```python
count_unique_metadata_by_filter_async(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Asynchronously returns the number of unique values for each specified metadata field among documents that
match the filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to restrict the documents considered.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **metadata_fields** (<code>list\[str\]</code>) – List of metadata field keys (inside `meta`) to count unique values for.

**Returns:**

- <code>dict\[str, int\]</code> – A dictionary mapping each metadata field name to the count of its unique values among the filtered
  documents.

#### `get_metadata_field_unique_values`

```python
get_metadata_field_unique_values(
    metadata_field: str,
    filters: dict[str, Any] | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Any]
```

Returns unique values for a metadata field, with optional filters and offset/limit pagination.

Unique values are ordered by first occurrence during scroll. Pagination is offset-based over that order.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field key (inside `meta`) to get unique values for.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional filters to restrict the documents considered.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **limit** (<code>int</code>) – Maximum number of unique values to return per page. Defaults to 100.
- **offset** (<code>int</code>) – Number of unique values to skip (for pagination). Defaults to 0.

**Returns:**

- <code>list\[Any\]</code> – A list of unique values for the field (at most `limit` items, starting at `offset`).

#### `get_metadata_field_unique_values_async`

```python
get_metadata_field_unique_values_async(
    metadata_field: str,
    filters: dict[str, Any] | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Any]
```

Asynchronously returns unique values for a metadata field, with optional filters and offset/limit pagination.

Unique values are ordered by first occurrence during scroll. Pagination is offset-based over that order.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field key (inside `meta`) to get unique values for.
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional filters to restrict the documents considered.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **limit** (<code>int</code>) – Maximum number of unique values to return per page. Defaults to 100.
- **offset** (<code>int</code>) – Number of unique values to skip (for pagination). Defaults to 0.

**Returns:**

- <code>list\[Any\]</code> – A list of unique values for the field (at most `limit` items, starting at `offset`).

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> QdrantDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>QdrantDocumentStore</code> – The deserialized component.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `get_documents_by_id`

```python
get_documents_by_id(ids: list[str]) -> list[Document]
```

Retrieves documents from Qdrant by their IDs.

**Parameters:**

- **ids** (<code>list\[str\]</code>) – A list of document IDs to retrieve.

**Returns:**

- <code>list\[Document\]</code> – A list of documents.

#### `get_documents_by_id_async`

```python
get_documents_by_id_async(ids: list[str]) -> list[Document]
```

Retrieves documents from Qdrant by their IDs.

**Parameters:**

- **ids** (<code>list\[str\]</code>) – A list of document IDs to retrieve.

**Returns:**

- <code>list\[Document\]</code> – A list of documents.

#### `get_distance`

```python
get_distance(similarity: str) -> rest.Distance
```

Retrieves the distance metric for the specified similarity measure.

**Parameters:**

- **similarity** (<code>str</code>) – The similarity measure to retrieve the distance.

**Returns:**

- <code>Distance</code> – The corresponding rest.Distance object.

**Raises:**

- <code>QdrantStoreError</code> – If the provided similarity measure is not supported.

#### `recreate_collection`

```python
recreate_collection(
    collection_name: str,
    distance: rest.Distance,
    embedding_dim: int,
    on_disk: bool | None = None,
    use_sparse_embeddings: bool | None = None,
    sparse_idf: bool = False,
) -> None
```

Recreates the Qdrant collection with the specified parameters.

**Parameters:**

- **collection_name** (<code>str</code>) – The name of the collection to recreate.
- **distance** (<code>Distance</code>) – The distance metric to use for the collection.
- **embedding_dim** (<code>int</code>) – The dimension of the embeddings.
- **on_disk** (<code>bool | None</code>) – Whether to store the collection on disk.
- **use_sparse_embeddings** (<code>bool | None</code>) – Whether to use sparse embeddings.
- **sparse_idf** (<code>bool</code>) – Whether to compute the Inverse Document Frequency (IDF) when using sparse embeddings. Required for BM42.

#### `recreate_collection_async`

```python
recreate_collection_async(
    collection_name: str,
    distance: rest.Distance,
    embedding_dim: int,
    on_disk: bool | None = None,
    use_sparse_embeddings: bool | None = None,
    sparse_idf: bool = False,
) -> None
```

Asynchronously recreates the Qdrant collection with the specified parameters.

**Parameters:**

- **collection_name** (<code>str</code>) – The name of the collection to recreate.
- **distance** (<code>Distance</code>) – The distance metric to use for the collection.
- **embedding_dim** (<code>int</code>) – The dimension of the embeddings.
- **on_disk** (<code>bool | None</code>) – Whether to store the collection on disk.
- **use_sparse_embeddings** (<code>bool | None</code>) – Whether to use sparse embeddings.
- **sparse_idf** (<code>bool</code>) – Whether to compute the Inverse Document Frequency (IDF) when using sparse embeddings. Required for BM42.

## `haystack_integrations.document_stores.qdrant.migrate_to_sparse`

### `migrate_to_sparse_embeddings_support`

```python
migrate_to_sparse_embeddings_support(
    old_document_store: QdrantDocumentStore, new_index: str
) -> None
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

**Parameters:**

- **old_document_store** (<code>QdrantDocumentStore</code>) – The existing QdrantDocumentStore instance to migrate from.
- **new_index** (<code>str</code>) – The name of the new index/collection to create with sparse embeddings support.
