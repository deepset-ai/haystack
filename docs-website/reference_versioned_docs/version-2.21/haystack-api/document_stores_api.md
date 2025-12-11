---
title: "Document Stores"
id: document-stores-api
description: "Stores your texts and meta data and provides them to the Retriever at query time."
slug: "/document-stores-api"
---

<a id="document_store"></a>

## Module document\_store

<a id="document_store.BM25DocumentStats"></a>

### BM25DocumentStats

A dataclass for managing document statistics for BM25 retrieval.

**Arguments**:

- `freq_token`: A Counter of token frequencies in the document.
- `doc_len`: Number of tokens in the document.

<a id="document_store.InMemoryDocumentStore"></a>

### InMemoryDocumentStore

Stores data in-memory. It's ephemeral and cannot be saved to disk.

<a id="document_store.InMemoryDocumentStore.__init__"></a>

#### InMemoryDocumentStore.\_\_init\_\_

```python
def __init__(bm25_tokenization_regex: str = r"(?u)\b\w\w+\b",
             bm25_algorithm: Literal["BM25Okapi", "BM25L",
                                     "BM25Plus"] = "BM25L",
             bm25_parameters: Optional[dict] = None,
             embedding_similarity_function: Literal["dot_product",
                                                    "cosine"] = "dot_product",
             index: Optional[str] = None,
             async_executor: Optional[ThreadPoolExecutor] = None,
             return_embedding: bool = True)
```

Initializes the DocumentStore.

**Arguments**:

- `bm25_tokenization_regex`: The regular expression used to tokenize the text for BM25 retrieval.
- `bm25_algorithm`: The BM25 algorithm to use. One of "BM25Okapi", "BM25L", or "BM25Plus".
- `bm25_parameters`: Parameters for BM25 implementation in a dictionary format.
For example: `{'k1':1.5, 'b':0.75, 'epsilon':0.25}`
You can learn more about these parameters by visiting https://github.com/dorianbrown/rank_bm25.
- `embedding_similarity_function`: The similarity function used to compare Documents embeddings.
One of "dot_product" (default) or "cosine". To choose the most appropriate function, look for information
about your embedding model.
- `index`: A specific index to store the documents. If not specified, a random UUID is used.
Using the same index allows you to store documents across multiple InMemoryDocumentStore instances.
- `async_executor`: Optional ThreadPoolExecutor to use for async calls. If not provided, a single-threaded
executor will be initialized and used.
- `return_embedding`: Whether to return the embedding of the retrieved Documents. Default is True.

<a id="document_store.InMemoryDocumentStore.__del__"></a>

#### InMemoryDocumentStore.\_\_del\_\_

```python
def __del__()
```

Cleanup when the instance is being destroyed.

<a id="document_store.InMemoryDocumentStore.shutdown"></a>

#### InMemoryDocumentStore.shutdown

```python
def shutdown()
```

Explicitly shutdown the executor if we own it.

<a id="document_store.InMemoryDocumentStore.storage"></a>

#### InMemoryDocumentStore.storage

```python
@property
def storage() -> dict[str, Document]
```

Utility property that returns the storage used by this instance of InMemoryDocumentStore.

<a id="document_store.InMemoryDocumentStore.to_dict"></a>

#### InMemoryDocumentStore.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="document_store.InMemoryDocumentStore.from_dict"></a>

#### InMemoryDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "InMemoryDocumentStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="document_store.InMemoryDocumentStore.save_to_disk"></a>

#### InMemoryDocumentStore.save\_to\_disk

```python
def save_to_disk(path: str) -> None
```

Write the database and its' data to disk as a JSON file.

**Arguments**:

- `path`: The path to the JSON file.

<a id="document_store.InMemoryDocumentStore.load_from_disk"></a>

#### InMemoryDocumentStore.load\_from\_disk

```python
@classmethod
def load_from_disk(cls, path: str) -> "InMemoryDocumentStore"
```

Load the database and its' data from disk as a JSON file.

**Arguments**:

- `path`: The path to the JSON file.

**Returns**:

The loaded InMemoryDocumentStore.

<a id="document_store.InMemoryDocumentStore.count_documents"></a>

#### InMemoryDocumentStore.count\_documents

```python
def count_documents() -> int
```

Returns the number of how many documents are present in the DocumentStore.

<a id="document_store.InMemoryDocumentStore.filter_documents"></a>

#### InMemoryDocumentStore.filter\_documents

```python
def filter_documents(
        filters: Optional[dict[str, Any]] = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters, refer to the DocumentStore.filter_documents() protocol
documentation.

**Arguments**:

- `filters`: The filters to apply to the document list.

**Returns**:

A list of Documents that match the given filters.

<a id="document_store.InMemoryDocumentStore.write_documents"></a>

#### InMemoryDocumentStore.write\_documents

```python
def write_documents(documents: list[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Refer to the DocumentStore.write_documents() protocol documentation.

If `policy` is set to `DuplicatePolicy.NONE` defaults to `DuplicatePolicy.FAIL`.

<a id="document_store.InMemoryDocumentStore.delete_documents"></a>

#### InMemoryDocumentStore.delete\_documents

```python
def delete_documents(document_ids: list[str]) -> None
```

Deletes all documents with matching document_ids from the DocumentStore.

**Arguments**:

- `document_ids`: The object_ids to delete.

<a id="document_store.InMemoryDocumentStore.bm25_retrieval"></a>

#### InMemoryDocumentStore.bm25\_retrieval

```python
def bm25_retrieval(query: str,
                   filters: Optional[dict[str, Any]] = None,
                   top_k: int = 10,
                   scale_score: bool = False) -> list[Document]
```

Retrieves documents that are most relevant to the query using BM25 algorithm.

**Arguments**:

- `query`: The query string.
- `filters`: A dictionary with filters to narrow down the search space.
- `top_k`: The number of top documents to retrieve. Default is 10.
- `scale_score`: Whether to scale the scores of the retrieved documents. Default is False.

**Returns**:

A list of the top_k documents most relevant to the query.

<a id="document_store.InMemoryDocumentStore.embedding_retrieval"></a>

#### InMemoryDocumentStore.embedding\_retrieval

```python
def embedding_retrieval(
        query_embedding: list[float],
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: Optional[bool] = False) -> list[Document]
```

Retrieves documents that are most similar to the query embedding using a vector similarity metric.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: A dictionary with filters to narrow down the search space.
- `top_k`: The number of top documents to retrieve. Default is 10.
- `scale_score`: Whether to scale the scores of the retrieved Documents. Default is False.
- `return_embedding`: Whether to return the embedding of the retrieved Documents.
If not provided, the value of the `return_embedding` parameter set at component
initialization will be used. Default is False.

**Returns**:

A list of the top_k documents most relevant to the query.

<a id="document_store.InMemoryDocumentStore.count_documents_async"></a>

#### InMemoryDocumentStore.count\_documents\_async

```python
async def count_documents_async() -> int
```

Returns the number of how many documents are present in the DocumentStore.

<a id="document_store.InMemoryDocumentStore.filter_documents_async"></a>

#### InMemoryDocumentStore.filter\_documents\_async

```python
async def filter_documents_async(
        filters: Optional[dict[str, Any]] = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters, refer to the DocumentStore.filter_documents() protocol
documentation.

**Arguments**:

- `filters`: The filters to apply to the document list.

**Returns**:

A list of Documents that match the given filters.

<a id="document_store.InMemoryDocumentStore.write_documents_async"></a>

#### InMemoryDocumentStore.write\_documents\_async

```python
async def write_documents_async(
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Refer to the DocumentStore.write_documents() protocol documentation.

If `policy` is set to `DuplicatePolicy.NONE` defaults to `DuplicatePolicy.FAIL`.

<a id="document_store.InMemoryDocumentStore.delete_documents_async"></a>

#### InMemoryDocumentStore.delete\_documents\_async

```python
async def delete_documents_async(document_ids: list[str]) -> None
```

Deletes all documents with matching document_ids from the DocumentStore.

**Arguments**:

- `document_ids`: The object_ids to delete.

<a id="document_store.InMemoryDocumentStore.bm25_retrieval_async"></a>

#### InMemoryDocumentStore.bm25\_retrieval\_async

```python
async def bm25_retrieval_async(query: str,
                               filters: Optional[dict[str, Any]] = None,
                               top_k: int = 10,
                               scale_score: bool = False) -> list[Document]
```

Retrieves documents that are most relevant to the query using BM25 algorithm.

**Arguments**:

- `query`: The query string.
- `filters`: A dictionary with filters to narrow down the search space.
- `top_k`: The number of top documents to retrieve. Default is 10.
- `scale_score`: Whether to scale the scores of the retrieved documents. Default is False.

**Returns**:

A list of the top_k documents most relevant to the query.

<a id="document_store.InMemoryDocumentStore.embedding_retrieval_async"></a>

#### InMemoryDocumentStore.embedding\_retrieval\_async

```python
async def embedding_retrieval_async(
        query_embedding: list[float],
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: bool = False) -> list[Document]
```

Retrieves documents that are most similar to the query embedding using a vector similarity metric.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: A dictionary with filters to narrow down the search space.
- `top_k`: The number of top documents to retrieve. Default is 10.
- `scale_score`: Whether to scale the scores of the retrieved Documents. Default is False.
- `return_embedding`: Whether to return the embedding of the retrieved Documents. Default is False.

**Returns**:

A list of the top_k documents most relevant to the query.

