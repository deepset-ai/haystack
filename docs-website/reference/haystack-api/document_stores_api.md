---
title: "Document Stores"
id: document-stores-api
description: "Stores your texts and meta data and provides them to the Retriever at query time."
slug: "/document-stores-api"
---


## `haystack.document_stores.in_memory.document_store`

### `haystack.document_stores.in_memory.document_store.BM25DocumentStats`

A dataclass for managing document statistics for BM25 retrieval.

**Parameters:**

- **freq_token** (<code>dict\[str, int\]</code>) – A Counter of token frequencies in the document.
- **doc_len** (<code>int</code>) – Number of tokens in the document.

### `haystack.document_stores.in_memory.document_store.InMemoryDocumentStore`

Stores data in-memory. It's ephemeral and cannot be saved to disk.

#### `__init__`

```python
__init__(
    bm25_tokenization_regex: str = "(?u)\\b\\w\\w+\\b",
    bm25_algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25L",
    bm25_parameters: dict | None = None,
    embedding_similarity_function: Literal[
        "dot_product", "cosine"
    ] = "dot_product",
    index: str | None = None,
    async_executor: ThreadPoolExecutor | None = None,
    return_embedding: bool = True,
)
```

Initializes the DocumentStore.

**Parameters:**

- **bm25_tokenization_regex** (<code>str</code>) – The regular expression used to tokenize the text for BM25 retrieval.
- **bm25_algorithm** (<code>Literal['BM25Okapi', 'BM25L', 'BM25Plus']</code>) – The BM25 algorithm to use. One of "BM25Okapi", "BM25L", or "BM25Plus".
- **bm25_parameters** (<code>dict | None</code>) – Parameters for BM25 implementation in a dictionary format.
  For example: `{'k1':1.5, 'b':0.75, 'epsilon':0.25}`
  You can learn more about these parameters by visiting https://github.com/dorianbrown/rank_bm25.
- **embedding_similarity_function** (<code>Literal['dot_product', 'cosine']</code>) – The similarity function used to compare Documents embeddings.
  One of "dot_product" (default) or "cosine". To choose the most appropriate function, look for information
  about your embedding model.
- **index** (<code>str | None</code>) – A specific index to store the documents. If not specified, a random UUID is used.
  Using the same index allows you to store documents across multiple InMemoryDocumentStore instances.
- **async_executor** (<code>ThreadPoolExecutor | None</code>) – Optional ThreadPoolExecutor to use for async calls. If not provided, a single-threaded
  executor will be initialized and used.
- **return_embedding** (<code>bool</code>) – Whether to return the embedding of the retrieved Documents. Default is True.

#### `shutdown`

```python
shutdown()
```

Explicitly shutdown the executor if we own it.

#### `storage`

```python
storage: dict[str, Document]
```

Utility property that returns the storage used by this instance of InMemoryDocumentStore.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> InMemoryDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>InMemoryDocumentStore</code> – The deserialized component.

#### `save_to_disk`

```python
save_to_disk(path: str) -> None
```

Write the database and its' data to disk as a JSON file.

**Parameters:**

- **path** (<code>str</code>) – The path to the JSON file.

#### `load_from_disk`

```python
load_from_disk(path: str) -> InMemoryDocumentStore
```

Load the database and its' data from disk as a JSON file.

**Parameters:**

- **path** (<code>str</code>) – The path to the JSON file.

**Returns:**

- <code>InMemoryDocumentStore</code> – The loaded InMemoryDocumentStore.

#### `count_documents`

```python
count_documents() -> int
```

Returns the number of how many documents are present in the DocumentStore.

#### `filter_documents`

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters, refer to the DocumentStore.filter_documents() protocol
documentation.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – The filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

#### `write_documents`

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Refer to the DocumentStore.write_documents() protocol documentation.

If `policy` is set to `DuplicatePolicy.NONE` defaults to `DuplicatePolicy.FAIL`.

#### `delete_documents`

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes all documents with matching document_ids from the DocumentStore.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – The object_ids to delete.

#### `delete_all_documents`

```python
delete_all_documents() -> None
```

Deletes all documents in the document store.

#### `update_by_filter`

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Updates the metadata of all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see filter_documents.
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update. These will be merged with existing metadata.

**Returns:**

- <code>int</code> – The number of documents updated.

**Raises:**

- <code>ValueError</code> – if filters have invalid syntax.

#### `delete_by_filter`

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes all documents that match the provided filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for deletion.
  For filter syntax, see filter_documents.

**Returns:**

- <code>int</code> – The number of documents deleted.

**Raises:**

- <code>ValueError</code> – if filters have invalid syntax.

#### `bm25_retrieval`

```python
bm25_retrieval(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    scale_score: bool = False,
) -> list[Document]
```

Retrieves documents that are most relevant to the query using BM25 algorithm.

**Parameters:**

- **query** (<code>str</code>) – The query string.
- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the search space.
- **top_k** (<code>int</code>) – The number of top documents to retrieve. Default is 10.
- **scale_score** (<code>bool</code>) – Whether to scale the scores of the retrieved documents. Default is False.

**Returns:**

- <code>list\[Document\]</code> – A list of the top_k documents most relevant to the query.

#### `embedding_retrieval`

```python
embedding_retrieval(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    scale_score: bool = False,
    return_embedding: bool | None = False,
) -> list[Document]
```

Retrieves documents that are most similar to the query embedding using a vector similarity metric.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the search space.
- **top_k** (<code>int</code>) – The number of top documents to retrieve. Default is 10.
- **scale_score** (<code>bool</code>) – Whether to scale the scores of the retrieved Documents. Default is False.
- **return_embedding** (<code>bool | None</code>) – Whether to return the embedding of the retrieved Documents.
  If not provided, the value of the `return_embedding` parameter set at component
  initialization will be used. Default is False.

**Returns:**

- <code>list\[Document\]</code> – A list of the top_k documents most relevant to the query.

**Raises:**

- <code>ValueError</code> – if filters have invalid syntax.

#### `count_documents_async`

```python
count_documents_async() -> int
```

Returns the number of how many documents are present in the DocumentStore.

#### `filter_documents_async`

```python
filter_documents_async(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters, refer to the DocumentStore.filter_documents() protocol
documentation.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – The filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

#### `write_documents_async`

```python
write_documents_async(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Refer to the DocumentStore.write_documents() protocol documentation.

If `policy` is set to `DuplicatePolicy.NONE` defaults to `DuplicatePolicy.FAIL`.

#### `delete_documents_async`

```python
delete_documents_async(document_ids: list[str]) -> None
```

Deletes all documents with matching document_ids from the DocumentStore.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – The object_ids to delete.

#### `bm25_retrieval_async`

```python
bm25_retrieval_async(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    scale_score: bool = False,
) -> list[Document]
```

Retrieves documents that are most relevant to the query using BM25 algorithm.

**Parameters:**

- **query** (<code>str</code>) – The query string.
- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the search space.
- **top_k** (<code>int</code>) – The number of top documents to retrieve. Default is 10.
- **scale_score** (<code>bool</code>) – Whether to scale the scores of the retrieved documents. Default is False.

**Returns:**

- <code>list\[Document\]</code> – A list of the top_k documents most relevant to the query.

#### `embedding_retrieval_async`

```python
embedding_retrieval_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    scale_score: bool = False,
    return_embedding: bool = False,
) -> list[Document]
```

Retrieves documents that are most similar to the query embedding using a vector similarity metric.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – A dictionary with filters to narrow down the search space.
- **top_k** (<code>int</code>) – The number of top documents to retrieve. Default is 10.
- **scale_score** (<code>bool</code>) – Whether to scale the scores of the retrieved Documents. Default is False.
- **return_embedding** (<code>bool</code>) – Whether to return the embedding of the retrieved Documents. Default is False.

**Returns:**

- <code>list\[Document\]</code> – A list of the top_k documents most relevant to the query.
