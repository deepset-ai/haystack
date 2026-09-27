---
title: "Amazon DynamoDB"
id: integrations-dynamodb
description: "Amazon DynamoDB integration for Haystack"
slug: "/integrations-dynamodb"
---


## haystack_integrations.components.retrievers.dynamodb.embedding_retriever

### DynamoDBEmbeddingRetriever

Retrieves documents from a `DynamoDBDocumentStore` using vector similarity on embeddings.

Uses DynamoDB's native `SearchVectors` API (cosine similarity). DynamoDB returns at most 100
candidates per search, so `top_k` cannot exceed 100. Metadata filters are applied client-side
to those candidates, so a selective filter can return fewer than `top_k` documents even when
more matching documents exist.

Example usage:

```python
from haystack_integrations.document_stores.dynamodb import DynamoDBDocumentStore
from haystack_integrations.components.retrievers.dynamodb import DynamoDBEmbeddingRetriever

store = DynamoDBDocumentStore(table_name="docs", index_name="doc-index", embedding_dimension=768)
retriever = DynamoDBEmbeddingRetriever(document_store=store, top_k=5)
result = retriever.run(query_embedding=[0.1, 0.2, ...])
```

#### __init__

```python
__init__(
    *,
    document_store: DynamoDBDocumentStore,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Creates a new DynamoDBEmbeddingRetriever.

**Parameters:**

- **document_store** (<code>DynamoDBDocumentStore</code>) – The `DynamoDBDocumentStore` to retrieve documents from.
- **top_k** (<code>int</code>) – Maximum number of documents to return, between 1 and 100 (the DynamoDB
  `SearchVectors` limit).
- **filters** (<code>dict\[str, Any\] | None</code>) – Optional Haystack metadata filters applied at retrieval time. Applied
  client-side after the native vector search, since DynamoDB's `SearchVectors`
  filter expressions can only reference attributes declared in the index's
  `SearchSchema` at index-creation time.
- **filter_policy** (<code>str | FilterPolicy</code>) – How run-time filters combine with `filters`: `REPLACE` (default)
  uses the run-time filters alone when they are given, `MERGE` combines both.

**Raises:**

- <code>ValueError</code> – If `document_store` is not a `DynamoDBDocumentStore` or `top_k` is
  outside the allowed range.

#### run

```python
run(
    query_embedding: list[float],
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, list[Document]]
```

Retrieves documents most similar to `query_embedding`.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – The query vector.
- **top_k** (<code>int | None</code>) – Overrides the instance-level `top_k` for this call; must stay between 1 and 100.
- **filters** (<code>dict\[str, Any\] | None</code>) – Run-time filters, combined with the instance-level `filters` according to
  `filter_policy`.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with `documents`, a list of `Document` objects sorted by score.

#### run_async

```python
run_async(
    query_embedding: list[float],
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, list[Document]]
```

Asynchronously retrieves documents most similar to `query_embedding`.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – The query vector.
- **top_k** (<code>int | None</code>) – Overrides the instance-level `top_k` for this call; must stay between 1 and 100.
- **filters** (<code>dict\[str, Any\] | None</code>) – Run-time filters, combined with the instance-level `filters` according to
  `filter_policy`.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with `documents`, a list of `Document` objects sorted by score.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> DynamoDBEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>DynamoDBEmbeddingRetriever</code> – Deserialized component.

## haystack_integrations.document_stores.dynamodb.document_store

### DynamoDBDocumentStore

A Haystack DocumentStore backed by Amazon DynamoDB native vector search.

Uses the `SearchVectors` API (GA 2026-08-05). Documents are stored as items in a
DynamoDB table with a vector index, and retrieved via cosine similarity search. Every
method has an `_async` counterpart built on `aiobotocore`.

Limitations to weigh before choosing this store:

- `filter_documents`, `count_documents` and the filter-based bulk operations run a
  consistent full-table `Scan` and evaluate Haystack filters client-side, so their cost
  grows with the table size. `SearchVectors` can only filter on attributes fixed in the
  index `SearchSchema` at creation time, which arbitrary Haystack filters cannot use.
- `SearchVectors` returns at most 100 candidates per request
  (`SEARCH_VECTORS_MAX_TOP_K`), so `top_k` cannot exceed 100 and filtered retrieval can
  only choose among those candidates.
- A DynamoDB item is limited to 400 KB, which bounds a document's content, metadata and
  embedding together.

Example usage:

```python
from haystack_integrations.document_stores.dynamodb import DynamoDBDocumentStore

store = DynamoDBDocumentStore(
    table_name="haystack-documents",
    index_name="haystack-vector-index",
    embedding_dimension=768,
    region_name="us-east-1",
)
```

#### __init__

```python
__init__(
    *,
    table_name: str = "haystack_documents",
    index_name: str = "haystack_vector_index",
    embedding_dimension: int = 768,
    region_name: str | None = None,
    aws_access_key_id: Secret = Secret.from_env_var(
        "AWS_ACCESS_KEY_ID", strict=False
    ),
    aws_secret_access_key: Secret = Secret.from_env_var(
        "AWS_SECRET_ACCESS_KEY", strict=False
    ),
    aws_session_token: Secret = Secret.from_env_var(
        "AWS_SESSION_TOKEN", strict=False
    ),
    create_table_if_not_exists: bool = True,
    similarity_function: str = "cosine"
) -> None
```

Creates a new DynamoDBDocumentStore instance.

**Parameters:**

- **table_name** (<code>str</code>) – Name of the DynamoDB table to store documents in. Created if it
  does not exist and `create_table_if_not_exists` is `True`.
- **index_name** (<code>str</code>) – Name of the vector index on the table.
- **embedding_dimension** (<code>int</code>) – Dimensionality of document embeddings.
- **region_name** (<code>str | None</code>) – AWS region. Defaults to the boto3 session's configured region.
- **aws_access_key_id** (<code>Secret</code>) – AWS access key as a `Secret`. Defaults to `AWS_ACCESS_KEY_ID`
  env var, falling back to the default boto3 credential chain if not set.
- **aws_secret_access_key** (<code>Secret</code>) – AWS secret key as a `Secret`. Defaults to
  `AWS_SECRET_ACCESS_KEY` env var.
- **aws_session_token** (<code>Secret</code>) – AWS session token as a `Secret`, for temporary credentials.
  Defaults to `AWS_SESSION_TOKEN` env var.
- **create_table_if_not_exists** (<code>bool</code>) – If `True`, create the table and vector index on
  first use if they don't already exist.
- **similarity_function** (<code>str</code>) – Vector similarity function. This integration currently supports
  only `"cosine"`. DynamoDB itself also offers `DOT_PRODUCT` and `EUCLIDEAN` indexes, but
  their score conversion is not implemented yet.

**Raises:**

- <code>ValueError</code> – If `similarity_function` is not `"cosine"`.

#### count_documents

```python
count_documents() -> int
```

Returns the number of documents in the store.

Counts with a consistent `Scan`, so the cost grows with the table size.

**Returns:**

- <code>int</code> – Exact document count.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns documents matching the provided filters.

DynamoDB's `SearchVectors`/`Query` filter expressions can only reference attributes
declared in the index's `SearchSchema` at index-creation time. Since Haystack's metadata
filters are arbitrary and not known at index-creation time, filtering here is applied
client-side after a consistent full-table scan, so the cost grows with the table size.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Haystack metadata filters. If `None`, all documents are returned.

**Returns:**

- <code>list\[Document\]</code> – List of matching `Document` objects.

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Writes documents to the store.

Documents are written one by one. With `FAIL`, documents preceding the first duplicate
stay written.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to write.
- **policy** (<code>DuplicatePolicy</code>) – How to handle duplicates: `OVERWRITE`, `SKIP`, or `FAIL`. `NONE` (the
  default) behaves like `FAIL`.

**Returns:**

- <code>int</code> – Number of documents written.

**Raises:**

- <code>ValueError</code> – If `documents` contains non-`Document` objects.
- <code>DuplicateDocumentError</code> – If a duplicate is found and policy is `FAIL`.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes documents by their IDs.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – List of document IDs to delete.

#### delete_all_documents

```python
delete_all_documents() -> None
```

Deletes all documents in the store.

Items are deleted one by one after a consistent scan; the table and its vector index are kept.

#### delete_by_filter

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes all documents matching the filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack metadata filters selecting the documents to delete. Must not be
  empty; use `delete_all_documents` to clear the store.

**Returns:**

- <code>int</code> – The number of documents deleted.

**Raises:**

- <code>ValueError</code> – If `filters` is empty.

#### update_by_filter

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Merges `meta` into the metadata of all documents matching the filters.

Existing metadata keys not present in `meta` are kept; matching keys are overwritten.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack metadata filters selecting the documents to update. Must not be empty.
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to set on each matching document.

**Returns:**

- <code>int</code> – The number of documents updated.

**Raises:**

- <code>ValueError</code> – If `filters` is empty.

#### count_documents_async

```python
count_documents_async() -> int
```

Asynchronously returns the number of documents in the store.

**Returns:**

- <code>int</code> – Exact document count.

#### filter_documents_async

```python
filter_documents_async(filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously returns documents matching the provided filters.

See `filter_documents` for how filters are evaluated.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Haystack metadata filters. If `None`, all documents are returned.

**Returns:**

- <code>list\[Document\]</code> – List of matching `Document` objects.

#### write_documents_async

```python
write_documents_async(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Asynchronously writes documents to the store.

See `write_documents` for the duplicate handling semantics.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to write.
- **policy** (<code>DuplicatePolicy</code>) – How to handle duplicates: `OVERWRITE`, `SKIP`, or `FAIL`. `NONE` (the
  default) behaves like `FAIL`.

**Returns:**

- <code>int</code> – Number of documents written.

**Raises:**

- <code>ValueError</code> – If `documents` contains non-`Document` objects.
- <code>DuplicateDocumentError</code> – If a duplicate is found and policy is `FAIL`.

#### delete_documents_async

```python
delete_documents_async(document_ids: list[str]) -> None
```

Asynchronously deletes documents by their IDs.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – List of document IDs to delete.

#### delete_all_documents_async

```python
delete_all_documents_async() -> None
```

Asynchronously deletes all documents in the store.

Items are deleted one by one after a consistent scan; the table and its vector index are kept.

#### delete_by_filter_async

```python
delete_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously deletes all documents matching the filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack metadata filters selecting the documents to delete. Must not be
  empty; use `delete_all_documents_async` to clear the store.

**Returns:**

- <code>int</code> – The number of documents deleted.

**Raises:**

- <code>ValueError</code> – If `filters` is empty.

#### update_by_filter_async

```python
update_by_filter_async(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Asynchronously merges `meta` into the metadata of all documents matching the filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack metadata filters selecting the documents to update. Must not be empty.
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to set on each matching document.

**Returns:**

- <code>int</code> – The number of documents updated.

**Raises:**

- <code>ValueError</code> – If `filters` is empty.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> DynamoDBDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>DynamoDBDocumentStore</code> – Deserialized component.
