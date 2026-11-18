---
title: "Amazon S3 Vectors"
id: integrations-amazon-s3-vectors
description: "Amazon S3 Vectors integration for Haystack"
slug: "/integrations-amazon-s3-vectors"
---


## haystack_integrations.components.retrievers.amazon_s3_vectors.embedding_retriever

### AmazonS3VectorsEmbeddingRetriever

Retrieve documents from an `AmazonS3VectorsDocumentStore` based on their dense embeddings.

Usage example:

```python
from haystack import Document, Pipeline
from haystack.document_stores.types import DuplicatePolicy
# Requires: pip install sentence-transformers-haystack
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack_integrations.components.retrievers.amazon_s3_vectors import AmazonS3VectorsEmbeddingRetriever
from haystack_integrations.document_stores.amazon_s3_vectors import AmazonS3VectorsDocumentStore

document_store = AmazonS3VectorsDocumentStore(
    vector_bucket_name="my-vectors",
    index_name="my-index",
    dimension=768,
)

documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(content="Elephants have been observed to behave in a way that indicates..."),
    Document(content="In certain places, you can witness the phenomenon of bioluminescent waves."),
]

document_embedder = SentenceTransformersDocumentEmbedder()
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)

document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
query_pipeline.add_component("retriever", AmazonS3VectorsEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "How many languages are there?"
res = query_pipeline.run({"text_embedder": {"text": query}})
```

#### __init__

```python
__init__(
    *,
    document_store: AmazonS3VectorsDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Initialize the AmazonS3VectorsEmbeddingRetriever.

**Parameters:**

- **document_store** (<code>AmazonS3VectorsDocumentStore</code>) – An instance of `AmazonS3VectorsDocumentStore`.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents.
- **top_k** (<code>int</code>) – Maximum number of Documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an `AmazonS3VectorsDocumentStore`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AmazonS3VectorsEmbeddingRetriever
```

Deserialize the component from a dictionary.

#### run

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents from the AmazonS3VectorsDocumentStore based on dense embeddings.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. Filters are applied server-side during
  the vector search.
- **top_k** (<code>int | None</code>) – Maximum number of Documents to return. S3 Vectors caps this at 10,000.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with key `"documents"` containing the retrieved Documents.
  Returned documents will not contain embeddings.

#### close

```python
close() -> None
```

Release the resources of the underlying Document Store.

## haystack_integrations.document_stores.amazon_s3_vectors.document_store

### AmazonS3VectorsDocumentStore

A Document Store using [Amazon S3 Vectors](https://aws.amazon.com/s3/features/vectors/).

Amazon S3 Vectors provides serverless vector storage and similarity search within Amazon S3.
This document store stores Haystack `Document` objects as vectors with associated metadata
in an S3 vector bucket and index.

Service limits:

- Maximum `top_k`: 10,000 results per query. Search is approximate, so a `top_k` close to the number of
  stored or matching documents can return fewer results
- Maximum vector dimension: 4,096
- Metadata per vector: 40 KB total, 2 KB filterable
- A metadata value must be a string, a boolean, a finite number, or a non-empty array of either
  all strings or all non-boolean numbers. Anything else (`None`, nested objects, empty or mixed
  arrays) is dropped with a warning rather than failing the write
- Every record needs a vector (`float32` only), so documents written without an embedding are
  stored with a placeholder vector (stripped again on read) and are only meaningfully
  retrievable via `filter_documents()`
- Only `DuplicatePolicy.OVERWRITE` is supported, since `put_vectors` is an upsert
- Distance metrics: `cosine` or `euclidean` (set at index creation, immutable)
- `filter_documents()` filters locally, so prefer `AmazonS3VectorsEmbeddingRetriever` with filters
- `Document.blob` is not stored

Usage example:

```python
from haystack_integrations.document_stores.amazon_s3_vectors import AmazonS3VectorsDocumentStore

document_store = AmazonS3VectorsDocumentStore(
    vector_bucket_name="my-vectors",
    index_name="my-index",
    dimension=768,
)
```

#### __init__

```python
__init__(
    *,
    vector_bucket_name: str,
    index_name: str,
    dimension: int,
    distance_metric: Literal["cosine", "euclidean"] = "cosine",
    region_name: str | None = None,
    aws_access_key_id: Secret | None = None,
    aws_secret_access_key: Secret | None = None,
    aws_session_token: Secret | None = None,
    create_bucket_and_index: bool = True,
    non_filterable_metadata_keys: list[str] | None = None
) -> None
```

Create an AmazonS3VectorsDocumentStore instance.

**Parameters:**

- **vector_bucket_name** (<code>str</code>) – Name of the S3 vector bucket.
- **index_name** (<code>str</code>) – Name of the vector index within the bucket.
- **dimension** (<code>int</code>) – Dimensionality of the embeddings when creating an index (e.g. 768, 1536).
  When connecting to an existing index, its actual dimension is used.
- **distance_metric** (<code>Literal['cosine', 'euclidean']</code>) – Distance metric for similarity search: `"cosine"` or `"euclidean"`.
- **region_name** (<code>str | None</code>) – AWS region. If not provided, uses the default from the environment/config.
- **aws_access_key_id** (<code>Secret | None</code>) – AWS access key ID. If not provided, uses the default credential chain.
- **aws_secret_access_key** (<code>Secret | None</code>) – AWS secret access key.
- **aws_session_token** (<code>Secret | None</code>) – AWS session token for temporary credentials.
- **create_bucket_and_index** (<code>bool</code>) – Whether to automatically create the vector bucket and index
  if they do not exist. Defaults to `True`.
- **non_filterable_metadata_keys** (<code>list\[str\] | None</code>) – Additional metadata keys to mark as non-filterable
  on the index (beyond the internal key used for Document content).

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this document store to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AmazonS3VectorsDocumentStore
```

Deserialize a document store from a dictionary.

#### count_documents

```python
count_documents() -> int
```

Return the number of documents in the document store.

Note: S3 Vectors does not provide a dedicated count API. This method lists all vector keys
via pagination, which can be slow for large indexes.

#### write_documents

```python
write_documents(
    documents: list[Document],
    policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
) -> int
```

Write Documents to the S3 Vectors index.

`AmazonS3VectorsDocumentStore` only supports `DuplicatePolicy.OVERWRITE`: `put_vectors` is an
upsert, and S3 Vectors offers no cheap existence check (only `get_vectors` on batches of
100 keys, or a full index scan), so enforcing any other policy would cost an extra read
pass on every write. Other policies are ignored with a warning.

Documents without an embedding are stored with a placeholder vector, since S3 Vectors
requires a vector on every record. `filter_documents()` strips the placeholder, so they read
back with `embedding=None`, but they are meaningless as similarity-search results.

Metadata per vector is limited to 40 KB total (2 KB filterable).

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to write.
- **policy** (<code>DuplicatePolicy</code>) – The duplicate policy. Only `DuplicatePolicy.OVERWRITE` is supported.

**Returns:**

- <code>int</code> – The number of documents written.

**Raises:**

- <code>ValueError</code> – If `documents` is not a list of `Document` instances.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Return documents matching the provided filters.

Note: S3 Vectors only supports metadata filtering during vector similarity queries, not as a
standalone operation. This method lists all vectors and filters them locally with Haystack,
which can be slow for large indexes. `AmazonS3VectorsEmbeddingRetriever` filters through S3
Vectors instead, and the two can match different documents for the same filter, especially
on array metadata: S3 Vectors matches an array field when any element matches, while
Haystack compares the whole array.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Haystack-format filters to apply.

**Returns:**

- <code>list\[Document\]</code> – A list of matching Documents.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Delete documents by their IDs.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – List of document IDs to delete.

#### delete_all_documents

```python
delete_all_documents() -> None
```

Delete all documents from the index.

Note: S3 Vectors has no truncate operation, so every key is listed and then deleted in
batches, which can be slow for large indexes.

#### close

```python
close() -> None
```

Close the underlying boto3 client and release its connection pool.

The store can be used again afterwards: the next call lazily creates a new client.
