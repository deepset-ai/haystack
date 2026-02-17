---
title: "Pinecone"
id: integrations-pinecone
description: "Pinecone integration for Haystack"
slug: "/integrations-pinecone"
---


## `haystack_integrations.components.retrievers.pinecone.embedding_retriever`

### `PineconeEmbeddingRetriever`

Retrieves documents from the `PineconeDocumentStore`, based on their dense embeddings.

Usage example:

```python
import os
from haystack.document_stores.types import DuplicatePolicy
from haystack import Document
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_API_KEY"
document_store = PineconeDocumentStore(index="my_index", namespace="my_namespace", dimension=768)

documents = [Document(content="There are over 7,000 languages spoken around the world today."),
             Document(content="Elephants have been observed to behave in a way that indicates..."),
             Document(content="In certain places, you can witness the phenomenon of bioluminescent waves.")]

document_embedder = SentenceTransformersDocumentEmbedder()
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)

document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
query_pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "How many languages are there?"

res = query_pipeline.run({"text_embedder": {"text": query}})
assert res['retriever']['documents'][0].content == "There are over 7,000 languages spoken around the world today."
```

#### `__init__`

```python
__init__(
    *,
    document_store: PineconeDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
)
```

**Parameters:**

- **document_store** (<code>PineconeDocumentStore</code>) – The Pinecone Document Store.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents.
- **top_k** (<code>int</code>) – Maximum number of Documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.

**Raises:**

- <code>ValueError</code> – If `document_store` is not an instance of `PineconeDocumentStore`.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> PineconeEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>PineconeEmbeddingRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents from the `PineconeDocumentStore`, based on their dense embeddings.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – Maximum number of `Document`s to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – List of Document similar to `query_embedding`.

#### `run_async`

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents from the `PineconeDocumentStore`, based on their dense embeddings.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Embedding of the query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See init method docstring for more
  details.
- **top_k** (<code>int | None</code>) – Maximum number of `Document`s to return.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – List of Document similar to `query_embedding`.

## `haystack_integrations.document_stores.pinecone.document_store`

### `PineconeDocumentStore`

A Document Store using [Pinecone vector database](https://www.pinecone.io/).

#### `__init__`

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("PINECONE_API_KEY"),
    index: str = "default",
    namespace: str = "default",
    batch_size: int = 100,
    dimension: int = 768,
    spec: dict[str, Any] | None = None,
    metric: Literal["cosine", "euclidean", "dotproduct"] = "cosine",
    show_progress: bool = True
)
```

Creates a new PineconeDocumentStore instance.
It is meant to be connected to a Pinecone index and namespace.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Pinecone API key.
- **index** (<code>str</code>) – The Pinecone index to connect to. If the index does not exist, it will be created.
- **namespace** (<code>str</code>) – The Pinecone namespace to connect to. If the namespace does not exist, it will be created
  at the first write.
- **batch_size** (<code>int</code>) – The number of documents to write in a single batch. When setting this parameter,
  consider [documented Pinecone limits](https://docs.pinecone.io/reference/quotas-and-limits).
- **dimension** (<code>int</code>) – The dimension of the embeddings. This parameter is only used when creating a new index.
- **spec** (<code>dict\[str, Any\] | None</code>) – The Pinecone spec to use when creating a new index. Allows choosing between serverless and pod
  deployment options and setting additional parameters. Refer to the
  [Pinecone documentation](https://docs.pinecone.io/reference/api/control-plane/create_index) for more
  details.
  If not provided, a default spec with serverless deployment in the `us-east-1` region will be used
  (compatible with the free tier).
- **metric** (<code>Literal['cosine', 'euclidean', 'dotproduct']</code>) – The metric to use for similarity search. This parameter is only used when creating a new index.
- **show_progress** (<code>bool</code>) – Whether to show a progress bar when upserting documents. Set to False to disable
  (e.g. in tests or scripts where quiet output is preferred).

#### `close`

```python
close()
```

Close the associated synchronous resources.

#### `close_async`

```python
close_async()
```

Close the associated asynchronous resources. To be invoked manually when the Document Store is no longer needed.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> PineconeDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>PineconeDocumentStore</code> – Deserialized component.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `count_documents`

```python
count_documents() -> int
```

Returns how many documents are present in the document store.

#### `count_documents_async`

```python
count_documents_async() -> int
```

Asynchronously returns how many documents are present in the document store.

#### `write_documents`

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Writes Documents to Pinecone.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to write to the document store.
- **policy** (<code>DuplicatePolicy</code>) – The duplicate policy to use when writing documents.
  PineconeDocumentStore only supports `DuplicatePolicy.OVERWRITE`.

**Returns:**

- <code>int</code> – The number of documents written to the document store.

#### `write_documents_async`

```python
write_documents_async(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Asynchronously writes Documents to Pinecone.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of Documents to write to the document store.
- **policy** (<code>DuplicatePolicy</code>) – The duplicate policy to use when writing documents.
  PineconeDocumentStore only supports `DuplicatePolicy.OVERWRITE`.

**Returns:**

- <code>int</code> – The number of documents written to the document store.

#### `filter_documents`

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – The filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

#### `filter_documents_async`

```python
filter_documents_async(filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously returns the documents that match the filters provided.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – The filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

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

#### `delete_all_documents`

```python
delete_all_documents() -> None
```

Deletes all documents in the document store.

#### `delete_all_documents_async`

```python
delete_all_documents_async() -> None
```

Asynchronously deletes all documents in the document store.

#### `delete_by_filter`

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes all documents that match the provided filters.

Pinecone does not support server-side delete by filter, so this method
first searches for matching documents, then deletes them by ID.

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

Pinecone does not support server-side delete by filter, so this method
first searches for matching documents, then deletes them by ID.

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

Pinecone does not support server-side update by filter, so this method
first searches for matching documents, then updates their metadata and re-writes them.

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

Pinecone does not support server-side update by filter, so this method
first searches for matching documents, then updates their metadata and re-writes them.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The metadata fields to update. This will be merged with existing metadata.

**Returns:**

- <code>int</code> – The number of documents updated.

#### `count_documents_by_filter`

```python
count_documents_by_filter(filters: dict[str, Any]) -> int
```

Returns the count of documents that match the provided filters.

Note: Due to Pinecone's limitations, this method fetches documents and counts them.
For large result sets, this is subject to Pinecone's TOP_K_LIMIT of 1000 documents.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to the document list.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns:**

- <code>int</code> – The number of documents that match the filters.

#### `count_documents_by_filter_async`

```python
count_documents_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously returns the count of documents that match the provided filters.

Note: Due to Pinecone's limitations, this method fetches documents and counts them.
For large result sets, this is subject to Pinecone's TOP_K_LIMIT of 1000 documents.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to the document list.

**Returns:**

- <code>int</code> – The number of documents that match the filters.

#### `count_unique_metadata_by_filter`

```python
count_unique_metadata_by_filter(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Counts unique values for each specified metadata field in documents matching the filters.

Note: Due to Pinecone's limitations, this method fetches documents and aggregates in Python.
Subject to Pinecone's TOP_K_LIMIT of 1000 documents.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents.
- **metadata_fields** (<code>list\[str\]</code>) – List of metadata field names to count unique values for.

**Returns:**

- <code>dict\[str, int\]</code> – Dictionary mapping field names to counts of unique values.

#### `count_unique_metadata_by_filter_async`

```python
count_unique_metadata_by_filter_async(
    filters: dict[str, Any], metadata_fields: list[str]
) -> dict[str, int]
```

Asynchronously counts unique values for each specified metadata field in documents matching the filters.

Note: Due to Pinecone's limitations, this method fetches documents and aggregates in Python.
Subject to Pinecone's TOP_K_LIMIT of 1000 documents.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents.
- **metadata_fields** (<code>list\[str\]</code>) – List of metadata field names to count unique values for.

**Returns:**

- <code>dict\[str, int\]</code> – Dictionary mapping field names to counts of unique values.

#### `get_metadata_fields_info`

```python
get_metadata_fields_info() -> dict[str, dict[str, str]]
```

Returns information about metadata fields and their types by sampling documents.

Note: Pinecone doesn't provide a schema introspection API, so this method infers field types
by examining the metadata of documents stored in the index (up to 1000 documents).

Type mappings:

- 'text': Document content field
- 'keyword': String metadata values
- 'long': Numeric metadata values (int or float)
- 'boolean': Boolean metadata values

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – Dictionary mapping field names to type information.
  Example:

```python
{
    'content': {'type': 'text'},
    'category': {'type': 'keyword'},
    'priority': {'type': 'long'},
}
```

#### `get_metadata_fields_info_async`

```python
get_metadata_fields_info_async() -> dict[str, dict[str, str]]
```

Asynchronously returns information about metadata fields and their types by sampling documents.

Note: Pinecone doesn't provide a schema introspection API, so this method infers field types
by examining the metadata of documents stored in the index (up to 1000 documents).

Type mappings:

- 'text': Document content field
- 'keyword': String metadata values
- 'long': Numeric metadata values (int or float)
- 'boolean': Boolean metadata values

**Returns:**

- <code>dict\[str, dict\[str, str\]\]</code> – Dictionary mapping field names to type information.
  Example:

```python
{
    'content': {'type': 'text'},
    'category': {'type': 'keyword'},
    'priority': {'type': 'long'},
}
```

#### `get_metadata_field_min_max`

```python
get_metadata_field_min_max(metadata_field: str) -> dict[str, Any]
```

Returns the minimum and maximum values for a metadata field.

Supports numeric (int, float), boolean, and string (keyword) types:

- Numeric: Returns min/max based on numeric value
- Boolean: Returns False as min, True as max
- String: Returns min/max based on alphabetical ordering

Note: This method fetches all documents and computes min/max in Python.
Subject to Pinecone's TOP_K_LIMIT of 1000 documents.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field name to analyze.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with 'min' and 'max' keys.

**Raises:**

- <code>ValueError</code> – If the field doesn't exist or has no values.

#### `get_metadata_field_min_max_async`

```python
get_metadata_field_min_max_async(metadata_field: str) -> dict[str, Any]
```

Asynchronously returns the minimum and maximum values for a metadata field.

Supports numeric (int, float), boolean, and string (keyword) types:

- Numeric: Returns min/max based on numeric value
- Boolean: Returns False as min, True as max
- String: Returns min/max based on alphabetical ordering

Note: This method fetches all documents and computes min/max in Python.
Subject to Pinecone's TOP_K_LIMIT of 1000 documents.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field name to analyze.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with 'min' and 'max' keys.

**Raises:**

- <code>ValueError</code> – If the field doesn't exist or has no values.

#### `get_metadata_field_unique_values`

```python
get_metadata_field_unique_values(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10,
) -> tuple[list[str], int]
```

Retrieves unique values for a metadata field with optional search and pagination.

Note: This method fetches documents and extracts unique values in Python.
Subject to Pinecone's TOP_K_LIMIT of 1000 documents.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field name to get unique values for.
- **search_term** (<code>str | None</code>) – Optional search term to filter values (case-insensitive substring match).
- **from\_** (<code>int</code>) – Starting offset for pagination (default: 0).
- **size** (<code>int</code>) – Number of values to return (default: 10).

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – Tuple of (list of unique values, total count of matching values).

#### `get_metadata_field_unique_values_async`

```python
get_metadata_field_unique_values_async(
    metadata_field: str,
    search_term: str | None = None,
    from_: int = 0,
    size: int = 10,
) -> tuple[list[str], int]
```

Asynchronously retrieves unique values for a metadata field with optional search and pagination.

Note: This method fetches documents and extracts unique values in Python.
Subject to Pinecone's TOP_K_LIMIT of 1000 documents.

**Parameters:**

- **metadata_field** (<code>str</code>) – The metadata field name to get unique values for.
- **search_term** (<code>str | None</code>) – Optional search term to filter values (case-insensitive substring match).
- **from\_** (<code>int</code>) – Starting offset for pagination (default: 0).
- **size** (<code>int</code>) – Number of values to return (default: 10).

**Returns:**

- <code>tuple\[list\[str\], int\]</code> – Tuple of (list of unique values, total count of matching values).
