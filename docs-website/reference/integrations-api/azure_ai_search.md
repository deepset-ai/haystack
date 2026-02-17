---
title: "Azure AI Search"
id: integrations-azure_ai_search
description: "Azure AI Search integration for Haystack"
slug: "/integrations-azure_ai_search"
---


## `haystack_integrations.components.retrievers.azure_ai_search.embedding_retriever`

### `AzureAISearchEmbeddingRetriever`

Retrieves documents from the AzureAISearchDocumentStore using a vector similarity metric.
Must be connected to the AzureAISearchDocumentStore to run.

#### `__init__`

```python
__init__(
    *,
    document_store: AzureAISearchDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    **kwargs: Any
)
```

Create the AzureAISearchEmbeddingRetriever component.

**Parameters:**

- **document_store** (<code>AzureAISearchDocumentStore</code>) – An instance of AzureAISearchDocumentStore to use with the Retriever.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied when fetching documents from the Document Store.
- **top_k** (<code>int</code>) – Maximum number of documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy to determine how filters are applied.
- **kwargs** (<code>Any</code>) – Additional keyword arguments to pass to the Azure AI's search endpoint.
  Some of the supported parameters:
  - `query_type`: A string indicating the type of query to perform. Possible values are
    'simple','full' and 'semantic'.
  - `semantic_configuration_name`: The name of semantic configuration to be used when
    processing semantic queries.
    For more information on parameters, see the
    [official Azure AI Search documentation](https://learn.microsoft.com/en-us/azure/search/).

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> AzureAISearchEmbeddingRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AzureAISearchEmbeddingRetriever</code> – Deserialized component.

#### `run`

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents from the AzureAISearchDocumentStore.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – A list of floats representing the query embedding.
- **filters** (<code>dict\[str, Any\] | None</code>) – Filters applied to the retrieved Documents. The way runtime filters are applied depends on
  the `filter_policy` chosen at retriever initialization. See `__init__` method docstring for more
  details.
- **top_k** (<code>int | None</code>) – The maximum number of documents to retrieve.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – Dictionary with the following keys:
- `documents`: A list of documents retrieved from the AzureAISearchDocumentStore.

## `haystack_integrations.document_stores.azure_ai_search.document_store`

### `AzureAISearchDocumentStore`

#### `__init__`

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var(
        "AZURE_AI_SEARCH_API_KEY", strict=False
    ),
    azure_endpoint: Secret = Secret.from_env_var(
        "AZURE_AI_SEARCH_ENDPOINT", strict=True
    ),
    index_name: str = "default",
    embedding_dimension: int = 768,
    metadata_fields: dict[str, SearchField | type] | None = None,
    vector_search_configuration: VectorSearch | None = None,
    include_search_metadata: bool = False,
    **index_creation_kwargs: Any
)
```

A document store using [Azure AI Search](https://azure.microsoft.com/products/ai-services/ai-search/)
as the backend.

**Parameters:**

- **azure_endpoint** (<code>Secret</code>) – The URL endpoint of an Azure AI Search service.
- **api_key** (<code>Secret</code>) – The API key to use for authentication.
- **index_name** (<code>str</code>) – Name of index in Azure AI Search, if it doesn't exist it will be created.
- **embedding_dimension** (<code>int</code>) – Dimension of the embeddings.
- **metadata_fields** (<code>dict\[str, SearchField | type\] | None</code>) – A dictionary mapping metadata field names to their corresponding field definitions.
  Each field can be defined either as:
- A SearchField object to specify detailed field configuration like type, searchability, and filterability
- A Python type (`str`, `bool`, `int`, `float`, or `datetime`) to create a simple filterable field

These fields are automatically added when creating the search index.
Example:

```python
metadata_fields={
    "Title": SearchField(
        name="Title",
        type="Edm.String",
        searchable=True,
        filterable=True
    ),
    "Pages": int
}
```

- **vector_search_configuration** (<code>VectorSearch | None</code>) – Configuration option related to vector search.
  Default configuration uses the HNSW algorithm with cosine similarity to handle vector searches.
- **include_search_metadata** (<code>bool</code>) – Whether to include Azure AI Search metadata fields
  in the returned documents. When set to True, the `meta` field of the returned
  documents will contain the @search.score, @search.reranker_score, @search.highlights,
  @search.captions, and other fields returned by Azure AI Search.
- **index_creation_kwargs** (<code>Any</code>) – Optional keyword parameters to be passed to `SearchIndex` class
  during index creation. Some of the supported parameters:
  \- `semantic_search`: Defines semantic configuration of the search index. This parameter is needed
  to enable semantic search capabilities in index.
  \- `similarity`: The type of similarity algorithm to be used when scoring and ranking the documents
  matching a search query. The similarity algorithm can only be defined at index creation time and
  cannot be modified on existing indexes.

For more information on parameters, see the [official Azure AI Search documentation](https://learn.microsoft.com/en-us/azure/search/).

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> AzureAISearchDocumentStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>AzureAISearchDocumentStore</code> – Deserialized component.

#### `count_documents`

```python
count_documents() -> int
```

Returns how many documents are present in the search index.

**Returns:**

- <code>int</code> – list of retrieved documents.

#### `write_documents`

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Writes the provided documents to search index.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – documents to write to the index.
- **policy** (<code>DuplicatePolicy</code>) – Policy to determine how duplicates are handled.

**Returns:**

- <code>int</code> – the number of documents added to index.

**Raises:**

- <code>ValueError</code> – If the documents are not of type Document.
- <code>TypeError</code> – If the document ids are not strings.

#### `delete_documents`

```python
delete_documents(document_ids: list[str]) -> None
```

Deletes all documents with a matching document_ids from the search index.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – ids of the documents to be deleted.

#### `delete_all_documents`

```python
delete_all_documents(recreate_index: bool = False) -> None
```

Deletes all documents in the document store.

**Parameters:**

- **recreate_index** (<code>bool</code>) – If True, the index will be deleted and recreated with the original schema.
  If False, all documents will be deleted while preserving the index.

#### `delete_by_filter`

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Deletes all documents that match the provided filters.

Azure AI Search does not support server-side delete by query, so this method
first searches for matching documents, then deletes them in a batch operation.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for deletion.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)

**Returns:**

- <code>int</code> – The number of documents deleted.

#### `update_by_filter`

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Updates the fields of all documents that match the provided filters.

Azure AI Search does not support server-side update by query, so this method
first searches for matching documents, then updates them using merge operations.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – The filters to apply to select documents for updating.
  For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- **meta** (<code>dict\[str, Any\]</code>) – The fields to update. These fields must exist in the index schema.

**Returns:**

- <code>int</code> – The number of documents updated.

#### `search_documents`

```python
search_documents(search_text: str = '*', top_k: int = 10) -> list[Document]
```

Returns all documents that match the provided search_text.
If search_text is None, returns all documents.

**Parameters:**

- **search_text** (<code>str</code>) – the text to search for in the Document list.
- **top_k** (<code>int</code>) – Maximum number of documents to return.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given search_text.

#### `filter_documents`

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Returns the documents that match the provided filters.
Filters should be given as a dictionary supporting filtering by metadata. For details on
filters, see the [metadata filtering documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – the filters to apply to the document list.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents that match the given filters.

## `haystack_integrations.document_stores.azure_ai_search.filters`
