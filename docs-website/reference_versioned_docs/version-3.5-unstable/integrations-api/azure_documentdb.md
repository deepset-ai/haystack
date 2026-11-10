---
title: "Azure DocumentDB"
id: integrations-azure-documentdb
description: "Azure DocumentDB integration for Haystack"
slug: "/integrations-azure-documentdb"
---


## haystack_integrations.components.retrievers.azure_documentdb.embedding_retriever

### AzureDocumentDBEmbeddingRetriever

Retrieve documents from Azure DocumentDB using `cosmosSearch` vector similarity.

#### __init__

```python
__init__(
    *,
    document_store: AzureDocumentDBDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Create the embedding retriever.

**Parameters:**

- **document_store** (<code>AzureDocumentDBDocumentStore</code>) – Azure DocumentDB document store to query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Default Haystack metadata filters.
- **top_k** (<code>int</code>) – Maximum number of documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy for combining initialization and runtime filters.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Serialized retriever configuration.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AzureDocumentDBEmbeddingRetriever
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Serialized retriever configuration.

**Returns:**

- <code>AzureDocumentDBEmbeddingRetriever</code> – The deserialized retriever.

#### close

```python
close() -> None
```

Release synchronous document-store resources.

#### close_async

```python
close_async() -> None
```

Release asynchronous document-store resources.

#### run

```python
run(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents by vector similarity.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Query vector.
- **filters** (<code>dict\[str, Any\] | None</code>) – Runtime Haystack metadata filters.
- **top_k** (<code>int | None</code>) – Runtime maximum number of documents.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary containing the retrieved `documents`.

#### run_async

```python
run_async(
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents by vector similarity.

**Parameters:**

- **query_embedding** (<code>list\[float\]</code>) – Query vector.
- **filters** (<code>dict\[str, Any\] | None</code>) – Runtime Haystack metadata filters.
- **top_k** (<code>int | None</code>) – Runtime maximum number of documents.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary containing the retrieved `documents`.

## haystack_integrations.components.retrievers.azure_documentdb.full_text_retriever

### AzureDocumentDBFullTextRetriever

Retrieve documents using Azure DocumentDB BM25 full-text search, currently a gated preview.

#### __init__

```python
__init__(
    *,
    document_store: AzureDocumentDBDocumentStore,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    filter_policy: str | FilterPolicy = FilterPolicy.REPLACE
) -> None
```

Create the full-text retriever.

**Parameters:**

- **document_store** (<code>AzureDocumentDBDocumentStore</code>) – Azure DocumentDB document store to query.
- **filters** (<code>dict\[str, Any\] | None</code>) – Default Haystack metadata filters.
- **top_k** (<code>int</code>) – Maximum number of documents to return.
- **filter_policy** (<code>str | FilterPolicy</code>) – Policy for combining initialization and runtime filters.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Serialized retriever configuration.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AzureDocumentDBFullTextRetriever
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Serialized retriever configuration.

**Returns:**

- <code>AzureDocumentDBFullTextRetriever</code> – The deserialized retriever.

#### close

```python
close() -> None
```

Release synchronous document-store resources.

#### close_async

```python
close_async() -> None
```

Release asynchronous document-store resources.

#### run

```python
run(
    query: str | list[str],
    fuzzy: dict[str, int] | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Retrieve documents by BM25 keyword search.

**Parameters:**

- **query** (<code>str | list\[str\]</code>) – Query string or strings.
- **fuzzy** (<code>dict\[str, int\] | None</code>) – Azure DocumentDB fuzzy-search options such as `maxEdits`.
- **filters** (<code>dict\[str, Any\] | None</code>) – Runtime Haystack metadata filters.
- **top_k** (<code>int | None</code>) – Runtime maximum number of documents.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary containing the retrieved `documents`.

#### run_async

```python
run_async(
    query: str | list[str],
    fuzzy: dict[str, int] | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents by BM25 keyword search.

**Parameters:**

- **query** (<code>str | list\[str\]</code>) – Query string or strings.
- **fuzzy** (<code>dict\[str, int\] | None</code>) – Azure DocumentDB fuzzy-search options such as `maxEdits`.
- **filters** (<code>dict\[str, Any\] | None</code>) – Runtime Haystack metadata filters.
- **top_k** (<code>int | None</code>) – Runtime maximum number of documents.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary containing the retrieved `documents`.

## haystack_integrations.document_stores.azure_documentdb.document_store

### AzureIdentityTokenCallback

Bases: <code>OIDCCallback</code>

Fetch Microsoft Entra access tokens for PyMongo's OIDC authentication.

#### fetch

```python
fetch(context: OIDCCallbackContext) -> OIDCCallbackResult
```

Fetch an access token for Azure DocumentDB.

**Parameters:**

- **context** (<code>OIDCCallbackContext</code>) – PyMongo OIDC callback context.

**Returns:**

- <code>OIDCCallbackResult</code> – The OIDC callback result containing a Microsoft Entra access token.

### AzureDocumentDBDocumentStore

A Haystack document store backed by Azure DocumentDB.

The default authentication mode uses Microsoft Entra ID through `DefaultAzureCredential`. Supply the Azure
DocumentDB cluster name with `cluster_name` or the `AZURE_DOCUMENTDB_CLUSTER_NAME` environment variable.

A connection string can be supplied through `mongo_connection_string` or
`AZURE_DOCUMENTDB_CONNECTION_STRING` for local development and integration tests. Connection strings can contain
credentials and aren't recommended for production workloads.

The collection must already exist. For embedding retrieval, create a `cosmosSearch` vector index by calling
`create_vector_index` or provisioning it separately. Filtered vector search also requires a regular index for
every filtered metadata field, such as `meta.category`. Values used with `>`, `>=`, `<`, or `<=` must be numbers
or ISO-formatted date strings.

Usage:

```python
from haystack_integrations.document_stores.azure_documentdb import AzureDocumentDBDocumentStore

document_store = AzureDocumentDBDocumentStore(database_name="haystack", collection_name="documents")
document_store.create_vector_index(dimensions=1536, similarity="COS")
```

#### __init__

```python
__init__(
    *,
    database_name: str,
    collection_name: str,
    vector_search_index: str = "haystack_vector_index",
    full_text_search_index: str | None = None,
    cluster_name: str | None = None,
    mongo_connection_string: Secret | None = Secret.from_env_var(
        "AZURE_DOCUMENTDB_CONNECTION_STRING", strict=False
    ),
    azure_token_credential: TokenCredential | None = None,
    embedding_field: str = "embedding",
    content_field: str = "content"
) -> None
```

Create an Azure DocumentDB document store.

**Parameters:**

- **database_name** (<code>str</code>) – Name of the existing database.
- **collection_name** (<code>str</code>) – Name of the existing collection.
- **vector_search_index** (<code>str</code>) – Name used when creating the vector index. Azure DocumentDB selects vector indexes
  by path at query time, so this name is not included in vector search queries.
- **full_text_search_index** (<code>str | None</code>) – Name of an Azure DocumentDB full-text search index. Full-text search is currently
  a gated preview and must be enabled on the cluster before using the full-text retriever.
- **cluster_name** (<code>str | None</code>) – Azure DocumentDB cluster name. If omitted, `AZURE_DOCUMENTDB_CLUSTER_NAME` is used.
- **mongo_connection_string** (<code>Secret | None</code>) – Optional MongoDB connection string intended only for local development and
  integration tests. Microsoft Entra authentication is used when this value is absent.
- **azure_token_credential** (<code>TokenCredential | None</code>) – Azure credential used for Microsoft Entra authentication. If omitted,
  `DefaultAzureCredential` is used.
- **embedding_field** (<code>str</code>) – Field containing document embeddings.
- **content_field** (<code>str</code>) – Field containing document content.

**Raises:**

- <code>ValueError</code> – If database, collection, or field names are invalid.

#### connection

```python
connection: MongoClient | AsyncMongoClient
```

Return the active Azure DocumentDB client.

**Returns:**

- <code>MongoClient | AsyncMongoClient</code> – The synchronous or asynchronous PyMongo client.

**Raises:**

- <code>DocumentStoreError</code> – If no connection has been established.

#### collection

```python
collection: Collection | AsyncCollection
```

Return the active Azure DocumentDB collection.

**Returns:**

- <code>Collection | AsyncCollection</code> – The synchronous or asynchronous PyMongo collection.

**Raises:**

- <code>DocumentStoreError</code> – If no collection has been initialized.

#### close

```python
close() -> None
```

Release synchronous client resources.

#### close_async

```python
close_async() -> None
```

Release asynchronous client resources.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this document store to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Serialized document-store configuration.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> AzureDocumentDBDocumentStore
```

Deserialize this document store from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Serialized document-store configuration.

**Returns:**

- <code>AzureDocumentDBDocumentStore</code> – The deserialized document store.

#### count_documents

```python
count_documents() -> int
```

Return the number of documents in the store.

**Returns:**

- <code>int</code> – The number of documents.

#### count_documents_async

```python
count_documents_async() -> int
```

Asynchronously return the number of documents in the store.

**Returns:**

- <code>int</code> – The number of documents.

#### filter_documents

```python
filter_documents(filters: dict[str, Any] | None = None) -> list[Document]
```

Return documents matching Haystack metadata filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Haystack metadata filters. Strings in ordered comparisons must be ISO-formatted dates.

**Returns:**

- <code>list\[Document\]</code> – Documents matching the filters.

#### filter_documents_async

```python
filter_documents_async(filters: dict[str, Any] | None = None) -> list[Document]
```

Asynchronously return documents matching Haystack metadata filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\] | None</code>) – Haystack metadata filters. Strings in ordered comparisons must be ISO-formatted dates.

**Returns:**

- <code>list\[Document\]</code> – Documents matching the filters.

#### write_documents

```python
write_documents(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Write documents to Azure DocumentDB using the requested duplicate policy.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to write.
- **policy** (<code>DuplicatePolicy</code>) – How to handle documents whose IDs already exist.

**Returns:**

- <code>int</code> – The number of documents written.

**Raises:**

- <code>ValueError</code> – If `documents` contains an object that is not a `Document`.
- <code>DuplicateDocumentError</code> – If a duplicate ID is written with `DuplicatePolicy.FAIL`.

#### write_documents_async

```python
write_documents_async(
    documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
) -> int
```

Asynchronously write documents using the requested duplicate policy.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – Documents to write.
- **policy** (<code>DuplicatePolicy</code>) – How to handle documents whose IDs already exist.

**Returns:**

- <code>int</code> – The number of documents written.

**Raises:**

- <code>ValueError</code> – If `documents` contains an object that is not a `Document`.
- <code>DuplicateDocumentError</code> – If a duplicate ID is written with `DuplicatePolicy.FAIL`.

#### delete_documents

```python
delete_documents(document_ids: list[str]) -> None
```

Delete documents with matching Haystack IDs.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – IDs of documents to delete.

#### delete_documents_async

```python
delete_documents_async(document_ids: list[str]) -> None
```

Asynchronously delete documents with matching Haystack IDs.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – IDs of documents to delete.

#### delete_by_filter

```python
delete_by_filter(filters: dict[str, Any]) -> int
```

Delete documents matching filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack metadata filters selecting documents to delete.

**Returns:**

- <code>int</code> – The number of documents deleted.

#### delete_by_filter_async

```python
delete_by_filter_async(filters: dict[str, Any]) -> int
```

Asynchronously delete documents matching filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack metadata filters selecting documents to delete.

**Returns:**

- <code>int</code> – The number of documents deleted.

#### update_by_filter

```python
update_by_filter(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Update metadata on documents matching filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack metadata filters selecting documents to update.
- **meta** (<code>dict\[str, Any\]</code>) – Metadata fields and values to set.

**Returns:**

- <code>int</code> – The number of documents updated.

#### update_by_filter_async

```python
update_by_filter_async(filters: dict[str, Any], meta: dict[str, Any]) -> int
```

Asynchronously update metadata on documents matching filters.

**Parameters:**

- **filters** (<code>dict\[str, Any\]</code>) – Haystack metadata filters selecting documents to update.
- **meta** (<code>dict\[str, Any\]</code>) – Metadata fields and values to set.

**Returns:**

- <code>int</code> – The number of documents updated.

#### delete_all_documents

```python
delete_all_documents(*, recreate_collection: bool = False) -> None
```

Delete all documents, optionally recreating the collection.

**Parameters:**

- **recreate_collection** (<code>bool</code>) – Drop and recreate the collection instead of deleting documents individually.

#### delete_all_documents_async

```python
delete_all_documents_async(*, recreate_collection: bool = False) -> None
```

Asynchronously delete all documents, optionally recreating the collection.

**Parameters:**

- **recreate_collection** (<code>bool</code>) – Drop and recreate the collection instead of deleting documents individually.

#### create_vector_index

```python
create_vector_index(
    *,
    dimensions: int,
    similarity: Literal["COS", "L2", "IP"] = "COS",
    kind: Literal[
        "vector-ivf", "vector-hnsw", "vector-diskann"
    ] = "vector-hnsw",
    **index_options: Any
) -> None
```

Create the configured Azure DocumentDB `cosmosSearch` vector index.

**Parameters:**

- **dimensions** (<code>int</code>) – Number of dimensions in each embedding.
- **similarity** (<code>Literal['COS', 'L2', 'IP']</code>) – Similarity metric: cosine (`COS`), Euclidean (`L2`), or inner product (`IP`).
- **kind** (<code>Literal['vector-ivf', 'vector-hnsw', 'vector-diskann']</code>) – Vector index algorithm.
- **index_options** (<code>Any</code>) – Algorithm-specific Azure DocumentDB index options.

**Raises:**

- <code>ValueError</code> – If `dimensions` is not positive.
- <code>DocumentStoreError</code> – If index creation fails.

#### create_vector_index_async

```python
create_vector_index_async(
    *,
    dimensions: int,
    similarity: Literal["COS", "L2", "IP"] = "COS",
    kind: Literal[
        "vector-ivf", "vector-hnsw", "vector-diskann"
    ] = "vector-hnsw",
    **index_options: Any
) -> None
```

Asynchronously create the configured `cosmosSearch` vector index.

**Parameters:**

- **dimensions** (<code>int</code>) – Number of dimensions in each embedding.
- **similarity** (<code>Literal['COS', 'L2', 'IP']</code>) – Similarity metric: cosine (`COS`), Euclidean (`L2`), or inner product (`IP`).
- **kind** (<code>Literal['vector-ivf', 'vector-hnsw', 'vector-diskann']</code>) – Vector index algorithm.
- **index_options** (<code>Any</code>) – Algorithm-specific Azure DocumentDB index options.

**Raises:**

- <code>ValueError</code> – If `dimensions` is not positive.
- <code>DocumentStoreError</code> – If index creation fails.

## haystack_integrations.document_stores.azure_documentdb.filters
