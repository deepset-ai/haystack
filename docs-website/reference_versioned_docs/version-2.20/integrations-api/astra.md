---
title: "Astra"
id: integrations-astra
description: "Astra integration for Haystack"
slug: "/integrations-astra"
---

<a id="haystack_integrations.components.retrievers.astra.retriever"></a>

## Module haystack\_integrations.components.retrievers.astra.retriever

<a id="haystack_integrations.components.retrievers.astra.retriever.AstraEmbeddingRetriever"></a>

### AstraEmbeddingRetriever

A component for retrieving documents from an AstraDocumentStore.

Usage example:
```python
from haystack_integrations.document_stores.astra import AstraDocumentStore
from haystack_integrations.components.retrievers.astra import AstraEmbeddingRetriever

document_store = AstraDocumentStore(
    api_endpoint=api_endpoint,
    token=token,
    collection_name=collection_name,
    duplicates_policy=DuplicatePolicy.SKIP,
    embedding_dim=384,
)

retriever = AstraEmbeddingRetriever(document_store=document_store)
```

<a id="haystack_integrations.components.retrievers.astra.retriever.AstraEmbeddingRetriever.__init__"></a>

#### AstraEmbeddingRetriever.\_\_init\_\_

```python
def __init__(document_store: AstraDocumentStore,
             filters: Optional[Dict[str, Any]] = None,
             top_k: int = 10,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE)
```

**Arguments**:

- `document_store`: An instance of AstraDocumentStore.
- `filters`: a dictionary with filters to narrow down the search space.
- `top_k`: the maximum number of documents to retrieve.
- `filter_policy`: Policy to determine how filters are applied.

<a id="haystack_integrations.components.retrievers.astra.retriever.AstraEmbeddingRetriever.run"></a>

#### AstraEmbeddingRetriever.run

```python
@component.output_types(documents=List[Document])
def run(query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None) -> Dict[str, List[Document]]
```

Retrieve documents from the AstraDocumentStore.

**Arguments**:

- `query_embedding`: floats representing the query embedding
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: the maximum number of documents to retrieve.

**Returns**:

a dictionary with the following keys:
- `documents`: A list of documents retrieved from the AstraDocumentStore.

<a id="haystack_integrations.components.retrievers.astra.retriever.AstraEmbeddingRetriever.to_dict"></a>

#### AstraEmbeddingRetriever.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.astra.retriever.AstraEmbeddingRetriever.from_dict"></a>

#### AstraEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "AstraEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.document_stores.astra.document_store"></a>

## Module haystack\_integrations.document\_stores.astra.document\_store

<a id="haystack_integrations.document_stores.astra.document_store.AstraDocumentStore"></a>

### AstraDocumentStore

An AstraDocumentStore document store for Haystack.

Example Usage:
```python
from haystack_integrations.document_stores.astra import AstraDocumentStore

document_store = AstraDocumentStore(
    api_endpoint=api_endpoint,
    token=token,
    collection_name=collection_name,
    duplicates_policy=DuplicatePolicy.SKIP,
    embedding_dim=384,
)
```

<a id="haystack_integrations.document_stores.astra.document_store.AstraDocumentStore.__init__"></a>

#### AstraDocumentStore.\_\_init\_\_

```python
def __init__(
        api_endpoint: Secret = Secret.from_env_var("ASTRA_DB_API_ENDPOINT"),
        token: Secret = Secret.from_env_var("ASTRA_DB_APPLICATION_TOKEN"),
        collection_name: str = "documents",
        embedding_dimension: int = 768,
        duplicates_policy: DuplicatePolicy = DuplicatePolicy.NONE,
        similarity: str = "cosine",
        namespace: Optional[str] = None)
```

The connection to Astra DB is established and managed through the JSON API.

The required credentials (api endpoint and application token) can be generated
through the UI by clicking and the connect tab, and then selecting JSON API and
Generate Configuration.

**Arguments**:

- `api_endpoint`: the Astra DB API endpoint.
- `token`: the Astra DB application token.
- `collection_name`: the current collection in the keyspace in the current Astra DB.
- `embedding_dimension`: dimension of embedding vector.
- `duplicates_policy`: handle duplicate documents based on DuplicatePolicy parameter options.
Parameter options : (`SKIP`, `OVERWRITE`, `FAIL`, `NONE`)
- `DuplicatePolicy.NONE`: Default policy, If a Document with the same ID already exists,
      it is skipped and not written.
- `DuplicatePolicy.SKIP`: if a Document with the same ID already exists, it is skipped and not written.
- `DuplicatePolicy.OVERWRITE`: if a Document with the same ID already exists, it is overwritten.
- `DuplicatePolicy.FAIL`: if a Document with the same ID already exists, an error is raised.
- `similarity`: the similarity function used to compare document vectors.

**Raises**:

- `ValueError`: if the API endpoint or token is not set.

<a id="haystack_integrations.document_stores.astra.document_store.AstraDocumentStore.from_dict"></a>

#### AstraDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "AstraDocumentStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.document_stores.astra.document_store.AstraDocumentStore.to_dict"></a>

#### AstraDocumentStore.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.astra.document_store.AstraDocumentStore.write_documents"></a>

#### AstraDocumentStore.write\_documents

```python
def write_documents(documents: List[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Indexes documents for later queries.

**Arguments**:

- `documents`: a list of Haystack Document objects.
- `policy`: handle duplicate documents based on DuplicatePolicy parameter options.
Parameter options : (`SKIP`, `OVERWRITE`, `FAIL`, `NONE`)
- `DuplicatePolicy.NONE`: Default policy, If a Document with the same ID already exists,
    it is skipped and not written.
- `DuplicatePolicy.SKIP`: If a Document with the same ID already exists,
    it is skipped and not written.
- `DuplicatePolicy.OVERWRITE`: If a Document with the same ID already exists, it is overwritten.
- `DuplicatePolicy.FAIL`: If a Document with the same ID already exists, an error is raised.

**Raises**:

- `ValueError`: if the documents are not of type Document or dict.
- `DuplicateDocumentError`: if a document with the same ID already exists and policy is set to FAIL.
- `Exception`: if the document ID is not a string or if `id` and `_id` are both present in the document.

**Returns**:

number of documents written.

<a id="haystack_integrations.document_stores.astra.document_store.AstraDocumentStore.count_documents"></a>

#### AstraDocumentStore.count\_documents

```python
def count_documents() -> int
```

Counts the number of documents in the document store.

**Returns**:

the number of documents in the document store.

<a id="haystack_integrations.document_stores.astra.document_store.AstraDocumentStore.filter_documents"></a>

#### AstraDocumentStore.filter\_documents

```python
def filter_documents(
        filters: Optional[Dict[str, Any]] = None) -> List[Document]
```

Returns at most 1000 documents that match the filter.

**Arguments**:

- `filters`: filters to apply.

**Raises**:

- `AstraDocumentStoreFilterError`: if the filter is invalid or not supported by this class.

**Returns**:

matching documents.

<a id="haystack_integrations.document_stores.astra.document_store.AstraDocumentStore.get_documents_by_id"></a>

#### AstraDocumentStore.get\_documents\_by\_id

```python
def get_documents_by_id(ids: List[str]) -> List[Document]
```

Gets documents by their IDs.

**Arguments**:

- `ids`: the IDs of the documents to retrieve.

**Returns**:

the matching documents.

<a id="haystack_integrations.document_stores.astra.document_store.AstraDocumentStore.get_document_by_id"></a>

#### AstraDocumentStore.get\_document\_by\_id

```python
def get_document_by_id(document_id: str) -> Document
```

Gets a document by its ID.

**Arguments**:

- `document_id`: the ID to filter by

**Raises**:

- `MissingDocumentError`: if the document is not found

**Returns**:

the found document

<a id="haystack_integrations.document_stores.astra.document_store.AstraDocumentStore.search"></a>

#### AstraDocumentStore.search

```python
def search(query_embedding: List[float],
           top_k: int,
           filters: Optional[Dict[str, Any]] = None) -> List[Document]
```

Perform a search for a list of queries.

**Arguments**:

- `query_embedding`: a list of query embeddings.
- `top_k`: the number of results to return.
- `filters`: filters to apply during search.

**Returns**:

matching documents.

<a id="haystack_integrations.document_stores.astra.document_store.AstraDocumentStore.delete_documents"></a>

#### AstraDocumentStore.delete\_documents

```python
def delete_documents(document_ids: List[str]) -> None
```

Deletes documents from the document store.

**Arguments**:

- `document_ids`: IDs of the documents to delete.
- `delete_all`: if `True`, delete all documents.

**Raises**:

- `MissingDocumentError`: if no document was deleted but document IDs were provided.

<a id="haystack_integrations.document_stores.astra.document_store.AstraDocumentStore.delete_all_documents"></a>

#### AstraDocumentStore.delete\_all\_documents

```python
def delete_all_documents() -> None
```

Deletes all documents from the document store.

<a id="haystack_integrations.document_stores.astra.errors"></a>

## Module haystack\_integrations.document\_stores.astra.errors

<a id="haystack_integrations.document_stores.astra.errors.AstraDocumentStoreError"></a>

### AstraDocumentStoreError

Parent class for all AstraDocumentStore errors.

<a id="haystack_integrations.document_stores.astra.errors.AstraDocumentStoreFilterError"></a>

### AstraDocumentStoreFilterError

Raised when an invalid filter is passed to AstraDocumentStore.

<a id="haystack_integrations.document_stores.astra.errors.AstraDocumentStoreConfigError"></a>

### AstraDocumentStoreConfigError

Raised when an invalid configuration is passed to AstraDocumentStore.
