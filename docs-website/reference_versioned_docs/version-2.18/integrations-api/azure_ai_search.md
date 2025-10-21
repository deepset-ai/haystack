---
title: "Azure AI Search"
id: integrations-azure_ai_search
description: "Azure AI Search integration for Haystack"
slug: "/integrations-azure_ai_search"
---

<a id="haystack_integrations.components.retrievers.azure_ai_search.embedding_retriever"></a>

# Module haystack\_integrations.components.retrievers.azure\_ai\_search.embedding\_retriever

<a id="haystack_integrations.components.retrievers.azure_ai_search.embedding_retriever.AzureAISearchEmbeddingRetriever"></a>

## AzureAISearchEmbeddingRetriever

Retrieves documents from the AzureAISearchDocumentStore using a vector similarity metric.
Must be connected to the AzureAISearchDocumentStore to run.

<a id="haystack_integrations.components.retrievers.azure_ai_search.embedding_retriever.AzureAISearchEmbeddingRetriever.__init__"></a>

#### AzureAISearchEmbeddingRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: AzureAISearchDocumentStore,
             filters: Optional[Dict[str, Any]] = None,
             top_k: int = 10,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
             **kwargs: Any)
```

Create the AzureAISearchEmbeddingRetriever component.

**Arguments**:

- `document_store`: An instance of AzureAISearchDocumentStore to use with the Retriever.
- `filters`: Filters applied when fetching documents from the Document Store.
- `top_k`: Maximum number of documents to return.
- `filter_policy`: Policy to determine how filters are applied.
- `kwargs`: Additional keyword arguments to pass to the Azure AI's search endpoint.
Some of the supported parameters:
    - `query_type`: A string indicating the type of query to perform. Possible values are
    'simple','full' and 'semantic'.
    - `semantic_configuration_name`: The name of semantic configuration to be used when
    processing semantic queries.
For more information on parameters, see the
[official Azure AI Search documentation](https://learn.microsoft.com/en-us/azure/search/).

<a id="haystack_integrations.components.retrievers.azure_ai_search.embedding_retriever.AzureAISearchEmbeddingRetriever.to_dict"></a>

#### AzureAISearchEmbeddingRetriever.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.azure_ai_search.embedding_retriever.AzureAISearchEmbeddingRetriever.from_dict"></a>

#### AzureAISearchEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "AzureAISearchEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.azure_ai_search.embedding_retriever.AzureAISearchEmbeddingRetriever.run"></a>

#### AzureAISearchEmbeddingRetriever.run

```python
@component.output_types(documents=List[Document])
def run(query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None) -> Dict[str, List[Document]]
```

Retrieve documents from the AzureAISearchDocumentStore.

**Arguments**:

- `query_embedding`: A list of floats representing the query embedding.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See `__init__` method docstring for more
details.
- `top_k`: The maximum number of documents to retrieve.

**Returns**:

Dictionary with the following keys:
- `documents`: A list of documents retrieved from the AzureAISearchDocumentStore.

<a id="haystack_integrations.document_stores.azure_ai_search.document_store"></a>

# Module haystack\_integrations.document\_stores.azure\_ai\_search.document\_store

<a id="haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore"></a>

## AzureAISearchDocumentStore

<a id="haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore.__init__"></a>

#### AzureAISearchDocumentStore.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var("AZURE_AI_SEARCH_API_KEY",
                                                   strict=False),
             azure_endpoint: Secret = Secret.from_env_var(
                 "AZURE_AI_SEARCH_ENDPOINT", strict=True),
             index_name: str = "default",
             embedding_dimension: int = 768,
             metadata_fields: Optional[Dict[str, Union[SearchField,
                                                       type]]] = None,
             vector_search_configuration: Optional[VectorSearch] = None,
             include_search_metadata: bool = False,
             **index_creation_kwargs: Any)
```

A document store using [Azure AI Search](https://azure.microsoft.com/products/ai-services/ai-search/)

as the backend.

**Arguments**:

- `azure_endpoint`: The URL endpoint of an Azure AI Search service.
- `api_key`: The API key to use for authentication.
- `index_name`: Name of index in Azure AI Search, if it doesn't exist it will be created.
- `embedding_dimension`: Dimension of the embeddings.
- `metadata_fields`: A dictionary mapping metadata field names to their corresponding field definitions.
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
- `vector_search_configuration`: Configuration option related to vector search.
Default configuration uses the HNSW algorithm with cosine similarity to handle vector searches.
- `include_search_metadata`: Whether to include Azure AI Search metadata fields
in the returned documents. When set to True, the `meta` field of the returned
documents will contain the @search.score, @search.reranker_score, @search.highlights,
@search.captions, and other fields returned by Azure AI Search.
- `index_creation_kwargs`: Optional keyword parameters to be passed to `SearchIndex` class
during index creation. Some of the supported parameters:
        - `semantic_search`: Defines semantic configuration of the search index. This parameter is needed
        to enable semantic search capabilities in index.
        - `similarity`: The type of similarity algorithm to be used when scoring and ranking the documents
        matching a search query. The similarity algorithm can only be defined at index creation time and
        cannot be modified on existing indexes.

For more information on parameters, see the [official Azure AI Search documentation](https://learn.microsoft.com/en-us/azure/search/).

<a id="haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore.to_dict"></a>

#### AzureAISearchDocumentStore.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore.from_dict"></a>

#### AzureAISearchDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "AzureAISearchDocumentStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore.count_documents"></a>

#### AzureAISearchDocumentStore.count\_documents

```python
def count_documents() -> int
```

Returns how many documents are present in the search index.

**Returns**:

list of retrieved documents.

<a id="haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore.write_documents"></a>

#### AzureAISearchDocumentStore.write\_documents

```python
def write_documents(documents: List[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Writes the provided documents to search index.

**Arguments**:

- `documents`: documents to write to the index.
- `policy`: Policy to determine how duplicates are handled.

**Raises**:

- `ValueError`: If the documents are not of type Document.
- `TypeError`: If the document ids are not strings.

**Returns**:

the number of documents added to index.

<a id="haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore.delete_documents"></a>

#### AzureAISearchDocumentStore.delete\_documents

```python
def delete_documents(document_ids: List[str]) -> None
```

Deletes all documents with a matching document_ids from the search index.

**Arguments**:

- `document_ids`: ids of the documents to be deleted.

<a id="haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore.search_documents"></a>

#### AzureAISearchDocumentStore.search\_documents

```python
def search_documents(search_text: str = "*",
                     top_k: int = 10) -> List[Document]
```

Returns all documents that match the provided search_text.

If search_text is None, returns all documents.

**Arguments**:

- `search_text`: the text to search for in the Document list.
- `top_k`: Maximum number of documents to return.

**Returns**:

A list of Documents that match the given search_text.

<a id="haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore.filter_documents"></a>

#### AzureAISearchDocumentStore.filter\_documents

```python
def filter_documents(
        filters: Optional[Dict[str, Any]] = None) -> List[Document]
```

Returns the documents that match the provided filters.

Filters should be given as a dictionary supporting filtering by metadata. For details on
filters, see the [metadata filtering documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

**Arguments**:

- `filters`: the filters to apply to the document list.

**Returns**:

A list of Documents that match the given filters.
