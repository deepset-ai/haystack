---
title: "Chroma"
id: integrations-chroma
description: "Chroma integration for Haystack"
slug: "/integrations-chroma"
---

<a id="haystack_integrations.components.retrievers.chroma.retriever"></a>

## Module haystack\_integrations.components.retrievers.chroma.retriever

<a id="haystack_integrations.components.retrievers.chroma.retriever.ChromaQueryTextRetriever"></a>

### ChromaQueryTextRetriever

A component for retrieving documents from a [Chroma database](https://docs.trychroma.com/) using the `query` API.

Example usage:
```python
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever

file_paths = ...

# Chroma is used in-memory so we use the same instances in the two pipelines below
document_store = ChromaDocumentStore()

indexing = Pipeline()
indexing.add_component("converter", TextFileToDocument())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("converter", "writer")
indexing.run({"converter": {"sources": file_paths}})

querying = Pipeline()
querying.add_component("retriever", ChromaQueryTextRetriever(document_store))
results = querying.run({"retriever": {"query": "Variable declarations", "top_k": 3}})

for d in results["retriever"]["documents"]:
    print(d.meta, d.score)
```

<a id="haystack_integrations.components.retrievers.chroma.retriever.ChromaQueryTextRetriever.__init__"></a>

#### ChromaQueryTextRetriever.\_\_init\_\_

```python
def __init__(document_store: ChromaDocumentStore,
             filters: Optional[Dict[str, Any]] = None,
             top_k: int = 10,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE)
```

**Arguments**:

- `document_store`: an instance of `ChromaDocumentStore`.
- `filters`: filters to narrow down the search space.
- `top_k`: the maximum number of documents to retrieve.
- `filter_policy`: Policy to determine how filters are applied.

<a id="haystack_integrations.components.retrievers.chroma.retriever.ChromaQueryTextRetriever.run"></a>

#### ChromaQueryTextRetriever.run

```python
@component.output_types(documents=List[Document])
def run(query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None) -> Dict[str, Any]
```

Run the retriever on the given input data.

**Arguments**:

- `query`: The input data for the retriever. In this case, a plain-text query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: The maximum number of documents to retrieve.
If not specified, the default value from the constructor is used.

**Raises**:

- `ValueError`: If the specified document store is not found or is not a MemoryDocumentStore instance.

**Returns**:

A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

<a id="haystack_integrations.components.retrievers.chroma.retriever.ChromaQueryTextRetriever.run_async"></a>

#### ChromaQueryTextRetriever.run\_async

```python
@component.output_types(documents=List[Document])
async def run_async(query: str,
                    filters: Optional[Dict[str, Any]] = None,
                    top_k: Optional[int] = None) -> Dict[str, Any]
```

Asynchronously run the retriever on the given input data.

Asynchronous methods are only supported for HTTP connections.

**Arguments**:

- `query`: The input data for the retriever. In this case, a plain-text query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: The maximum number of documents to retrieve.
If not specified, the default value from the constructor is used.

**Raises**:

- `ValueError`: If the specified document store is not found or is not a MemoryDocumentStore instance.

**Returns**:

A dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

<a id="haystack_integrations.components.retrievers.chroma.retriever.ChromaQueryTextRetriever.from_dict"></a>

#### ChromaQueryTextRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "ChromaQueryTextRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.chroma.retriever.ChromaQueryTextRetriever.to_dict"></a>

#### ChromaQueryTextRetriever.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.chroma.retriever.ChromaEmbeddingRetriever"></a>

### ChromaEmbeddingRetriever

A component for retrieving documents from a [Chroma database](https://docs.trychroma.com/) using embeddings.

<a id="haystack_integrations.components.retrievers.chroma.retriever.ChromaEmbeddingRetriever.__init__"></a>

#### ChromaEmbeddingRetriever.\_\_init\_\_

```python
def __init__(document_store: ChromaDocumentStore,
             filters: Optional[Dict[str, Any]] = None,
             top_k: int = 10,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE)
```

**Arguments**:

- `document_store`: an instance of `ChromaDocumentStore`.
- `filters`: filters to narrow down the search space.
- `top_k`: the maximum number of documents to retrieve.
- `filter_policy`: Policy to determine how filters are applied.

<a id="haystack_integrations.components.retrievers.chroma.retriever.ChromaEmbeddingRetriever.run"></a>

#### ChromaEmbeddingRetriever.run

```python
@component.output_types(documents=List[Document])
def run(query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None) -> Dict[str, Any]
```

Run the retriever on the given input data.

**Arguments**:

- `query_embedding`: the query embeddings.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: the maximum number of documents to retrieve.
If not specified, the default value from the constructor is used.

**Returns**:

a dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

<a id="haystack_integrations.components.retrievers.chroma.retriever.ChromaEmbeddingRetriever.run_async"></a>

#### ChromaEmbeddingRetriever.run\_async

```python
@component.output_types(documents=List[Document])
async def run_async(query_embedding: List[float],
                    filters: Optional[Dict[str, Any]] = None,
                    top_k: Optional[int] = None) -> Dict[str, Any]
```

Asynchronously run the retriever on the given input data.

Asynchronous methods are only supported for HTTP connections.

**Arguments**:

- `query_embedding`: the query embeddings.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: the maximum number of documents to retrieve.
If not specified, the default value from the constructor is used.

**Returns**:

a dictionary with the following keys:
- `documents`: List of documents returned by the search engine.

<a id="haystack_integrations.components.retrievers.chroma.retriever.ChromaEmbeddingRetriever.from_dict"></a>

#### ChromaEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "ChromaEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.chroma.retriever.ChromaEmbeddingRetriever.to_dict"></a>

#### ChromaEmbeddingRetriever.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.chroma.document_store"></a>

## Module haystack\_integrations.document\_stores.chroma.document\_store

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore"></a>

### ChromaDocumentStore

A document store using [Chroma](https://docs.trychroma.com/) as the backend.

We use the `collection.get` API to implement the document store protocol,
the `collection.search` API will be used in the retriever instead.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.__init__"></a>

#### ChromaDocumentStore.\_\_init\_\_

```python
def __init__(collection_name: str = "documents",
             embedding_function: str = "default",
             persist_path: Optional[str] = None,
             host: Optional[str] = None,
             port: Optional[int] = None,
             distance_function: Literal["l2", "cosine", "ip"] = "l2",
             metadata: Optional[dict] = None,
             **embedding_function_params: Any)
```

Creates a new ChromaDocumentStore instance.

It is meant to be connected to a Chroma collection.

Note: for the component to be part of a serializable pipeline, the __init__
parameters must be serializable, reason why we use a registry to configure the
embedding function passing a string.

**Arguments**:

- `collection_name`: the name of the collection to use in the database.
- `embedding_function`: the name of the embedding function to use to embed the query
- `persist_path`: Path for local persistent storage. Cannot be used in combination with `host` and `port`.
If none of `persist_path`, `host`, and `port` is specified, the database will be `in-memory`.
- `host`: The host address for the remote Chroma HTTP client connection. Cannot be used with `persist_path`.
- `port`: The port number for the remote Chroma HTTP client connection. Cannot be used with `persist_path`.
- `distance_function`: The distance metric for the embedding space.
- `"l2"` computes the Euclidean (straight-line) distance between vectors,
where smaller scores indicate more similarity.
- `"cosine"` computes the cosine similarity between vectors,
with higher scores indicating greater similarity.
- `"ip"` stands for inner product, where higher scores indicate greater similarity between vectors.
**Note**: `distance_function` can only be set during the creation of a collection.
To change the distance metric of an existing collection, consider cloning the collection.
- `metadata`: a dictionary of chromadb collection parameters passed directly to chromadb's client
method `create_collection`. If it contains the key `"hnsw:space"`, the value will take precedence over the
`distance_function` parameter above.
- `embedding_function_params`: additional parameters to pass to the embedding function.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.count_documents"></a>

#### ChromaDocumentStore.count\_documents

```python
def count_documents() -> int
```

Returns how many documents are present in the document store.

**Returns**:

how many documents are present in the document store.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.count_documents_async"></a>

#### ChromaDocumentStore.count\_documents\_async

```python
async def count_documents_async() -> int
```

Asynchronously returns how many documents are present in the document store.

Asynchronous methods are only supported for HTTP connections.

**Returns**:

how many documents are present in the document store.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.filter_documents"></a>

#### ChromaDocumentStore.filter\_documents

```python
def filter_documents(
        filters: Optional[Dict[str, Any]] = None) -> List[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

**Arguments**:

- `filters`: the filters to apply to the document list.

**Returns**:

a list of Documents that match the given filters.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.filter_documents_async"></a>

#### ChromaDocumentStore.filter\_documents\_async

```python
async def filter_documents_async(
        filters: Optional[Dict[str, Any]] = None) -> List[Document]
```

Asynchronously returns the documents that match the filters provided.

Asynchronous methods are only supported for HTTP connections.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

**Arguments**:

- `filters`: the filters to apply to the document list.

**Returns**:

a list of Documents that match the given filters.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.write_documents"></a>

#### ChromaDocumentStore.write\_documents

```python
def write_documents(documents: List[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int
```

Writes (or overwrites) documents into the store.

**Arguments**:

- `documents`: A list of documents to write into the document store.
- `policy`: Not supported at the moment.

**Raises**:

- `ValueError`: When input is not valid.

**Returns**:

The number of documents written

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.write_documents_async"></a>

#### ChromaDocumentStore.write\_documents\_async

```python
async def write_documents_async(
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int
```

Asynchronously writes (or overwrites) documents into the store.

Asynchronous methods are only supported for HTTP connections.

**Arguments**:

- `documents`: A list of documents to write into the document store.
- `policy`: Not supported at the moment.

**Raises**:

- `ValueError`: When input is not valid.

**Returns**:

The number of documents written

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.delete_documents"></a>

#### ChromaDocumentStore.delete\_documents

```python
def delete_documents(document_ids: List[str]) -> None
```

Deletes all documents with a matching document_ids from the document store.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.delete_documents_async"></a>

#### ChromaDocumentStore.delete\_documents\_async

```python
async def delete_documents_async(document_ids: List[str]) -> None
```

Asynchronously deletes all documents with a matching document_ids from the document store.

Asynchronous methods are only supported for HTTP connections.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.delete_all_documents"></a>

#### ChromaDocumentStore.delete\_all\_documents

```python
def delete_all_documents(*, recreate_index: bool = False) -> None
```

Deletes all documents in the document store.

A fast way to clear all documents from the document store while preserving any collection settings and mappings.

**Arguments**:

- `recreate_index`: Whether to recreate the index after deleting all documents.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.delete_all_documents_async"></a>

#### ChromaDocumentStore.delete\_all\_documents\_async

```python
async def delete_all_documents_async(*, recreate_index: bool = False) -> None
```

Asynchronously deletes all documents in the document store.

A fast way to clear all documents from the document store while preserving any collection settings and mappings.

**Arguments**:

- `recreate_index`: Whether to recreate the index after deleting all documents.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.search"></a>

#### ChromaDocumentStore.search

```python
def search(queries: List[str],
           top_k: int,
           filters: Optional[Dict[str, Any]] = None) -> List[List[Document]]
```

Search the documents in the store using the provided text queries.

**Arguments**:

- `queries`: the list of queries to search for.
- `top_k`: top_k documents to return for each query.
- `filters`: a dictionary of filters to apply to the search. Accepts filters in haystack format.

**Returns**:

matching documents for each query.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.search_async"></a>

#### ChromaDocumentStore.search\_async

```python
async def search_async(
        queries: List[str],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None) -> List[List[Document]]
```

Asynchronously search the documents in the store using the provided text queries.

Asynchronous methods are only supported for HTTP connections.

**Arguments**:

- `queries`: the list of queries to search for.
- `top_k`: top_k documents to return for each query.
- `filters`: a dictionary of filters to apply to the search. Accepts filters in haystack format.

**Returns**:

matching documents for each query.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.search_embeddings"></a>

#### ChromaDocumentStore.search\_embeddings

```python
def search_embeddings(
        query_embeddings: List[List[float]],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None) -> List[List[Document]]
```

Perform vector search on the stored document, pass the embeddings of the queries instead of their text.

**Arguments**:

- `query_embeddings`: a list of embeddings to use as queries.
- `top_k`: the maximum number of documents to retrieve.
- `filters`: a dictionary of filters to apply to the search. Accepts filters in haystack format.

**Returns**:

a list of lists of documents that match the given filters.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.search_embeddings_async"></a>

#### ChromaDocumentStore.search\_embeddings\_async

```python
async def search_embeddings_async(
        query_embeddings: List[List[float]],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None) -> List[List[Document]]
```

Asynchronously perform vector search on the stored document, pass the embeddings of the queries instead of

their text.

Asynchronous methods are only supported for HTTP connections.

**Arguments**:

- `query_embeddings`: a list of embeddings to use as queries.
- `top_k`: the maximum number of documents to retrieve.
- `filters`: a dictionary of filters to apply to the search. Accepts filters in haystack format.

**Returns**:

a list of lists of documents that match the given filters.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.from_dict"></a>

#### ChromaDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "ChromaDocumentStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore.to_dict"></a>

#### ChromaDocumentStore.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.chroma.errors"></a>

## Module haystack\_integrations.document\_stores.chroma.errors

<a id="haystack_integrations.document_stores.chroma.errors.ChromaDocumentStoreError"></a>

### ChromaDocumentStoreError

Parent class for all ChromaDocumentStore exceptions.

<a id="haystack_integrations.document_stores.chroma.errors.ChromaDocumentStoreFilterError"></a>

### ChromaDocumentStoreFilterError

Raised when a filter is not valid for a ChromaDocumentStore.

<a id="haystack_integrations.document_stores.chroma.errors.ChromaDocumentStoreConfigError"></a>

### ChromaDocumentStoreConfigError

Raised when a configuration is not valid for a ChromaDocumentStore.

<a id="haystack_integrations.document_stores.chroma.utils"></a>

## Module haystack\_integrations.document\_stores.chroma.utils

<a id="haystack_integrations.document_stores.chroma.utils.get_embedding_function"></a>

#### get\_embedding\_function

```python
def get_embedding_function(function_name: str,
                           **kwargs: Any) -> EmbeddingFunction
```

Load an embedding function by name.

**Arguments**:

- `function_name`: the name of the embedding function.
- `kwargs`: additional arguments to pass to the embedding function.

**Raises**:

- `ChromaDocumentStoreConfigError`: if the function name is invalid.

**Returns**:

the loaded embedding function.
