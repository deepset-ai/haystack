---
title: "Pinecone"
id: integrations-pinecone
description: "Pinecone integration for Haystack"
slug: "/integrations-pinecone"
---

<a id="haystack_integrations.components.retrievers.pinecone.embedding_retriever"></a>

## Module haystack\_integrations.components.retrievers.pinecone.embedding\_retriever

<a id="haystack_integrations.components.retrievers.pinecone.embedding_retriever.PineconeEmbeddingRetriever"></a>

### PineconeEmbeddingRetriever

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

<a id="haystack_integrations.components.retrievers.pinecone.embedding_retriever.PineconeEmbeddingRetriever.__init__"></a>

#### PineconeEmbeddingRetriever.\_\_init\_\_

```python
def __init__(*,
             document_store: PineconeDocumentStore,
             filters: Optional[Dict[str, Any]] = None,
             top_k: int = 10,
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE)
```

**Arguments**:

- `document_store`: The Pinecone Document Store.
- `filters`: Filters applied to the retrieved Documents.
- `top_k`: Maximum number of Documents to return.
- `filter_policy`: Policy to determine how filters are applied.

**Raises**:

- `ValueError`: If `document_store` is not an instance of `PineconeDocumentStore`.

<a id="haystack_integrations.components.retrievers.pinecone.embedding_retriever.PineconeEmbeddingRetriever.to_dict"></a>

#### PineconeEmbeddingRetriever.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.pinecone.embedding_retriever.PineconeEmbeddingRetriever.from_dict"></a>

#### PineconeEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "PineconeEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.retrievers.pinecone.embedding_retriever.PineconeEmbeddingRetriever.run"></a>

#### PineconeEmbeddingRetriever.run

```python
@component.output_types(documents=List[Document])
def run(query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None) -> Dict[str, List[Document]]
```

Retrieve documents from the `PineconeDocumentStore`, based on their dense embeddings.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of `Document`s to return.

**Returns**:

List of Document similar to `query_embedding`.

<a id="haystack_integrations.components.retrievers.pinecone.embedding_retriever.PineconeEmbeddingRetriever.run_async"></a>

#### PineconeEmbeddingRetriever.run\_async

```python
@component.output_types(documents=List[Document])
async def run_async(query_embedding: List[float],
                    filters: Optional[Dict[str, Any]] = None,
                    top_k: Optional[int] = None) -> Dict[str, List[Document]]
```

Asynchronously retrieve documents from the `PineconeDocumentStore`, based on their dense embeddings.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
the `filter_policy` chosen at retriever initialization. See init method docstring for more
details.
- `top_k`: Maximum number of `Document`s to return.

**Returns**:

List of Document similar to `query_embedding`.

<a id="haystack_integrations.document_stores.pinecone.document_store"></a>

## Module haystack\_integrations.document\_stores.pinecone.document\_store

<a id="haystack_integrations.document_stores.pinecone.document_store.METADATA_SUPPORTED_TYPES"></a>

#### METADATA\_SUPPORTED\_TYPES

List[str] is supported and checked separately

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore"></a>

### PineconeDocumentStore

A Document Store using [Pinecone vector database](https://www.pinecone.io/).

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.__init__"></a>

#### PineconeDocumentStore.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var("PINECONE_API_KEY"),
             index: str = "default",
             namespace: str = "default",
             batch_size: int = 100,
             dimension: int = 768,
             spec: Optional[Dict[str, Any]] = None,
             metric: Literal["cosine", "euclidean", "dotproduct"] = "cosine")
```

Creates a new PineconeDocumentStore instance.

It is meant to be connected to a Pinecone index and namespace.

**Arguments**:

- `api_key`: The Pinecone API key.
- `index`: The Pinecone index to connect to. If the index does not exist, it will be created.
- `namespace`: The Pinecone namespace to connect to. If the namespace does not exist, it will be created
at the first write.
- `batch_size`: The number of documents to write in a single batch. When setting this parameter,
consider [documented Pinecone limits](https://docs.pinecone.io/reference/quotas-and-limits).
- `dimension`: The dimension of the embeddings. This parameter is only used when creating a new index.
- `spec`: The Pinecone spec to use when creating a new index. Allows choosing between serverless and pod
deployment options and setting additional parameters. Refer to the
[Pinecone documentation](https://docs.pinecone.io/reference/api/control-plane/create_index) for more
details.
If not provided, a default spec with serverless deployment in the `us-east-1` region will be used
(compatible with the free tier).
- `metric`: The metric to use for similarity search. This parameter is only used when creating a new index.

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.close"></a>

#### PineconeDocumentStore.close

```python
def close()
```

Close the associated synchronous resources.

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.close_async"></a>

#### PineconeDocumentStore.close\_async

```python
async def close_async()
```

Close the associated asynchronous resources. To be invoked manually when the Document Store is no longer needed.

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.from_dict"></a>

#### PineconeDocumentStore.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "PineconeDocumentStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.to_dict"></a>

#### PineconeDocumentStore.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.count_documents"></a>

#### PineconeDocumentStore.count\_documents

```python
def count_documents() -> int
```

Returns how many documents are present in the document store.

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.count_documents_async"></a>

#### PineconeDocumentStore.count\_documents\_async

```python
async def count_documents_async() -> int
```

Asynchronously returns how many documents are present in the document store.

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.write_documents"></a>

#### PineconeDocumentStore.write\_documents

```python
def write_documents(documents: List[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Writes Documents to Pinecone.

**Arguments**:

- `documents`: A list of Documents to write to the document store.
- `policy`: The duplicate policy to use when writing documents.
PineconeDocumentStore only supports `DuplicatePolicy.OVERWRITE`.

**Returns**:

The number of documents written to the document store.

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.write_documents_async"></a>

#### PineconeDocumentStore.write\_documents\_async

```python
async def write_documents_async(
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int
```

Asynchronously writes Documents to Pinecone.

**Arguments**:

- `documents`: A list of Documents to write to the document store.
- `policy`: The duplicate policy to use when writing documents.
PineconeDocumentStore only supports `DuplicatePolicy.OVERWRITE`.

**Returns**:

The number of documents written to the document store.

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.filter_documents"></a>

#### PineconeDocumentStore.filter\_documents

```python
def filter_documents(
        filters: Optional[Dict[str, Any]] = None) -> List[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

**Arguments**:

- `filters`: The filters to apply to the document list.

**Returns**:

A list of Documents that match the given filters.

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.filter_documents_async"></a>

#### PineconeDocumentStore.filter\_documents\_async

```python
async def filter_documents_async(
        filters: Optional[Dict[str, Any]] = None) -> List[Document]
```

Asynchronously returns the documents that match the filters provided.

**Arguments**:

- `filters`: The filters to apply to the document list.

**Returns**:

A list of Documents that match the given filters.

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.delete_documents"></a>

#### PineconeDocumentStore.delete\_documents

```python
def delete_documents(document_ids: List[str]) -> None
```

Deletes documents that match the provided `document_ids` from the document store.

**Arguments**:

- `document_ids`: the document ids to delete

<a id="haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore.delete_documents_async"></a>

#### PineconeDocumentStore.delete\_documents\_async

```python
async def delete_documents_async(document_ids: List[str]) -> None
```

Asynchronously deletes documents that match the provided `document_ids` from the document store.

**Arguments**:

- `document_ids`: the document ids to delete
