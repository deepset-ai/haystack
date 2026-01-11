---
title: "Retrievers"
id: retrievers-api
description: "Sweeps through a Document Store and returns a set of candidate Documents that are relevant to the query."
slug: "/retrievers-api"
---

<a id="auto_merging_retriever"></a>

## Module auto\_merging\_retriever

<a id="auto_merging_retriever.AutoMergingRetriever"></a>

### AutoMergingRetriever

A retriever which returns parent documents of the matched leaf nodes documents, based on a threshold setting.

The AutoMergingRetriever assumes you have a hierarchical tree structure of documents, where the leaf nodes
are indexed in a document store. See the HierarchicalDocumentSplitter for more information on how to create
such a structure. During retrieval, if the number of matched leaf documents below the same parent is
higher than a defined threshold, the retriever will return the parent document instead of the individual leaf
documents.

The rational is, given that a paragraph is split into multiple chunks represented as leaf documents, and if for
a given query, multiple chunks are matched, the whole paragraph might be more informative than the individual
chunks alone.

Currently the AutoMergingRetriever can only be used by the following DocumentStores:
- [AstraDB](https://haystack.deepset.ai/integrations/astradb)
- [ElasticSearch](https://haystack.deepset.ai/docs/latest/documentstore/elasticsearch)
- [OpenSearch](https://haystack.deepset.ai/docs/latest/documentstore/opensearch)
- [PGVector](https://haystack.deepset.ai/docs/latest/documentstore/pgvector)
- [Qdrant](https://haystack.deepset.ai/docs/latest/documentstore/qdrant)

```python
from haystack import Document
from haystack.components.preprocessors import HierarchicalDocumentSplitter
from haystack.components.retrievers.auto_merging_retriever import AutoMergingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

# create a hierarchical document structure with 3 levels, where the parent document has 3 children
text = "The sun rose early in the morning. It cast a warm glow over the trees. Birds began to sing."
original_document = Document(content=text)
builder = HierarchicalDocumentSplitter(block_sizes=[10, 3], split_overlap=0, split_by="word")
docs = builder.run([original_document])["documents"]

# store level-1 parent documents and initialize the retriever
doc_store_parents = InMemoryDocumentStore()
for doc in docs["documents"]:
    if doc.meta["children_ids"] and doc.meta["level"] == 1:
        doc_store_parents.write_documents([doc])
retriever = AutoMergingRetriever(doc_store_parents, threshold=0.5)

# assume we retrieved 2 leaf docs from the same parent, the parent document should be returned,
# since it has 3 children and the threshold=0.5, and we retrieved 2 children (2/3 > 0.66(6))
leaf_docs = [doc for doc in docs["documents"] if not doc.meta["children_ids"]]
docs = retriever.run(leaf_docs[4:6])
>> {'documents': [Document(id=538..),
>> content: 'warm glow over the trees. Birds began to sing.',
>> meta: {'block_size': 10, 'parent_id': '835..', 'children_ids': ['c17...', '3ff...', '352...'], 'level': 1, 'source_id': '835...',
>> 'page_number': 1, 'split_id': 1, 'split_idx_start': 45})]}
```

<a id="auto_merging_retriever.AutoMergingRetriever.__init__"></a>

#### AutoMergingRetriever.\_\_init\_\_

```python
def __init__(document_store: DocumentStore, threshold: float = 0.5)
```

Initialize the AutoMergingRetriever.

**Arguments**:

- `document_store`: DocumentStore from which to retrieve the parent documents
- `threshold`: Threshold to decide whether the parent instead of the individual documents is returned

<a id="auto_merging_retriever.AutoMergingRetriever.to_dict"></a>

#### AutoMergingRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="auto_merging_retriever.AutoMergingRetriever.from_dict"></a>

#### AutoMergingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "AutoMergingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary with serialized data.

**Returns**:

An instance of the component.

<a id="auto_merging_retriever.AutoMergingRetriever.run"></a>

#### AutoMergingRetriever.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document])
```

Run the AutoMergingRetriever.

Recursively groups documents by their parents and merges them if they meet the threshold,
continuing up the hierarchy until no more merges are possible.

**Arguments**:

- `documents`: List of leaf documents that were matched by a retriever

**Returns**:

List of documents (could be a mix of different hierarchy levels)

<a id="in_memory/bm25_retriever"></a>

## Module in\_memory/bm25\_retriever

<a id="in_memory/bm25_retriever.InMemoryBM25Retriever"></a>

### InMemoryBM25Retriever

Retrieves documents that are most similar to the query using keyword-based algorithm.

Use this retriever with the InMemoryDocumentStore.

### Usage example

```python
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

docs = [
    Document(content="Python is a popular programming language"),
    Document(content="python ist eine beliebte Programmiersprache"),
]

doc_store = InMemoryDocumentStore()
doc_store.write_documents(docs)
retriever = InMemoryBM25Retriever(doc_store)

result = retriever.run(query="Programmiersprache")

print(result["documents"])
```

<a id="in_memory/bm25_retriever.InMemoryBM25Retriever.__init__"></a>

#### InMemoryBM25Retriever.\_\_init\_\_

```python
def __init__(document_store: InMemoryDocumentStore,
             filters: Optional[dict[str, Any]] = None,
             top_k: int = 10,
             scale_score: bool = False,
             filter_policy: FilterPolicy = FilterPolicy.REPLACE)
```

Create the InMemoryBM25Retriever component.

**Arguments**:

- `document_store`: An instance of InMemoryDocumentStore where the retriever should search for relevant documents.
- `filters`: A dictionary with filters to narrow down the retriever's search space in the document store.
- `top_k`: The maximum number of documents to retrieve.
- `scale_score`: When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
When `False`, uses raw similarity scores.
- `filter_policy`: The filter policy to apply during retrieval.
Filter policy determines how filters are applied when retrieving documents. You can choose:
- `REPLACE` (default): Overrides the initialization filters with the filters specified at runtime.
Use this policy to dynamically change filtering for specific queries.
- `MERGE`: Combines runtime filters with initialization filters to narrow down the search.

**Raises**:

- `ValueError`: If the specified `top_k` is not > 0.

<a id="in_memory/bm25_retriever.InMemoryBM25Retriever.to_dict"></a>

#### InMemoryBM25Retriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="in_memory/bm25_retriever.InMemoryBM25Retriever.from_dict"></a>

#### InMemoryBM25Retriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "InMemoryBM25Retriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="in_memory/bm25_retriever.InMemoryBM25Retriever.run"></a>

#### InMemoryBM25Retriever.run

```python
@component.output_types(documents=list[Document])
def run(query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None)
```

Run the InMemoryBM25Retriever on the given input data.

**Arguments**:

- `query`: The query string for the Retriever.
- `filters`: A dictionary with filters to narrow down the search space when retrieving documents.
- `top_k`: The maximum number of documents to return.
- `scale_score`: When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
When `False`, uses raw similarity scores.

**Raises**:

- `ValueError`: If the specified DocumentStore is not found or is not a InMemoryDocumentStore instance.

**Returns**:

The retrieved documents.

<a id="in_memory/bm25_retriever.InMemoryBM25Retriever.run_async"></a>

#### InMemoryBM25Retriever.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(query: str,
                    filters: Optional[dict[str, Any]] = None,
                    top_k: Optional[int] = None,
                    scale_score: Optional[bool] = None)
```

Run the InMemoryBM25Retriever on the given input data.

**Arguments**:

- `query`: The query string for the Retriever.
- `filters`: A dictionary with filters to narrow down the search space when retrieving documents.
- `top_k`: The maximum number of documents to return.
- `scale_score`: When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
When `False`, uses raw similarity scores.

**Raises**:

- `ValueError`: If the specified DocumentStore is not found or is not a InMemoryDocumentStore instance.

**Returns**:

The retrieved documents.

<a id="in_memory/embedding_retriever"></a>

## Module in\_memory/embedding\_retriever

<a id="in_memory/embedding_retriever.InMemoryEmbeddingRetriever"></a>

### InMemoryEmbeddingRetriever

Retrieves documents that are most semantically similar to the query.

Use this retriever with the InMemoryDocumentStore.

When using this retriever, make sure it has query and document embeddings available.
In indexing pipelines, use a DocumentEmbedder to embed documents.
In query pipelines, use a TextEmbedder to embed queries and send them to the retriever.

### Usage example
```python
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

docs = [
    Document(content="Python is a popular programming language"),
    Document(content="python ist eine beliebte Programmiersprache"),
]
doc_embedder = SentenceTransformersDocumentEmbedder()
doc_embedder.warm_up()
docs_with_embeddings = doc_embedder.run(docs)["documents"]

doc_store = InMemoryDocumentStore()
doc_store.write_documents(docs_with_embeddings)
retriever = InMemoryEmbeddingRetriever(doc_store)

query="Programmiersprache"
text_embedder = SentenceTransformersTextEmbedder()
text_embedder.warm_up()
query_embedding = text_embedder.run(query)["embedding"]

result = retriever.run(query_embedding=query_embedding)

print(result["documents"])
```

<a id="in_memory/embedding_retriever.InMemoryEmbeddingRetriever.__init__"></a>

#### InMemoryEmbeddingRetriever.\_\_init\_\_

```python
def __init__(document_store: InMemoryDocumentStore,
             filters: Optional[dict[str, Any]] = None,
             top_k: int = 10,
             scale_score: bool = False,
             return_embedding: bool = False,
             filter_policy: FilterPolicy = FilterPolicy.REPLACE)
```

Create the InMemoryEmbeddingRetriever component.

**Arguments**:

- `document_store`: An instance of InMemoryDocumentStore where the retriever should search for relevant documents.
- `filters`: A dictionary with filters to narrow down the retriever's search space in the document store.
- `top_k`: The maximum number of documents to retrieve.
- `scale_score`: When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
When `False`, uses raw similarity scores.
- `return_embedding`: When `True`, returns the embedding of the retrieved documents.
When `False`, returns just the documents, without their embeddings.
- `filter_policy`: The filter policy to apply during retrieval.
Filter policy determines how filters are applied when retrieving documents. You can choose:
- `REPLACE` (default): Overrides the initialization filters with the filters specified at runtime.
Use this policy to dynamically change filtering for specific queries.
- `MERGE`: Combines runtime filters with initialization filters to narrow down the search.

**Raises**:

- `ValueError`: If the specified top_k is not > 0.

<a id="in_memory/embedding_retriever.InMemoryEmbeddingRetriever.to_dict"></a>

#### InMemoryEmbeddingRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="in_memory/embedding_retriever.InMemoryEmbeddingRetriever.from_dict"></a>

#### InMemoryEmbeddingRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "InMemoryEmbeddingRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="in_memory/embedding_retriever.InMemoryEmbeddingRetriever.run"></a>

#### InMemoryEmbeddingRetriever.run

```python
@component.output_types(documents=list[Document])
def run(query_embedding: list[float],
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None)
```

Run the InMemoryEmbeddingRetriever on the given input data.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: A dictionary with filters to narrow down the search space when retrieving documents.
- `top_k`: The maximum number of documents to return.
- `scale_score`: When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
When `False`, uses raw similarity scores.
- `return_embedding`: When `True`, returns the embedding of the retrieved documents.
When `False`, returns just the documents, without their embeddings.

**Raises**:

- `ValueError`: If the specified DocumentStore is not found or is not an InMemoryDocumentStore instance.

**Returns**:

The retrieved documents.

<a id="in_memory/embedding_retriever.InMemoryEmbeddingRetriever.run_async"></a>

#### InMemoryEmbeddingRetriever.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(query_embedding: list[float],
                    filters: Optional[dict[str, Any]] = None,
                    top_k: Optional[int] = None,
                    scale_score: Optional[bool] = None,
                    return_embedding: Optional[bool] = None)
```

Run the InMemoryEmbeddingRetriever on the given input data.

**Arguments**:

- `query_embedding`: Embedding of the query.
- `filters`: A dictionary with filters to narrow down the search space when retrieving documents.
- `top_k`: The maximum number of documents to return.
- `scale_score`: When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
When `False`, uses raw similarity scores.
- `return_embedding`: When `True`, returns the embedding of the retrieved documents.
When `False`, returns just the documents, without their embeddings.

**Raises**:

- `ValueError`: If the specified DocumentStore is not found or is not an InMemoryDocumentStore instance.

**Returns**:

The retrieved documents.

<a id="filter_retriever"></a>

## Module filter\_retriever

<a id="filter_retriever.FilterRetriever"></a>

### FilterRetriever

Retrieves documents that match the provided filters.

### Usage example

```python
from haystack import Document
from haystack.components.retrievers import FilterRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

docs = [
    Document(content="Python is a popular programming language", meta={"lang": "en"}),
    Document(content="python ist eine beliebte Programmiersprache", meta={"lang": "de"}),
]

doc_store = InMemoryDocumentStore()
doc_store.write_documents(docs)
retriever = FilterRetriever(doc_store, filters={"field": "lang", "operator": "==", "value": "en"})

# if passed in the run method, filters override those provided at initialization
result = retriever.run(filters={"field": "lang", "operator": "==", "value": "de"})

print(result["documents"])
```

<a id="filter_retriever.FilterRetriever.__init__"></a>

#### FilterRetriever.\_\_init\_\_

```python
def __init__(document_store: DocumentStore,
             filters: Optional[dict[str, Any]] = None)
```

Create the FilterRetriever component.

**Arguments**:

- `document_store`: An instance of a Document Store to use with the Retriever.
- `filters`: A dictionary with filters to narrow down the search space.

<a id="filter_retriever.FilterRetriever.to_dict"></a>

#### FilterRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="filter_retriever.FilterRetriever.from_dict"></a>

#### FilterRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "FilterRetriever"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="filter_retriever.FilterRetriever.run"></a>

#### FilterRetriever.run

```python
@component.output_types(documents=list[Document])
def run(filters: Optional[dict[str, Any]] = None)
```

Run the FilterRetriever on the given input data.

**Arguments**:

- `filters`: A dictionary with filters to narrow down the search space.
If not specified, the FilterRetriever uses the values provided at initialization.

**Returns**:

A list of retrieved documents.

<a id="sentence_window_retriever"></a>

## Module sentence\_window\_retriever

<a id="sentence_window_retriever.SentenceWindowRetriever"></a>

### SentenceWindowRetriever

Retrieves neighboring documents from a DocumentStore to provide context for query results.

This component is intended to be used after a Retriever (e.g., BM25Retriever, EmbeddingRetriever).
It enhances retrieved results by fetching adjacent document chunks to give
additional context for the user.

The documents must include metadata indicating their origin and position:
- `source_id` is used to group sentence chunks belonging to the same original document.
- `split_id` represents the position/order of the chunk within the document.

The number of adjacent documents to include on each side of the retrieved document can be configured using the
`window_size` parameter. You can also specify which metadata fields to use for source and split ID
via `source_id_meta_field` and `split_id_meta_field`.

The SentenceWindowRetriever is compatible with the following DocumentStores:
- [Astra](https://docs.haystack.deepset.ai/docs/astradocumentstore)
- [Elasticsearch](https://docs.haystack.deepset.ai/docs/elasticsearch-document-store)
- [OpenSearch](https://docs.haystack.deepset.ai/docs/opensearch-document-store)
- [Pgvector](https://docs.haystack.deepset.ai/docs/pgvectordocumentstore)
- [Pinecone](https://docs.haystack.deepset.ai/docs/pinecone-document-store)
- [Qdrant](https://docs.haystack.deepset.ai/docs/qdrant-document-store)

### Usage example

```python
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.retrievers import SentenceWindowRetriever
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore

splitter = DocumentSplitter(split_length=10, split_overlap=5, split_by="word")
text = (
        "This is a text with some words. There is a second sentence. And there is also a third sentence. "
        "It also contains a fourth sentence. And a fifth sentence. And a sixth sentence. And a seventh sentence"
)
doc = Document(content=text)
docs = splitter.run([doc])
doc_store = InMemoryDocumentStore()
doc_store.write_documents(docs["documents"])


rag = Pipeline()
rag.add_component("bm25_retriever", InMemoryBM25Retriever(doc_store, top_k=1))
rag.add_component("sentence_window_retriever", SentenceWindowRetriever(document_store=doc_store, window_size=2))
rag.connect("bm25_retriever", "sentence_window_retriever")

rag.run({'bm25_retriever': {"query":"third"}})

>> {'sentence_window_retriever': {'context_windows': ['some words. There is a second sentence.
>> And there is also a third sentence. It also contains a fourth sentence. And a fifth sentence. And a sixth
>> sentence. And a'], 'context_documents': [[Document(id=..., content: 'some words. There is a second sentence.
>> And there is ', meta: {'source_id': '...', 'page_number': 1, 'split_id': 1, 'split_idx_start': 20,
>> '_split_overlap': [{'doc_id': '...', 'range': (20, 43)}, {'doc_id': '...', 'range': (0, 30)}]}),
>> Document(id=..., content: 'second sentence. And there is also a third sentence. It ',
>> meta: {'source_id': '74ea87deb38012873cf8c07e...f19d01a26a098447113e1d7b83efd30c02987114', 'page_number': 1,
>> 'split_id': 2, 'split_idx_start': 43, '_split_overlap': [{'doc_id': '...', 'range': (23, 53)}, {'doc_id': '...',
>> 'range': (0, 26)}]}), Document(id=..., content: 'also a third sentence. It also contains a fourth sentence. ',
>> meta: {'source_id': '...', 'page_number': 1, 'split_id': 3, 'split_idx_start': 73, '_split_overlap':
>> [{'doc_id': '...', 'range': (30, 56)}, {'doc_id': '...', 'range': (0, 33)}]}), Document(id=..., content:
>> 'also contains a fourth sentence. And a fifth sentence. And ', meta: {'source_id': '...', 'page_number': 1,
>> 'split_id': 4, 'split_idx_start': 99, '_split_overlap': [{'doc_id': '...', 'range': (26, 59)},
>> {'doc_id': '...', 'range': (0, 26)}]}), Document(id=..., content: 'And a fifth sentence. And a sixth sentence.
>> And a ', meta: {'source_id': '...', 'page_number': 1, 'split_id': 5, 'split_idx_start': 132,
>> '_split_overlap': [{'doc_id': '...', 'range': (33, 59)}, {'doc_id': '...', 'range': (0, 24)}]})]]}}}}
```

<a id="sentence_window_retriever.SentenceWindowRetriever.__init__"></a>

#### SentenceWindowRetriever.\_\_init\_\_

```python
def __init__(document_store: DocumentStore,
             window_size: int = 3,
             *,
             source_id_meta_field: Union[str, list[str]] = "source_id",
             split_id_meta_field: str = "split_id",
             raise_on_missing_meta_fields: bool = True)
```

Creates a new SentenceWindowRetriever component.

**Arguments**:

- `document_store`: The Document Store to retrieve the surrounding documents from.
- `window_size`: The number of documents to retrieve before and after the relevant one.
For example, `window_size: 2` fetches 2 preceding and 2 following documents.
- `source_id_meta_field`: The metadata field that contains the source ID of the document.
This can be a single field or a list of fields. If multiple fields are provided, the retriever will
consider the document as part of the same source if all the fields match.
- `split_id_meta_field`: The metadata field that contains the split ID of the document.
- `raise_on_missing_meta_fields`: If True, raises an error if the documents do not contain the required
metadata fields. If False, it will skip retrieving the context for documents that are missing
the required metadata fields, but will still include the original document in the results.

<a id="sentence_window_retriever.SentenceWindowRetriever.merge_documents_text"></a>

#### SentenceWindowRetriever.merge\_documents\_text

```python
@staticmethod
def merge_documents_text(documents: list[Document]) -> str
```

Merge a list of document text into a single string.

This functions concatenates the textual content of a list of documents into a single string, eliminating any
overlapping content.

**Arguments**:

- `documents`: List of Documents to merge.

<a id="sentence_window_retriever.SentenceWindowRetriever.to_dict"></a>

#### SentenceWindowRetriever.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="sentence_window_retriever.SentenceWindowRetriever.from_dict"></a>

#### SentenceWindowRetriever.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "SentenceWindowRetriever"
```

Deserializes the component from a dictionary.

**Returns**:

Deserialized component.

<a id="sentence_window_retriever.SentenceWindowRetriever.run"></a>

#### SentenceWindowRetriever.run

```python
@component.output_types(context_windows=list[str],
                        context_documents=list[Document])
def run(retrieved_documents: list[Document],
        window_size: Optional[int] = None)
```

Based on the `source_id` and on the `doc.meta['split_id']` get surrounding documents from the document store.

Implements the logic behind the sentence-window technique, retrieving the surrounding documents of a given
document from the document store.

**Arguments**:

- `retrieved_documents`: List of retrieved documents from the previous retriever.
- `window_size`: The number of documents to retrieve before and after the relevant one. This will overwrite
the `window_size` parameter set in the constructor.

**Returns**:

A dictionary with the following keys:
- `context_windows`: A list of strings, where each string represents the concatenated text from the
                     context window of the corresponding document in `retrieved_documents`.
- `context_documents`: A list `Document` objects, containing the retrieved documents plus the context
                      document surrounding them. The documents are sorted by the `split_idx_start`
                      meta field.

<a id="sentence_window_retriever.SentenceWindowRetriever.run_async"></a>

#### SentenceWindowRetriever.run\_async

```python
@component.output_types(context_windows=list[str],
                        context_documents=list[Document])
async def run_async(retrieved_documents: list[Document],
                    window_size: Optional[int] = None)
```

Based on the `source_id` and on the `doc.meta['split_id']` get surrounding documents from the document store.

Implements the logic behind the sentence-window technique, retrieving the surrounding documents of a given
document from the document store.

**Arguments**:

- `retrieved_documents`: List of retrieved documents from the previous retriever.
- `window_size`: The number of documents to retrieve before and after the relevant one. This will overwrite
the `window_size` parameter set in the constructor.

**Returns**:

A dictionary with the following keys:
- `context_windows`: A list of strings, where each string represents the concatenated text from the
                     context window of the corresponding document in `retrieved_documents`.
- `context_documents`: A list `Document` objects, containing the retrieved documents plus the context
                      document surrounding them. The documents are sorted by the `split_idx_start`
                      meta field.

