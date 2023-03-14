- Title: DocumentStores and Retrievers
- Decision driver: @ZanSara
- Start Date: 2023-03-09
- Proposal PR: 4370
- Github Issue or Discussion: (only if available, link the original request for this change)

# Summary

Haystack's Document Stores are a very central component in Haystack and, as the name suggest, they were initially designed around the concept of `Document`.

As the framework grew, so did the number of Document Stores and their API, until the point where keeping them aligned aligned on the same feature set started to become a serious challenge.

In this proposal we outline a reviewed design of the same concept.

Note: these stores are designed to work **only** alongside Haystack 2.0 Pipelines (see https://github.com/deepset-ai/haystack/pull/4284)

# Motivation

Current `DocumentStore` face several issues mostly due to their organic growth. Some of them are:

- `DocumentStore`s perform the bulk of retrieval, but they need to be tighly coupled to a `Retriever` object to work. We believe this coupling can be broken by a clear API boundary between `DocumentStores`, `Retriever`s and `Embedder`s. In this PR we focus on decoupling them.

- `DocumentStore`s tend to bring in complex dependencies, so less used stores should be easy to decouple into external packages at need.

# Basic example

Stores will have to follow a contract rather than subclassing a base class. We define a contract for `DocumentStore` that defines a very simple CRUD API for Documents. Then, we provide one implementation for each underlying technology (`MemoryDocumentStore`, `ElasticsearchDocumentStore`, `FaissDocumentStore`) that respects such contract.

Once stores are defined, we will create one `Retriever` for each `DocumentStore`. Such retrievers are going to be highly specialized nodes that expect one specific document store and can handle all its specific requirements without being bound to a generic interface.

For example, this is how embedding-based retrieval would look like:

```python
from haystack import Pipeline
from haystack.nodes import (
    TxtConverter,
    PreProcessor,
    DocumentWriter,
    DocumentEmbedder,
    StringEmbedder,
    MemoryRetriever,
    Reader,
)
from haystack.document_stores import MemoryDocumentStore

docstore = MemoryDocumentStore()

indexing_pipe = Pipeline()
indexing_pipe.add_store("document_store", docstore)
indexing_pipe.add_node("txt_converter", TxtConverter())
indexing_pipe.add_node("preprocessor", PreProcessor())
indexing_pipe.add_node("embedder", DocumentEmbedder(model_name="deepset/model-name"))
indexing_pipe.add_node("writer", DocumentWriter(store="document_store"))
query_pipe.connect("txt_converter", "preprocessor")
query_pipe.connect("preprocessor", "embedder")
query_pipe.connect("embedder", "writer")

indexing_pipe.run(...)

query_pipe = Pipeline()
query_pipe.add_store("document_store", docstore)
query_pipe.add_node("embedder", StringEmbedder(model_name="deepset/model-name"))
query_pipe.add_node("retriever", MemoryRetriever(store="document_store", retrieval_method="embedding"))
query_pipe.add_node("reader", Reader(model_name="deepset/model-name"))
query_pipe.connect("embedder", "retriever")
query_pipe.connect("retriever", "reader")

results = query_pipe.run(...)
```

Note a few key differences with the existing Haystack process:

- During indexing we do not use any `Retriever`, but rather a `DocumentEmbedder`. This class accepts a model name and simply adds embeddings to the `Document`s it receives.

- We used an explicit `DocumentWriter` node instead of adding the `DocumentStore` at the end of the pipeline. That node will be generic for any document store, because the `DocumentStore` contract declares a `write_documents` method (see "Detailed Design").

- During query, the first step is not a `Retriever` anymore, but a `StringEmbedder`. Such node will convert the query into its embedding representation and forward it over to a `Retriever` that expects it. In this case, an imaginary `MemoryRetriever` can be configured to expect an embedding by setting the `retrieval_method` flag to `embedding`.



Note how we are using `MemoryRetriever`, which for example might accept a `retrieval_method` parameter to select between BM25 and embedding-based retrieval. Other document stores might support that, for example Weaviate, while others might not, like FAISS. this distinction will be evident in the parameters of their respective Retrievers.

With respect to embedding retrieval, the process is slightly different


# Detailed design

## Design of the `Data` hierarchy

The design for the Data subclasses is fairly straigthforward and consists mostly of very small, immutable dataclasses. Here we propose an implementation example with `Data`, four content type variants, `Document` and its content types variants again.

```python
from typing import List, Any, Dict, Literal
from math import inf
from pathlib import Path
import logging
import json
from dataclasses import asdict, dataclass, field
import mmh3

#: List of all `content_type` supported
ContentTypes = Literal["text", "table", "image", "audio"]

@dataclass(frozen=True, kw_only=True, eq=True)
class Data:

    id: str = field(default_factory=str)
    content: Any = field(default_factory=lambda: None)
    # The content_type field makes it much simpler to deserialize into the correct class
    content_type: ContentTypes
    meta: Dict[str, Any] = field(default_factory=dict, hash=False)
    id_hash_keys: List[str] = field(default_factory=lambda: ["content"], hash=False)

    def __str__(self):
        return f"{self.__class__.__name__}('{self.content}')"

    def to_dict(self):
        return asdict(self)

    def to_json(self, **json_kwargs):
        return json.dumps(self.to_dict(), *json_kwargs)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)

    @classmethod
    def from_json(cls, data, **json_kwargs):
        dictionary = json.loads(data, **json_kwargs)
        return cls.from_dict(dictionary=dictionary)

@dataclass(frozen=True, kw_only=True)
class Text(Data):
    content: str
    content_type: ContentTypes = "text"

@dataclass(frozen=True, kw_only=True)
class Table(Data):
    content: 'pd.DataFrame'
    content_type: ContentTypes = "table"

@dataclass(frozen=True, kw_only=True)
class Image(Data):
    content: Path
    content_type: ContentTypes = "image"

@dataclass(frozen=True, kw_only=True)
class Audio(Data):
    content: Path
    content_type: ContentTypes = "audio"


@dataclass(frozen=True, kw_only=True)
class Document(Data):

    score: Optional[float] = None
    embedding: Optional[np.ndarray] = field(default=lambda:None, repr=False)

    def __lt__(self, other):
        if not hasattr(other, "score"):
            raise ValueError("Documents can only be compared with other Documents.")
        return (self.score if self.score is not None else -inf) < (
            other.score if other.score is not None else -inf
        )

    def __le__(self, other):
        if not hasattr(other, "score"):
            raise ValueError("Documents can only be compared with other Documents.")
        return (self.score if self.score is not None else -inf) <= (
            other.score if other.score is not None else -inf
        )

@dataclass(frozen=True, kw_only=True)
class TextDocument(Text, Document):
    pass

@dataclass(frozen=True, kw_only=True)
class TableDocument(Table, Document):
    pass

@dataclass(frozen=True, kw_only=True)
class ImageDocument(Image, Document):
    pass

@dataclass(frozen=True, kw_only=True)
class AudioDocument(Audio, Document):
    pass
```

### Foreseen subclasses

From the point of view of modality, for now we foresee 4 content types: `text`, `table`, `image` and `audio`.

From the point of view of semantics, the rule of thumb is: "if it could ever make sense to assign metadata or embeddings to it, it can be a data class". As a consequence, we shortlisted the following: `Document`, `Answer`, `Label`, `MultiLabel` (to be evaluated) and `Query` (to be evaluated). Note the lack of `Span`, also to be discussed.

Note for the reviewers: this is clearly an early attempt. This list is not binding. Making stores unaware of which dataclass they're storing makes it way easier to add/remove dataclasses later, so we should not focus too much on this list right now and rather iterate before Haystack v2.0.

## Design of the `Store` hierarchy

`Store`s are a bit more complex as they have to be disentangled from `Retriever`s.

First, let's define the API of a basic `Store`:

```python
class Store(ABC):

    def __init__(self):
        pass

    def has_item(self, id: str) -> bool:
        pass

    def get_item(self, id: str) -> Dict[str, Any]:
        pass

    def count_items(self, filters: Dict[str, Any]) -> int:
        pass

    def get_ids(self, filters: Dict[str, Any]) -> List[str]:
        pass

    def get_items(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass

    def write_items(self, items: Iterable[Dict[str, Any]], duplicates: Literal["skip", "overwrite", "fail"]) -> None:
        pass

    def delete_items(self, ids: List[str], fail_on_missing_item: bool = False) -> None:
        pass
```

As you can see, many concepts were kept from old `DocumentStore`s, with a few notable exceptions:

- There's no more `count_documents`, `get_documents`, etc, but only `count_items`, `get_item`. This store is generic and can contain any type of data. Therefore, we can store `Document`s separately from `Label`s and we don't need to care about supporting both: if a generic `Store` exists for that backend, it can automatically store both `Document`s and `Label`s by adding a very thin layer on top (see below).

- No mention of retrieval other than `get_items`, which only accepts `filters`. That's because `get_items` is NOT supposed to be used for retrieval but only, when needed, for filtering.

- We removed the concept of `index`. Stores are not a monolithic representation of any specific (vector) database: if users want to leverage ES' indices or Pinecone's Namespaces, they can create a Store for each of them. We want to keep the Store's API as minimal and generic as possible, to allow agnostic nodes to use the stores intechangeably.

Let's now assume we create a `StoreInMemory` that implements the methods above. From such class we could then create a wrapper called `DocumentStoreInMemory`, which would look like this:

```python
class DocumentStoreInMemory:

    def __init__(self, ...params...):
        self.store = StoreInMemory()
        self.bm25 = BM25Index()
        ...

    def has_document(self, id: str) -> bool:
        ...

    def get_document(self, id: str) -> Optional[Document]:
        item = self.store.get_item(id=id, pool=pool)
        # Deserializes it back to a Document object because we know we're storing
        # Documents in this store
        return Document.from_dict(item)

    def count_documents(self, filters: Dict[str, Any]):
        ...

    def get_document_ids(self, filters: Dict[str, Any]) -> List[str]:
        ...

    def get_documents(self, filters: Dict[str, Any]) -> List[Document]:
        items = self.store.get_items(id=id, filters=filters)
        # Deserializes it back to Document objects because we know we're storing
        # Documents in this store
        return [Document.from_dict(item) for item in items]

    def write_documents(
        self,
        documents: List[Document],
        duplicates: Literal["skip", "overwrite", "fail"] = "overwrite",
    ) -> None:
        self.store.write_items(
            # Serializes it to dictionary before writing
            items=(doc.to_dict() for doc in documents),
            duplicates=duplicates,
        )
        # Additional code to support BM25 retrieval
        if self.bm25:
            self.bm25.update_bm25(self.get_documents(filters={}))

    def delete_documents(self, ids: List[str], fail_on_missing_item: bool = False) -> None:
        ...

    def get_relevant_documents(
        self,
        queries: List[Query],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        use_bm25: bool = True,
        similarity: str = "dot_product",
        scoring_batch_size: int = 500000,
        scale_score: bool = True,
    ) -> Dict[str, List[Document]]:

        ######################
        # Performs retrieval #
        ######################

        filters = filters or {}

        # BM25 Retrieval
        if use_bm25:
            relevant_documents = {}
            for query in queries:
                relevant_documents[query] = list(self._bm25_retrieval(...))
            return relevant_documents

        # Embedding Retrieval
        relevant_documents = self._embedding_retrieval(...)
        return relevant_documents

    def _bm25_retrieval(...) -> List[Document]:
        ...

    def _embedding_retrieval(...) -> Dict[str, List[Document]]:
        ...
```

Note two important details:

- `DocumentStoreInMemory` DOES NOT inherit from `StoreInMemory`. It uses the store internally. This spares us the "signature creep" seen in many DocumentStores currently, which are being forced to have all identical signatures even in such context where it makes no sense (how all DocumentStores have to support `headers` and then just throw warnings if set). Composition in this case is an extremely valuable tool.

- On the other hand, `DocumentStoreInMemory` only adds one single (public) method to the `Store` signature: `get_relevant_documents`. This is the Retriever contract: see the next paragraph for details.

## The `Retriever`s contract

We define an explicit contract between `DocumentStore`s (not all `Store`s, just their `Document` variety) and `Retriever`s. This contract is very simple and states that:

> All document stores that support retrieval should define a `get_relevant_documents()` method.

Each `Retriever` will define a set of parameters it is going to pass to its `DocumentStore`, and all `DocumentStore`s that want to support that `Retriever` type must accept its set of parameters. Note that such signature is not enforced by any inheritance.

For example:

```python
@node
class BM25Retriever:
    """
    BM25Retriever works with any store that accepts BM25
    parameters in `get_relevant_documents`:
    - `use_bm25`
    - `bm25_parameters`
    """
    def __init__(self, ...):
        pass

    def run(self, ...):
        ...

        if not hasattr(stores[store_name], "get_relevant_documents"):
            raise ValueError(f"{store_name} is not a DocumentStore or it does not support Retrievers.")

        documents = stores["documents"].get_relevant_documents(
            queries=queries,
            filters=filters,
            top_k=top_k,
            use_bm25=True,   # This is specific to BM25Retriever
            bm25_parameters=bm25_parameters   # This is specific to BM25Retriever
        )
        ...

@node
class EmbeddingRetriever:
    """
    EmbeddingRetriever works with any store that accepts
    dense retrieval parameters in `get_relevant_documents`:
    - `use_embedding`
    - `similarity`
    - `scale_score`
    """
    def __init__(self, ...):
        pass

    def run(self, ...):
        ...
        if not hasattr(stores[store_name], "get_relevant_documents"):
            raise ValueError(f"{store_name} is not a DocumentStore or it does not support Retrievers.")

        documents = stores["documents"].get_relevant_documents(
            queries=queries,
            filters=filters,
            top_k=top_k,
            use_embedding=True, # This is specific to EmbeddingRetriever
            similarity=similarity,   # This is specific to EmbeddingRetriever
            scale_score=scale_score   # This is specific to EmbeddingRetriever
        )
        ...
```

### Expected question: "Why not one method for each retrieval type?"

It migth seem appealing to decide that for every retrieval method, DocumentStores wishing to support it should add a separate `get_relevant_documents_with_[bm25|embedding|magic|...]()`. That would allows document stores to type-check the signature and do more validation before hand.

However, I believe such design would tie too strictly `Store`s and `Retriever`s. Some stores might be able to share a lot of logic, parameters, etc... across different retrieval methods (think about `top_k`, for example), while adding a method for each technique incentivizes code repetition.

This is, however, up for debate and further evaluation.

# Drawbacks

### Migration effort

We will need to migrate all DocumentStores into their Store implementation. Although it is going to be a massive undertaking, this process will allow us to drop less used DocumentStore backends and focus on the most important ones.

An example shortlist could be:

- `StoreInMemory`
- `StoreIn(Elasticsearch|Opensearch)` (choose one)
- `StoreIn(Qdrant|Weaviate|Pinecone)` (choose one or two)
- `StoreInFAISS`
- `StoreInSQL`

all with their `DocumentStore` variant only. We will later consider additional implementations.

# Alternatives

Not really any. We could force support for the old Docstores into the new Pipelines, but I see no value in such effort given that with the same investment we can get a massively smaller codebase.

# Adoption strategy

This proposal is part of the Haystack 2.0 rollout strategy. See https://github.com/deepset-ai/haystack/pull/4284.

# How we teach this

Documentation is going to be crucial, as much as tutorials and demos. We plan to start working on those as soon as basic nodes (one reader and one retriever) are added to Haystack v2 and `DocumentStoreInMemory` receives its first implementation.

# Unresolved questions

_Waiting for review_.
