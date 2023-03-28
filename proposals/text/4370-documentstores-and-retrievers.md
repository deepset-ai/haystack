- Title: `DocumentStores` and `Retrievers`
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
indexing_pipe.connect("txt_converter", "preprocessor")
indexing_pipe.connect("preprocessor", "embedder")
indexing_pipe.connect("embedder", "writer")

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


# Detailed design

## `DocumentStore` contract

Here is a summary of the basic contract that all `DocumentStore`s are expected to follow.

```python
class MyDocumentStore:

    def count_documents(self, **kwargs) -> int:
        ...

    def filter_documents(self, filters: Dict[str, Any], **kwargs) -> List[Document]:
        ...

    def write_documents(self, documents: List[Document], **kwargs) -> None:
        ...

    def delete_documents(self, ids: List[str], **kwargs) -> None:
        ...
```

The contract is quite narrow to encourage the use of specialized nodes. `DocumentStore`s' primary focus should be storing documents: the fact that most vector stores also support retrieval should be outside of this abstraction and made available through methods that do not belong to the contract. This allows `Retriever`s to carry out their tasks while avoiding clutter on `DocumentStore`s that do not support some features.

Note also how the concept of `index` is not present anymore, as it it mostly ES-specific.

For example, a `MemoryDocumentStore` could offer the following API:

```python
class MemoryDocumentStore:

    def filter_documents(self, filters: Dict[str, Any], **kwargs) -> List[Document]:
        ...

    def write_documents(self, documents: List[Document], **kwargs) -> None:
        ...

    def delete_documents(self, ids: List[str], **kwargs) -> None:
        ...

    def bm25_retrieval(
        self,
        queries: List[str],   # Note: takes strings!
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[List[Document]]:
        ...

    def vector_similarity_retrieval(
        self,
        queries: List[np.array],   # Note: takes embeddings!
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[List[Document]]:
        ...

    def knn_retrieval(
        self,
        queries: List[np.array],   # Note: takes embeddings!
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[List[Document]]:
        ...
```

In this way, a `DocumentWriter` could easily use the `write_documents` method defined in the contract on all document stores, while `MemoryRetriever` can leverage the fact that it only supports `MemoryDocumentStore`, so it can assume all its custom methods like `bm25_retrieval`, `vector_similarity_retrieval`, etc... are present.

Here is, for comparison, an example implementation of a `DocumentWriter`, a document-store agnostic node.

```python
@node
class DocumentWriter:

    def __init__(self, inputs=['documents'], stores=["documents"]):
        self.store_names = stores
        self.inputs = inputs
        self.outputs = []
        self.init_parameters = {"inputs": inputs, "stores": stores}

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        writer_parameters = parameters.get(name, {})
        stores = writer_parameters.pop("stores", {})

        all_documents = []
        for _, documents in data:
            all_documents += documents

        for store_name in self.store_names:
            stores[store_name].write_documents(documents=all_documents, **writer_parameters)

        return ({}, parameters)
```
This class does not check which document store it is using, because it can safely assume they are going to have a `write_documents` method.

Here instead we can see an example implementation of a `MemoryRetriever`, a document-store aware node.

```python
@node
class MemoryRetriever:

    def __init__(self, inputs=['query'], output="documents", stores=["documents"]):
        self.store_names = stores
        self.inputs = inputs
        self.outputs = [output]
        self.init_parameters = {"inputs": inputs, "output": output "stores": stores}

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:

        retriever_parameters = parameters.get(name, {})
        stores = retriever_parameters.pop("stores", {})
        retrieval_method = retriever_parameters.pop("retrieval_method", "bm25")

        for store_name in self.store_names:
            if not isinstance(stores[store_name], MemoryStore):
                raise ValueError("MemoryRetriever only works with MemoryDocumentStore.")

            if retrieval_method == "bm25":
                documents = stores[store_name].bm25_retrieval(queries=queries, **retriever_parameters)
            elif retrieval_method == "embedding":
                documents = stores[store_name].vector_similarity_retrieval(queries=queries, **retriever_parameters)
            ...

        return ({self.outputs[0]: documents}, parameters)
```

Note how `MemoryRetriever` is making use of methods that are not specified in the contract and therefore has to check that the document store it has been connected to is a proper one.

# Drawbacks

### Migration effort

We will need to migrate all `DocumentStore`s and heavily cut their API. Although it is going to be a massive undertaking, this process will allow us to drop less used `DocumentStore` backends and focus on the most important ones. It will also highly reduce the code we have to maintain.

We will also need to re-implement the ehtire Retrieval stack. We believe a lot of code could be reused, but we will focus on leveraging each document store facilities a lot more, and that will require almost complete rewriters. The upside is that the resulting code should be several times shorter, so the maintenance burden should be limited.

# Alternatives

We could force support for the old Docstores into the new Pipelines, but I see no value in such effort given that with the same investment we can get a massively smaller codebase.

# Adoption strategy

This proposal is part of the Haystack 2.0 rollout strategy. See https://github.com/deepset-ai/haystack/pull/4284.

# How we teach this

Documentation is going to be crucial, as much as tutorials and demos. We plan to start working on those as soon as basic nodes (one reader and one retriever) are added to Haystack v2 and `MemoryDocumentStore` receives its first implementation.

# Open questions

- We should enable validation of `DocumentStore`s for nodes that are document-store aware. It could be done by an additional `validation` method with relative ease, but it's currently not mentioned in the node/pipeline contract.
