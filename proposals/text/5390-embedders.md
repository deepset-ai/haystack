- Title: Embedders
- Decision driver: @anakin87
- Start Date: 2023-07-19
- Proposal PR: https://github.com/deepset-ai/haystack/pull/5390

# Summary

As decided in the previous proposals ([Embedding Retriever](3558-embedding_retriever.md) and [DocumentStores and Retrievers](4370-document_stores_and_retrievers.md)), in Haystack V2 we want to introduce a new component: the Embedder.

**Separation of concerns**
- DocumentStores: store the Documents, their metadata and representations (vectors); they offer a CRUD API.
- Retrievers: retrieve Documents from the DocumentStores; they are specific and aware of the used Store (e.g., MemoryRetriever for the MemoryDocumentStore). They will be commonly used in query pipelines (not in indexing pipelines).
- **Embedders**: encode a list of data points (strings, images, etc.) into a list of vectors (i.e., the embeddings) using a model. They are used both in indexing pipelines (to encode the Documents) and query pipelines (to encode the query).

*In the current implementation, the Embedder is part of Retriever, which is unintuitive and comes with several disadvantages (explained in the previous proposals).*

**This proposal aims to define the Embedder design.**

# Basic example

*This code snippet is merely an example and may not be completely up-to-date.*


```python
from haystack import Pipeline
from haystack.components import (
    TxtConverter,
    PreProcessor,
    DocumentWriter,
    OpenAITextEmbedder,
    OpenAIDocumentEmbedder,
    MemoryRetriever,
    Reader,
)
from haystack.document_stores import MemoryDocumentStore
docstore = MemoryDocumentStore()

indexing_pipe = Pipeline()
indexing_pipe.add_store("document_store", docstore)
indexing_pipe.add_node("txt_converter", TxtConverter())
indexing_pipe.add_node("preprocessor", PreProcessor())
indexing_pipe.add_node("embedder", OpenAIDocumentEmbedder(model_name="text-embedding-ada-002"))
indexing_pipe.add_node("writer", DocumentWriter(store="document_store"))
indexing_pipe.connect("txt_converter", "preprocessor")
indexing_pipe.connect("preprocessor", "embedder")
indexing_pipe.connect("embedder", "writer")

indexing_pipe.run(...)

query_pipe = Pipeline()
query_pipe.add_store("document_store", docstore)
query_pipe.add_node("embedder", OpenAITextEmbedder(model_name="text-embedding-ada-002"))
query_pipe.add_node("retriever", MemoryRetriever(store="document_store", retrieval_method="embedding"))
query_pipe.add_node("reader", Reader(model_name="deepset/model-name"))
query_pipe.connect("embedder", "retriever")
query_pipe.connect("retriever", "reader")

results = query_pipe.run(...)
```

- The `OpenAITextEmbedder` uses OpenAI models to convert a list of strings into a list of vectors. It is used in the query pipeline to embed the query.
- The `OpenAIDocumentEmbedder` uses OpenAI models to enrich a list of Documents with the corresponding vectors (stored in the `embedding` field). It is used in the indexing pipeline to embed the Documents.
- The Retriever is no longer needed in the indexing pipeline.

# Motivation

The motivations behind this change were already provided in the previous proposals ([Embedding Retriever](3558-embedding_retriever.md) and [DocumentStores and Retrievers](4370-document_stores_and_retrievers.md)). Here is a summary:
- Retrievers should't be responsible for embedding Documents.
- Currently, Retrievers have many parameters just to support and configure different underlying Encoders(≈Embedders).
- Adding support for new embedding providers or strategies is difficult. It requires changing the Retriever code.

# Detailed design

## Handle queries and Documents
This is the most critical aspect of the design.

- When embedding queries, we expect the Embedder to transform a list of string into a list of vectors.
- When embedding Documents, we expect the Embedder to enrich a list of Documents with the corresponding vectors (stored in the `embedding` field).
- For Documents, we may want to embed some meta fields in addition to the content. Therefore, the Embedder should also handle this preparation work, which involves joining all the relevant content into a string to be embedded.

**Below, I will focus on the public API. The internal implementation is discussed in [Implementation details](#implementation-details).**

```python
@component
class HFTextEmbedder:

    class Input:
        data: List[str]

    class Output:
        embeddings: List[np.ndarray]

    ...

    def run(self, data):
        return self.output(self.embedding_service.embed(data.data))


@component
class HFDocumentEmbedder(HFTextEmbedder):

    class Input:
        documents: List[Document]

    class Output:
        documents: List[np.ndarray]

    def run(self, data):
        text_strings = [document.content for document in data.documents]
        embeddings = self.embedding_service.embed(text_strings)
        documents_with_embeddings = [Document.from_dict(**doc.to_dict, "embedding": emb) for doc, emb in zip(documents, embeddings)]
        return self.output(documents = documents_with_embeddings)
```

## Different providers/strategies

- We can define different classes depending on the providers: `OpenAIEmbedder`, `CohereEmbedder`, `HuggingFaceEmbedder`, `SentenceTransformersEmbedder`, etc.
- Additionally, we could define different classes depending on the embedding strategy if necessary.
While this is not a prominent use case, there are scenarios where [new strategies](https://github.com/deepset-ai/haystack/issues/5242) are introduced, requiring different libraries (`InstructorEmbedder`) or involving a different string preparation (`E5Embedder`). Supporting these scenarios with minimal effort would be nice.

## Different models in the same embedding/retrieval task

As you can observe from the [current implementation](https://github.com/deepset-ai/haystack/blob/main/haystack/nodes/retriever/dense.py), some embedding/retrieval tasks require the usage of different models.

This is not the most popular approach today, compared to what we call Embedding Retrieval (based on a single model). But it still has some relevant applications.

Some examples:
- In Dense Passage Retrieval, you need a model to encode queries and another model to encode Documents
- in the TableTextRetriever, we use 3 different models: one for queries, one for textual passages and one for tables
- in Multimodal Retrieval, we can specify different models to encode queries and Documents

Since the Embedder will not be included in the Retriever, it makes sense to have different Embedders, each one using a single model.

```python
dpr_query_embedder = SentenceTransformersTextEmbedder(model_name="facebook/dpr-question_encoder-single-nq-base")
dpr_doc_embedder = SentenceTransformersDocumentEmbedder(model_name="facebook/dpr-ctx_encoder-single-nq-base")
```

## Implementation details

*You can skip this section if you are primarily interested in user experience.*

There have been much discussion on how to effectively implement this proposal.
The most important aspects to consider:
- we want different Embedders for queries and Documents as they require a different treatment
- if the same model is internally used for different Embedders, we want to reuse the same instance in order to save memory

@ZanSara formulated the following implementation idea, that I like.

We make three classes:

- An `EmbeddingService`, which is NOT a component, handling raw data + a factory method to reuse instances. It will live in a different package, **is for internal use only and will never be user-facing.** (The name of this class can be changed; feel free to propose better alternatives)
```python
class HFEmbeddingService:
    """
    NOT A COMPONENT!
    """

    instances: List[HFEmbeddingService] = []

    def __new__(cls, *args, **kwargs):
		"""
        Factory method.
		If an instance with the same identical params was already created,
        return that instance instead of initializing a new one.
        """
        if <hash of name and init params> in HFEmbeddingService.instances:
          return HFEmbeddingService.instances[<hash of name and init params>]

        embedding_service = cls(*args, **kwargs)
        HFEmbeddingService.instances[<hash of name and init params>] = embedding_service
        return embedding_service

    def __init__(self, model_name: str, ... init params ...):
        """
        init takes the minimum parameters needed at init time, not
        the params needed at inference, so they're easier to reuse.
        """
        self.model = ...

    def embed(self, data: str, ... inference params ... ) -> np.ndarray:
        # compute embedding
        return embedding


class OpenAIEmbeddingService:
    ... same as above ...
```

Given that EmbeddingServices are created through a factory method, when you request an instance, if another identical exists, the method returns that instance instead of a new one.

This makes model reusability automatic in all cases, which can save lots of memory without asking the user to think about it.

- A `(Text/Table/Image/Audio)Embedder` component that does nothing but “wrapping” an EmbeddingService.
**Part of the public API**.
```python
@component
class HFTextEmbedder:

    class Input:
        data: List[str]

    class Output:
        embeddings: List[np.ndarray]

    def __init__(self, model_name: str, ... init params ...):
        self.model_name = model_name
        self.model_params = ... params ...

    def warm_up(self):
        self.embedding_service = HFEmbeddingService(self.model_name, **self.model_params)

    def run(self, data):
        return self.output(self.embedding_service.embed(data.data))
```

- A DocumentEmbedder component that handles the documents and the offloads the computation to an embedder.
**Part of the public API**.

```python
@component
class HFDocumentEmbedder(HFTextEmbedder):
  """
  Note: in this toy example inheritance from HFTextEmbedder makes sense because
  init and warm_up are identical. If they would differ significantly, let's remove
  the inheritance to simplify the architecture: it's not mandatory for the rest
  of the system to work.
  """

    class Input:
        documents: List[Document]

    class Output:
        documents: List[np.ndarray]

    def run(self, data):
        text_strings = [document.content for document in data.documents]
        embeddings = self.embedding_service.embed(text_strings)
        documents_with_embeddings = [Document.from_dict(**doc.to_dict, "embedding": emb) for doc, emb in zip(documents, embeddings)]
        return self.output(documents = documents_with_embeddings)
```

# Drawbacks

The drawbacks of concept separation between Retrievers and Embedders were discussed in [this proposal](https://github.com/deepset-ai/haystack/blob/main/proposals/text/4370-documentstores-and-retrievers.md) and mainly consist of **migration effort**.

Several ideas were discussed (mostly concerning how to handle queries and Documents in the Embedders) and this seems to be the best approach. However, there are some potential cons to consider:
  - Proliferation of classes (though they will be small and easy to maintain).
  - Users need to properly understand which models are appropriate for a task or for embedding queries vs. Documents (see [Different models in the same embedding/retrieval task](#different-models-in-the-same-embeddingretrieval-task)). On the other hand, this can help raise their awareness.

# Alternatives

Several alternatives to this design were considered. The main challenge was handling the differences between queries and Documents.
Some ideas:
- Have a single Embedder component for text (HFTextEmbedder instead of HFEmbeddingService, HFTextEmbedder and HFDocumentEmbedder) and adapt Documents before and after that, using other Components. --> Many components.
- Make Embedders only work on Documents and represent the query as a Document. --> Unintuitive and require changes in the Retriever.
- Create another primitive like Data (content + embedding) and use it for both queries and Documents. --> More conversion components like DataToDocument.
- Have the DocumentEmbedder take a TextEmbedder as an input parameter. --> Fewer classes but serialization issues.

# Adoption strategy

This change will constitute a part of Haystack v2.

# How we teach this

Documentation and tutorials will be of fundamental importance.

# Unresolved questions

- Migration and refactoring of existing Encoders hidden in Retrievers.
I prepared a table. Should it be shared here?
- The migration and refactoring of TableTextRetriever require input and ownership from people involved in TableQA.
- How to approach MultiModal Embedding? How many classes? Take into consideration that a query could also be an Image or a Table.
