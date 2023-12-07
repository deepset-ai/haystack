import re
from abc import ABC, abstractmethod
from typing import Optional, Any
from huggingface_hub import HfApi
from haystack import Pipeline
from haystack.dataclasses import Answer
from haystack.document_stores import InMemoryDocumentStore, DocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import GPTGenerator, HuggingFaceTGIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder


def build_rag_pipeline(
    document_store: DocumentStore,
    generation_model: str = "gpt-3.5-turbo",
    prompt_template: Optional[str] = None,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    llm_api_key: Optional[str] = None,
    retriever_class: Optional[Any] = None,
):
    """
    Returns a prebuilt pipeline to perform retrieval augmented generation with or without an embedding model
    (without embeddings, it performs retrieval using BM25).

    Example usage:

    ```python
    from haystack.utils import build_rag_pipeline
    pipeline = build_rag_pipeline(document_store=your_document_store_instance)
    pipeline.run(query="What's the capital of France?")

    >>> Answer(data="The capital of France is Paris.")
    ```

    :param document_store: An instance of a DocumentStore to read from.
    :param generation_model: The name of the model to use for generation.
    :param prompt_template: The template to use for the prompt. If not given, a default template is used.
    :param embedding_model: The name of the model to use for embedding. If not given, BM25 is used.
    :param llm_api_key: The API key to use for the OpenAI Language Model. If not given, the value of the
    :param retriever_class: The retriever class to use. If not given, it will be inferred from the document store.
    llm_api_key will be attempted to be read from the environment variable OPENAI_API_KEY.
    """
    # create the retriever
    retriever_clazz = resolve_retriever(document_store, retriever_class)
    retriever = retriever_clazz(document_store=document_store)

    # embedding model must be given, only sentence-transformers models are supported (for now)
    # create the embedder
    if not embedding_model:
        raise ValueError("Embedding model must be given. Currently, only sentence-transformers models are supported.")
    embedder = SentenceTransformersTextEmbedder(model_name_or_path=embedding_model)

    # create the generator
    generator = resolve_generator(generation_model, llm_api_key)

    # create the internal pipeline instance
    # will be deprecated in the near future
    return _RAGPipeline(retriever=retriever, embedder=embedder, generator=generator, prompt_template=prompt_template)


def resolve_retriever(document_store: DocumentStore, retriever_class: Optional[str] = None) -> Optional[Any]:
    """
    Resolves the retriever class to use for the given document store.
    :param document_store: The document store to use.
    :param retriever_class: The retriever class to use. If not given, it will be inferred from the document store.
    """
    # first match the document store to the retriever
    # TODO: add more retrievers
    embedding_retriever_map = {InMemoryDocumentStore: InMemoryEmbeddingRetriever}

    retriever_clazz = (
        retriever_class or embedding_retriever_map[type(document_store)]
        if type(document_store) in embedding_retriever_map
        else None
    )
    if not retriever_clazz:
        raise ValueError(
            f"Document store {type(document_store)} is not supported. Please provide a retriever class or use "
            f"one of the following document stores: {list(embedding_retriever_map.keys())}"
        )
    return retriever_clazz


def resolve_generator(generation_model: str, llm_api_key: Optional[str] = None) -> Optional[Any]:
    """
    Resolves the generator to use for the given generation model.
    :param generation_model: The generation model to use.
    :param llm_api_key: The API key to use for the language model.
    """
    generator = None
    for resolver_clazz in _GeneratorResolver.get_resolvers():
        resolver = resolver_clazz()
        generator = resolver.resolve(generation_model, llm_api_key)
        if generator:
            break
    if not generator:
        raise ValueError(f"Could not resolve LLM generator for the given model {generation_model}")
    return generator


class _GeneratorResolver(ABC):
    _resolvers = []  # track of all resolvers

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _GeneratorResolver._resolvers.append(cls)

    @abstractmethod
    def resolve(self, model_key: str, api_key: str) -> Any:
        pass

    @classmethod
    def get_resolvers(cls):
        return cls._resolvers


class _OpenAIResolved(_GeneratorResolver):
    """
    Resolves the OpenAI GPTGenerator.
    """

    def resolve(self, model_key: str, api_key: str) -> Any:
        # does the model_key match the pattern OpenAI GPT pattern?
        if re.match(r"^gpt-4-.*", model_key) or re.match(r"^gpt-3.5-.*", model_key):
            return GPTGenerator(model_name=model_key, api_key=api_key)
        return None


class _HuggingFaceTGIGeneratorResolved(_GeneratorResolver):
    """
    Resolves the HuggingFaceTGIGenerator.
    """

    def resolve(self, model_key: str, api_key: str) -> Any:
        hf = HfApi()
        try:
            hf.model_info(model_key)
            return HuggingFaceTGIGenerator(model=model_key, token=api_key, generation_kwargs={"max_new_tokens": 1024})
        except Exception:
            return None


class _RAGPipeline:
    """
    A simple ready-made pipeline for RAG. It requires a populated document store.

    If an embedding model is given, it uses embedding retrieval. Otherwise, it falls back to BM25 retrieval.

    Example usage:

    ```python
    rag_pipe = RAGPipeline(document_store=InMemoryDocumentStore())
    answers = rag_pipe.run(query="Who lives in Rome?")
    >>> Answer(data="Giorgio")
    ```

    """

    def __init__(self, retriever: Any, embedder: Any, generator: Any, prompt_template: Optional[str] = None):
        """
        Initializes the pipeline.
        :param retriever: The retriever to use.
        :param embedder: The embedder to use.
        :param generator: The generator to use.
        :param prompt_template: The template to use for the prompt. If not given, a default template is used.
        """
        prompt_template = (
            prompt_template
            or """
        Given these documents, answer the question.

        Documents:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}

        Question: {{question}}

        Answer:
        """
        )
        self.pipeline = Pipeline()
        self.pipeline.add_component(instance=embedder, name="text_embedder")
        self.pipeline.add_component(instance=retriever, name="retriever")
        self.pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
        self.pipeline.add_component(instance=generator, name="llm")
        self.pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
        self.pipeline.connect("text_embedder", "retriever")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "llm.prompt")
        self.pipeline.connect("llm.replies", "answer_builder.replies")
        self.pipeline.connect("llm.metadata", "answer_builder.metadata")
        self.pipeline.connect("retriever", "answer_builder.documents")

    def run(self, query: str) -> Answer:
        """
        Performs RAG using the given query.

        :param query: The query to ask.
        :return: An Answer object.
        """
        run_values = {
            "prompt_builder": {"question": query},
            "answer_builder": {"query": query},
            "text_embedder": {"text": query},
        }
        return self.pipeline.run(run_values)["answer_builder"]["answers"][0]
