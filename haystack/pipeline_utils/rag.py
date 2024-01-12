import re
from abc import ABC, abstractmethod
from typing import Optional, Any

from huggingface_hub import HfApi

from haystack import Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator, HuggingFaceTGIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import Answer
from haystack.document_stores.types import DocumentStore
from haystack.document_stores.in_memory import InMemoryDocumentStore


def build_rag_pipeline(
    document_store: DocumentStore,
    embedding_model: str = "intfloat/e5-base-v2",
    generation_model: str = "gpt-3.5-turbo",
    llm_api_key: Optional[str] = None,
    prompt_template: Optional[str] = None,
):
    """
    Returns a prebuilt pipeline to perform retrieval augmented generation
    :param document_store: An instance of a DocumentStore to read from.
    :param embedding_model: The name of the model to use for embedding. Only SentenceTransformer models supported in this getting started code.
    :param prompt_template: The template to use for the prompt. If not given, a default RAG template is used.
    :param generation_model: The name of the model to use for generation.
                             Currently supporting: OpenAI generation models and Huggingface TGI models for text generation
    :param llm_api_key: The API key to use for the OpenAI Language Model. If not given, the value of the
                        llm_api_key will be attempted to be read from the environment variable OPENAI_API_KEY.
    """
    # Resolve components based on the chosen parameters
    retriever = resolve_retriever(document_store)
    embedder = resolve_embedder(embedding_model)
    generator = resolve_generator(generation_model, llm_api_key)
    prompt_template = resolve_prompt_template(prompt_template)

    # Add them to the Pipeline and connect them
    pipeline = _RAGPipeline(
        retriever=retriever, embedder=embedder, generator=generator, prompt_template=prompt_template
    )
    return pipeline


class _RAGPipeline:
    """
    A simple ready-made pipeline for RAG. It requires a populated document store.
    """

    def __init__(self, retriever: Any, embedder: Any, generator: Any, prompt_template: str):
        """
        Initializes the pipeline.
        :param retriever: The retriever to use.
        :param embedder: The embedder to use.
        :param generator: The generator to use.
        :param prompt_template: The template to use for the prompt.
        """
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
        self.pipeline.connect("llm.meta", "answer_builder.meta")
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


def resolve_embedder(embedding_model: str) -> SentenceTransformersTextEmbedder:
    """
    Resolves the embedder
    :param embedding_model: The embedding model to use.
    """
    try:
        embedder = SentenceTransformersTextEmbedder(model=embedding_model)
    except Exception:
        raise ValueError(
            f"Embedding model: {embedding_model} is not supported. Please provide a SentenceTransformers model."
            f"You can download the models through the huggingface model hub here: https://huggingface.co/sentence-transformers"
        )
    return embedder


def resolve_retriever(document_store, retriever_class: Optional[str] = None) -> Optional[Any]:
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

    retriever = retriever_clazz(document_store=document_store)  # type: ignore
    return retriever


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


def resolve_prompt_template(prompt_template: Optional[str]) -> str:
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
    return prompt_template


class _GeneratorResolver(ABC):
    _resolvers = []  # type: ignore

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
    Resolves the OpenAIGenerator.
    """

    def resolve(self, model_key: str, api_key: str) -> Any:
        # does the model_key match the pattern OpenAI GPT pattern?
        if re.match(r"^gpt-4-.*", model_key) or re.match(r"^gpt-3.5-.*", model_key):
            return OpenAIGenerator(model=model_key, api_key=api_key)
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
