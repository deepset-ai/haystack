from typing import Optional, List

from haystack.preview import Pipeline
from haystack.preview.dataclasses import Answer
from haystack.preview.document_stores import InMemoryDocumentStore
from haystack.preview.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.preview.components.embedders import SentenceTransformersTextEmbedder
from haystack.preview.components.generators import GPTGenerator
from haystack.preview.components.builders.answer_builder import AnswerBuilder
from haystack.preview.components.builders.prompt_builder import PromptBuilder


class RAGPipeline:
    """
    A simple ready-made pipeline for RAG. It requires a populated document store.

    If an embedding model is given, it uses embedding retrieval. Otherwise it falls back to BM25 retrieval.

    Example usage:

    ```python
    rag_pipe = RAGPipeline(document_store=InMemoryDocumentStore())
    answers = rag_pipe.run(query="Who lives in Rome?")
    >>> Answer(data="Giorgio")
    ```

    """

    def __init__(
        self,
        document_store: InMemoryDocumentStore,
        generation_model: str = "gpt-3.5-turbo",
        prompt_template: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        :param document_store: An instance of a DocumentStore to retrieve documents from.
        :param generation_model: The name of the model to use for generation.
        :param prompt_template: The template to use for the prompt. If not given, a default template is used.
        :param embedding_model: The name of the model to use for embedding. If not given, BM25 is used.
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
        if not isinstance(document_store, InMemoryDocumentStore):
            raise ValueError("RAGPipeline only works with an InMemoryDocumentStore.")

        self.pipeline = Pipeline()

        if embedding_model:
            self.pipeline.add_component(
                instance=SentenceTransformersTextEmbedder(model_name_or_path=embedding_model), name="text_embedder"
            )
            self.pipeline.add_component(
                instance=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()), name="retriever"
            )
            self.pipeline.connect("text_embedder", "retriever")
        else:
            self.pipeline.add_component(
                instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()), name="retriever"
            )

        self.pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
        self.pipeline.add_component(instance=GPTGenerator(model_name=generation_model), name="llm")
        self.pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "llm.prompt")
        self.pipeline.connect("llm.replies", "answer_builder.replies")
        self.pipeline.connect("llm.metadata", "answer_builder.metadata")
        self.pipeline.connect("retriever", "answer_builder.documents")

    def run(self, query: str) -> Answer:
        """
        Performs RAG using the given query.

        :param query: The query to ask.
        :return: A Answer object.
        """
        run_values = {"prompt_builder": {"question": query}, "answer_builder": {"query": query}}
        if self.pipeline.graph.nodes.get("text_embedder"):
            run_values["text_embedder"] = {"query": query}
        else:
            run_values["retriever"] = {"query": query}

        return self.pipeline.run(run_values)["answer_builder"]["answers"][0]
