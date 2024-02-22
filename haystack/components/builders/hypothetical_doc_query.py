from haystack import Pipeline, component, Document
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders import PromptBuilder

from typing import List, Dict, Any
import numpy as np


@component
class InstructLM:
    """
    Subcomponent to generate hypothetical documents using an instruction-following language model (InstructLM).
    """

    def __init__(self, instruct_llm, nr_completions):
        self.prompt_builder = PromptBuilder(
            template="""
        Given a question, generate a paragraph of text that answers the question.
        Question: {{question}}
        Paragraph:
        """
        )
        self.generator = OpenAIGenerator(
            model=instruct_llm, generation_kwargs={"n": nr_completions, "temperature": 0.75, "max_tokens": 400}
        )

        self.pipeline = Pipeline()
        self.pipeline.add_component(name="prompt_builder", instance=self.prompt_builder)
        self.pipeline.add_component(name="generator", instance=self.generator)
        self.pipeline.connect("prompt_builder", "generator")

    def to_dict(self) -> Dict[str, Any]:
        # ToDo
        pass

    def from_dict(cls, data: Dict[str, Any]) -> "InstructLM":
        # ToDo
        pass

    @component.output_types(replies=List[str])
    def run(self, query: str):
        result = self.pipeline.run(data={"prompt_builder": {"question": query}})
        return result["generator"]


@component
class HyDEBuilder:
    """
    Subcomponent to build the Hypothetical Document Embedding (HyDE) from the generated hypothetical documents.
    """

    def __init__(self):
        self.embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2", progress_bar=False
        )
        self.embedder.warm_up()

    def to_dict(self) -> Dict[str, Any]:
        # ToDo
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyDEBuilder":
        # ToDo
        pass

    @component.output_types(embedding=List[float])
    def run(self, answers: List[str]):
        embeddings = self.embedder.run([Document(content=answer) for answer in answers])
        stacked_embeddings = np.array([doc.embedding for doc in embeddings["documents"]])
        avg_embeddings = np.mean(stacked_embeddings, axis=0)
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_documents": answers, "hypothetical_embedding": hyde_vector[0].tolist()}


@component
class HypotheticalDocumentEmbedder:
    """
    Hypothetical Document Embeddings (HyDE)

    Given a query, HyDE first zero-shot prompts an instruction-following language model to generate a "fake"
    hypothetical document but that captures relevant textual patterns.

    Then, it encodes the document into an embedding vector, identifying a neighborhood in the corpus
    embedding space, from which similar real documents are retrieved based on vector similarity.

    see:  "Precise Zero-Shot Dense Retrieval without Relevance Labels" (https://aclanthology.org/2023.acl-long.99/)

    Usage example:
    ```python
    from haystack.components.embedders import HypotheticalDocumentEmbedder
    query = "What should I see in the capital of France?"
    hyde.run(query=query)
    # [0.08679415807127952, -0.007719121221452951, 0.056458056718111035, -0.005992368655279278, ...]
    ```
    """

    def __init__(self, instruct_llm: str = "gpt-3.5-turbo", nr_completions: int = 5):
        """
        Initialize the HyDE component with a language model and the number of completions.

        :param instruct_llm: name of the language model to use for instruction following
        :param nr_completions: Number of completions to generate
        """
        self.instruct_lm = InstructLM(instruct_llm, nr_completions=nr_completions)
        self.hyde_builder = HyDEBuilder()

        # connecting the components
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="instruct_lm", instance=self.instruct_lm)
        self.pipeline.add_component(name="hyde_builder", instance=self.hyde_builder)
        self.pipeline.connect("instruct_lm.replies", "hyde_builder.answers")

    def to_dict(self) -> Dict[str, Any]:
        # ToDo
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HypotheticalDocumentEmbedder":
        # ToDo
        pass

    @component.output_types(embedding=List[float])
    def run(self, query: str):
        result = self.pipeline.run({"query": query})
        return result["hyde_builder"]["hypothetical_embedding"]
