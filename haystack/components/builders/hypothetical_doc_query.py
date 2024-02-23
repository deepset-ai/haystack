from haystack import Pipeline, component, Document, default_to_dict, default_from_dict
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders import PromptBuilder

from typing import Dict, Any
from numpy import array, mean


@component
class HypotheticalDocumentEmbedder:
    """
    Hypothetical Document Embeddings (HyDE)

    Given a query, HyDE first zero-shot prompts an instruction-following language model to generate a "fake"
    hypothetical document but that captures relevant textual patterns.

    Then, it encodes the document into an embedding vector, identifying a neighborhood in the corpus
    embedding space, from which similar real documents are retrieved based on vector similarity.

    see: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (https://aclanthology.org/2023.acl-long.99/)

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
        Create a HypotheticalDocumentEmbedder component.

        :param instruct_llm: The name of the instruction-following language model to use.
        :param nr_completions: The number of completions to generate.
        """

        self.nr_completions = nr_completions
        self.generator = OpenAIGenerator(
            model=instruct_llm, generation_kwargs={"n": nr_completions, "temperature": 0.75, "max_tokens": 400}
        )
        self.prompt_builder = PromptBuilder(
            template="""
                Given a question, generate a paragraph of text that answers the question.
                Question: {{question}}
                Paragraph:
                """
        )
        self.embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2", progress_bar=False
        )
        self.embedder.warm_up()

        self.pipeline = Pipeline()
        self.pipeline.add_component(name="prompt_builder", instance=self.prompt_builder)
        self.pipeline.add_component(name="generator", instance=self.generator)
        self.pipeline.connect("prompt_builder", "generator")

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            instruct_llm=self.generator.to_dict(),
            prompt_builder=self.prompt_builder.to_dict(),
            embedder=self.embedder.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HypotheticalDocumentEmbedder":
        return default_from_dict(cls, data)

    @component.output_types(replies=Dict[str, Any])
    def run(self, query: str):
        result = self.pipeline.run(data={"prompt_builder": {"question": query}})
        answers = result["generator"]["replies"]
        embeddings = self.embedder.run([Document(content=answer) for answer in answers])
        stacked_embeddings = array([doc.embedding for doc in embeddings["documents"]])
        avg_embeddings = mean(stacked_embeddings, axis=0)
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_documents": answers, "hypothetical_embedding": hyde_vector[0].tolist()}
