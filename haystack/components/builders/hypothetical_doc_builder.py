from haystack import Pipeline, component, Document, default_to_dict, default_from_dict
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders import PromptBuilder

from typing import Dict, Any, List
from numpy import array, mean

from haystack.utils import Secret


@component
class HypotheticalDocumentEmbedder:
    """
    Hypothetical Document Embeddings (HyDE)

    Given a query, HyDE first zero-shot prompts an instruction-following language model to generate a "fake"
    hypothetical document or multiple hypothetical documents that capture relevant textual patterns.

    Then, it encodes the document(s) into an embedding vector, and averages the embeddings to obtain a single vector
    identifying a neighborhood in the corpus embedding space, from which similar real documents are retrieved based
    on vector similarity.

    see: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (https://aclanthology.org/2023.acl-long.99/)

    Usage example:
    ```python
    from haystack.components.embedders import HypotheticalDocumentEmbedder
    hyde = HypotheticalDocumentEmbedder(model="gpt-3.5-turbo", nr_completions=5)
    hyde.run(query="What should I see in the capital of France?")

    # {'hypothetical_documents': ['When visiting the capital of France, ....', '....', '...', '...', '...'],
    # 'hypothetical_embedding': [0.0990725576877594, -0.017647066991776227, 0.05918873250484467, ...]}
    ```
    """

    def __init__(
        self,
        instruct_llm: str = "gpt-3.5-turbo",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        nr_completions: int = 5,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Create a HypotheticalDocumentEmbedder component.

        :param instruct_llm: The name of the OpenAIGenerator instruction-following language model to use.
        :param api_key: The OpenAI API key.
        :param nr_completions: The number of completions to generate.
        :param embedder_model: Name of the SentenceTransformers model to use for encoding the hypothetical documents.
        """
        self.instruct_llm = instruct_llm
        self.instruct_llm_api_key = api_key
        self.nr_completions = nr_completions
        self.embedder_model = embedder_model
        self.generator = OpenAIGenerator(
            api_key=self.instruct_llm_api_key,
            model=self.instruct_llm,
            generation_kwargs={"n": self.nr_completions, "temperature": 0.75, "max_tokens": 400},
        )
        self.prompt_builder = PromptBuilder(
            template="""Given a question, generate a paragraph of text that answers the question.
            Question: {{question}}
            Paragraph:
            """
        )
        self.embedder = SentenceTransformersDocumentEmbedder(model=embedder_model, progress_bar=False)
        self.embedder.warm_up()

        self.pipeline = Pipeline()
        self.pipeline.add_component(name="prompt_builder", instance=self.prompt_builder)
        self.pipeline.add_component(name="generator", instance=self.generator)
        self.pipeline.connect("prompt_builder", "generator")

    def to_dict(self) -> Dict[str, Any]:
        data = default_to_dict(self, instruct_llm=self.instruct_llm, nr_completions=self.nr_completions)
        data["pipeline"] = self.pipeline.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HypotheticalDocumentEmbedder":
        hyde_obj = default_from_dict(cls, data)
        hyde_obj.pipeline = Pipeline.from_dict(data["pipeline"])
        return hyde_obj

    @component.output_types(hypothetical_embedding=List[float])
    def run(self, query: str):
        result = self.pipeline.run(data={"prompt_builder": {"question": query}})
        answers = result["generator"]["replies"]
        # embed the hypothetical documents and average the embeddings
        embeddings = self.embedder.run([Document(content=answer) for answer in answers])
        stacked_embeddings = array([doc.embedding for doc in embeddings["documents"]])
        avg_embeddings = mean(stacked_embeddings, axis=0)
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_embedding": hyde_vector[0].tolist()}
