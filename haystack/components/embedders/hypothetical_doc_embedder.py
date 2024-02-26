from haystack import Pipeline, component, Document, default_to_dict, default_from_dict
from haystack.components.converters import OutputAdapter
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

    Usage as a stand-alone example:
    ```python
    from haystack.components.embedders import HypotheticalDocumentEmbedder
    hyde = HypotheticalDocumentEmbedder(model="gpt-3.5-turbo", nr_completions=5)
    hyde.run(query="What should I see in the capital of France?")

    # {'hypothetical_documents': ['When visiting the capital of France, ....', '....', '...', '...', '...'],
    # 'hypothetical_embedding': [0.0990725576877594, -0.017647066991776227, 0.05918873250484467, ...]}
    ```

    Incorporating into an existing Pipeline, and assuming a DocumentStore `doc_store` is already set up:
    ```python
    retriever = InMemoryEmbeddingRetriever(document_store=doc_store)
    hyde = HypotheticalDocumentEmbedder(instruct_llm="gpt-3.5-turbo", nr_completions=5)

    extractive_qa_pipeline = Pipeline()
    extractive_qa_pipeline.add_component(instance=hyde, name="builder")
    extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
    extractive_qa_pipeline.connect("builder.hypothetical_embedding", "retriever.query_embedding")

    query = "What's the capital of France?"
    extractive_hyde_pipeline.run(data={"builder": {"query": query}, "retriever": {"top_k": 5}})

    NOTE: The hypothetical document embedder needs to use the same Embedder model as the DocumentStore used to index
    the documents, so that the embeddings are comparable in the same vector space.

    """

    def __init__(
        self,
        instruct_llm: str = "gpt-3.5-turbo",
        instruct_llm_api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
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
        self.instruct_llm_api_key = instruct_llm_api_key
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

        self.adapter = OutputAdapter(
            template="{{answers | build_doc}}",
            output_type=List[Document],
            custom_filters={"build_doc": lambda data: [Document(content=d) for d in data]},
        )

        self.embedder = SentenceTransformersDocumentEmbedder(model=embedder_model, progress_bar=False)
        self.embedder.warm_up()

        self.pipeline = Pipeline()
        self.pipeline.add_component(name="prompt_builder", instance=self.prompt_builder)
        self.pipeline.add_component(name="generator", instance=self.generator)
        self.pipeline.add_component(name="adapter", instance=self.adapter)
        self.pipeline.add_component(name="embedder", instance=self.embedder)
        self.pipeline.connect("prompt_builder", "generator")
        self.pipeline.connect("generator.replies", "adapter.answers")
        self.pipeline.connect("adapter.output", "embedder.documents")

    def to_dict(self) -> Dict[str, Any]:
        data = default_to_dict(
            self,
            instruct_llm=self.instruct_llm,
            instruct_llm_api_key=self.instruct_llm_api_key,
            nr_completions=self.nr_completions,
            embedder_model=self.embedder_model,
        )
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
        # return a single query vector embedding representing the average of the hypothetical document embeddings
        stacked_embeddings = array([doc.embedding for doc in result["embedder"]["documents"]])
        avg_embeddings = mean(stacked_embeddings, axis=0)
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_embedding": hyde_vector[0].tolist()}
