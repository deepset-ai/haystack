from typing import List, Any, Optional, Dict

import logging
from pprint import pprint

from haystack import Pipeline, Document, component, default_to_dict, default_from_dict, DeserializationError
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.others import Multiplexer
from haystack.components.routers.conditional_router import ConditionalRouter
from haystack.core.component.types import Variadic


logging.getLogger().setLevel(logging.DEBUG)


@component
class PaginatedRetriever:
    """
    This component is used to paginate the results of a retriever.
    It is useful when the retriever returns a large number of documents, and we want to pass them to the LLM
    in batches.

    It is useful in cases where the LLM's context length is limited, and we want to avoid passing too many
    documents to it at once.
    """

    def __init__(self, retriever: Any, page_size: int = 1, top_k: int = 100):
        self.retriever = retriever
        self.page_size = page_size
        self.top_k = top_k
        self.retrieved_documents = None

    def to_dict(self):
        return default_to_dict(self, retriever=self.retriever.to_dict(), page_size=self.page_size)

    @classmethod
    def from_dict(cls, data):
        if not "retriever" in data["init_parameters"]:
            raise DeserializationError("Missing required field 'retriever' in SlidingWindowRetriever")

        retriever_data = data["init_parameters"]["retriever"]
        if "type" not in retriever_data:
            raise DeserializationError("Missing 'type' in retriever's serialization data")
        if retriever_data["type"] not in component.registry:
            raise DeserializationError(f"Component type '{retriever_data['type']}' not found")
        retriever_class = component.registry[retriever_data["type"]]

        data["init_parameters"]["retriever"] = retriever_class.from_dict(retriever_data)
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: Variadic[str],
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        scale_score: Optional[bool] = None,
    ):
        if not top_k:
            top_k = self.top_k

        if self.retrieved_documents is None:
            self.retrieved_documents = self.retriever.run(
                query=query[0], filters=filters, top_k=top_k, scale_score=scale_score  # type: ignore
            )["documents"]

        if not self.retrieved_documents:
            raise ValueError("No more documents available :(")

        next_page = self.retrieved_documents[: self.page_size]
        self.retrieved_documents = self.retrieved_documents[self.page_size :]
        return {"documents": next_page}


def self_correcting_pipeline():
    # Create the RAG pipeline
    rag_pipeline = Pipeline(max_loops_allowed=10)
    rag_pipeline.add_component(instance=Multiplexer(str), name="query_multiplexer")
    rag_pipeline.add_component(
        instance=PaginatedRetriever(InMemoryBM25Retriever(document_store=InMemoryDocumentStore())), name="retriever"
    )
    rag_pipeline.add_component(
        instance=PromptBuilder(
            template="""
    Given these documents, answer the question.
    If the documents don't provide enough information to answer the question, answer with the string "UNKNOWN".

    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """
        ),
        name="prompt_builder",
    )
    rag_pipeline.add_component(instance=OpenAIGenerator(), name="llm")
    rag_pipeline.add_component(
        instance=ConditionalRouter(
            routes=[
                {
                    "condition": "{{ 'UNKNOWN' in replies|join(' ') }}",
                    "output": "{{ query }}",
                    "output_name": "unanswered_query",
                    "output_type": str,
                },
                {
                    "condition": "{{ 'UNKNOWN' not in replies|join(' ') }}",
                    "output": "{{ replies }}",
                    "output_name": "replies",
                    "output_type": List[str],
                },
            ]
        ),
        name="answer_checker",
    )

    rag_pipeline.connect("query_multiplexer", "retriever")
    rag_pipeline.connect("query_multiplexer", "prompt_builder.question")
    rag_pipeline.connect("query_multiplexer", "answer_checker.query")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_checker.replies")
    rag_pipeline.connect("answer_checker.unanswered_query", "query_multiplexer")

    # Draw the pipeline
    rag_pipeline.draw("self_correcting_pipeline.png")

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
        Document(content="My name is Juan and I live in Madrid."),
    ]
    rag_pipeline.get_component("retriever").retriever.document_store.write_documents(documents)

    # Query and assert
    question = "Who lives in Germany?"

    result = rag_pipeline.run({"query_multiplexer": {"value": question}})

    pprint(result)


if __name__ == "__main__":
    self_correcting_pipeline()
