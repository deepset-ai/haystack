from typing import List, Any, Optional, Dict, Type

import json
import logging
from pathlib import Path

from canals.component.types import Variadic
from haystack import Pipeline, Document, component, default_to_dict, default_from_dict, DeserializationError
from haystack.document_stores import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import GPTGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.routers.conditional_router import serialize_type, deserialize_type


logging.getLogger().setLevel(logging.DEBUG)


@component
class Switch:
    def __init__(self, type_: Type):
        self.type_ = type_
        component.set_input_types(self, value=Variadic[self.type_])
        component.set_output_types(self, value=self.type_)

    def to_dict(self):
        return default_to_dict(self, type_=serialize_type(self.type_))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Switch":
        data["init_parameters"]["type_"] = deserialize_type(data["init_parameters"]["type_"])
        return default_from_dict(cls, data)

    def run(self, **kwargs):
        return {"value": kwargs["value"][0]}


@component
class CheckForMissingAnswer:
    def __init__(self, missing_answer_token: str = "UNKNOWN"):
        self.missing_answer_token = missing_answer_token

    def to_dict(self):
        return default_to_dict(self, missing_answer_token=self.missing_answer_token)

    @component.output_types(replies=List[str], unanswered_query=str)
    def run(self, query: str, replies: List[str]):
        meaningful_replies = [reply for reply in replies if reply != self.missing_answer_token]
        if not meaningful_replies:
            return {"unanswered_query": query}
        return {"replies": meaningful_replies}


@component
class PaginatedRetriever:
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
                query=query[0], filters=filters, top_k=top_k, scale_score=scale_score
            )["documents"]

        if not self.retrieved_documents:
            raise ValueError("No more documents available :(")

        next_page = self.retrieved_documents[: self.page_size]
        self.retrieved_documents = self.retrieved_documents[self.page_size :]
        return {"documents": next_page}


# Create the RAG pipeline
prompt_template = """
Given these documents, answer the question.
If the documents don't provide enough information to answer the question, answer with the string "UNKNOWN".

Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
rag_pipeline = Pipeline(max_loops_allowed=10)
rag_pipeline.add_component(instance=Switch(type_=str), name="query")
rag_pipeline.add_component(
    instance=PaginatedRetriever(InMemoryBM25Retriever(document_store=InMemoryDocumentStore())), name="retriever"
)
rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
rag_pipeline.add_component(instance=GPTGenerator(), name="llm")
rag_pipeline.add_component(instance=CheckForMissingAnswer(), name="answer_checker")

rag_pipeline.connect("query", "retriever")
rag_pipeline.connect("query", "prompt_builder.question")
rag_pipeline.connect("query", "answer_checker")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")
rag_pipeline.connect("llm.replies", "answer_checker.replies")
rag_pipeline.connect("answer_checker.unanswered_query", "query")

# Draw the pipeline
rag_pipeline.draw("test_bm25_rag_pipeline.png")

# Serialize the pipeline to JSON
with open("test_bm25_rag_pipeline.json", "w") as f:
    json.dump(rag_pipeline.to_dict(), f)

# Load the pipeline back
with open("test_bm25_rag_pipeline.json", "r") as f:
    rag_pipeline = Pipeline.from_dict(json.load(f))

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

result = rag_pipeline.run({"query": {"value": question}})

from pprint import pprint

print()
pprint(result)
print()
