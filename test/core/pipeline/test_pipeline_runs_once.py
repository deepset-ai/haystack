from typing import Any, Dict, List

from haystack import Document, Pipeline, component
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


@component
class MockComponent:
    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str = None):
        return {"replies": ["Some reply"], "meta": ["Some meta"]}


def test(tmp_path):
    # Create a pipeline and save it to file
    docs = [Document(content="Rome is the capital of Italy"), Document(content="Paris is the capital of France")]
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs)
    template = """
    Given the following information, answer the question.
    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}
    Question: {{ query }}?
    """
    pipe = Pipeline()
    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=doc_store))
    pipe.add_component("prompt_builder", PromptBuilder(template=template))
    pipe.add_component("llm", MockComponent())
    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder", "llm")
    with open(tmp_path / "test.yaml", "w") as file:
        pipe.dump(file)

    # Read the pipeline from file and run it
    with open(tmp_path / "test.yaml", "r") as file:
        pipe_from_file = Pipeline.load(file)
    pipe_from_file.get_component("retriever").document_store.write_documents(docs)
    query = "What is the capital of France?"
    pipe_from_file.run({"prompt_builder": {"query": query}, "retriever": {"query": query}})

    # Assert that each component in the pipeline was visited only once
    for node in pipe_from_file.graph.nodes:
        assert pipe_from_file.graph.nodes[node]["visits"] == 1
