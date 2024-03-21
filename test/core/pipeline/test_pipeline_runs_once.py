from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.utils import Secret


def test():
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

    # ToDo: mock OpenAIGenerator
    pipe.add_component("llm", OpenAIGenerator(api_key=Secret.from_env_var('OPENAI_API_KEY')))
    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder", "llm")

    with open('test.yaml', "w") as file:
        pipe.dump(file)

    with open('test.yaml', "r") as file:
        pipe_from_file = Pipeline.load(file)

    query = "What is the capital of France?"
    pipe_from_file.get_component("retriever").document_store.write_documents(docs)
    pipe_from_file.run({"prompt_builder": {"query": query}, "retriever": {"query": query}})

    for node in pipe_from_file.graph.nodes:
        assert pipe_from_file.graph.nodes[node]['visits'] == 1
