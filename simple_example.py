from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tracing import disable_tracing

disable_tracing()


document_store = InMemoryDocumentStore()

documents = [
    Document(content="Bob lives in Paris."),
    Document(content="Alice lives in London."),
    Document(content="Ivy lives in Melbourne."),
    Document(content="Kate lives in Brisbane."),
    Document(content="Liam lives in Adelaide."),
]

document_store.write_documents(documents)

template = """{% message role="user" %}
Rewrite the following query to be used for keyword search.
{{ query }}
{% endmessage %}
"""

p = Pipeline()
p.add_component("prompt_builder", ChatPromptBuilder(template=template))
p.add_component("llm", OpenAIChatGenerator(model="gpt-4.1-mini"))
p.add_component("retriever", InMemoryBM25Retriever(document_store=document_store, top_k=3))

p.connect("prompt_builder", "llm")
p.connect("llm", "retriever")

query = (
    "One day I would love visiting Brisbane. But for now I would like to know the names of the people who live there."
)
results = p.run(data={"prompt_builder": {"query": query}})

print(results)
