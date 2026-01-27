"""
Complex example to show how flexible connections/conversion could work.
In real world, some of the connections we make here do not make much sense. :-)
"""

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
document_store.write_documents(
    [
        Document(
            content="""The InMemoryDocumentStore is a very simple document store with no extra services or dependencies.
            It is great for experimenting with Haystack, however we do not recommend using it for production."""
        ),
        Document(
            content="""Chroma is an open source vector database capable of storing collections of documents along with
            their metadata, creating embeddings for documents and queries, and searching the collections filtering by
            document metadata or content. Additionally, Chroma supports multi-modal embedding functions.

Chroma can be used in-memory, as an embedded database, or in a client-server fashion. When running in-memory, Chroma
can still keep its contents on disk across different sessions. This allows users to quickly put together prototypes
using the in-memory version and later move to production, where the client-server version is deployed."""
        ),
        Document(
            content="""Qdrant is a powerful high-performance, massive-scale vector database. The QdrantDocumentStore
            can be used with any Qdrant instance, in-memory, locally persisted, hosted, and the official Qdrant Cloud."""
        ),
        Document(
            content="""How can I make sure that my GPU is being engaged when I use Haystack?
You will want to ensure that a CUDA enabled GPU is being engaged when Haystack is running (you can check by running
nvidia-smi -l on your command line). Components which can be sped up by GPU have a device argument in their constructor.
 For more details, check the Device Management page."""
        ),
        Document(
            content="""Are you tracking my Haystack usage?
We only collect anonymous usage statistics of Haystack pipeline components. Read more about telemetry in Haystack or
how you can opt out on the Telemetry page."""
        ),
    ]
)

llm_refiner = OpenAIChatGenerator(model="gpt-4.1-mini")
agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"))

refiner_prompt_builder = ChatPromptBuilder(
    template="""
{% message role="user" %}
Extract the core search keywords from this query: {{ query }}
{% endmessage %}
"""
)

retriever = InMemoryBM25Retriever(document_store=document_store, top_k=3)

prompt_builder = PromptBuilder(
    template="""Please answer this user question: {{ query }}

Use the following documents to answer the question:
{% for doc in documents %}
Document {{ loop.index }}:
{{ doc.content }}
{% endfor %}
"""
)

p = Pipeline()
p.add_component("refiner_prompt_builder", refiner_prompt_builder)
p.add_component("llm_refiner", llm_refiner)
p.add_component("retriever", retriever)
p.add_component("prompt_builder", prompt_builder)
p.add_component("agent", agent)

# --- CONNECTIONS demonstrating simplifications ---

# A. Exact Match: list[ChatMessage] -> list[ChatMessage]
p.connect("refiner_prompt_builder", "llm_refiner")

# B. Restricted List Unwrapping + Automatic Chat/Text Bridging
# llm_refiner.replies (list[ChatMessage]) -> retriever.query (str)
p.connect("llm_refiner", "retriever")
p.connect("retriever.documents", "prompt_builder.documents")

# C. Automatic Chat/Text Bridging + Automatic List Wrapping
# string_producer.prompt (str) -> agent.messages (list[ChatMessage])
p.connect("prompt_builder.prompt", "agent.messages")


query = (
    "I am building a Haystack application with massive scale in mind. What is the best document store/database to use?"
)

results = p.run(data={"refiner_prompt_builder": {"query": query}, "prompt_builder": {"query": query}})

print(f"Agent Final Response: {results['agent']['last_message'].text}")
