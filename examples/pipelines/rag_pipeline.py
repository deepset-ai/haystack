import os
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder

# We are model agnostic :) Here, we use OpenAI models and load an api key from environment variables.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# We support many different databases. Here we load a simple and lightweight in-memory document store.
document_store = InMemoryDocumentStore()

# Create some example documents and add them to the document store.
documents = [
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome."),
]
document_store.write_documents(documents)

# Build a RAG pipeline with a Retriever to get relevant documents to the query and a OpenAIGenerator interacting with LLMs using a custom prompt.
prompt_template = """
Given these documents, answer the question.\nDocuments:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}

\nQuestion: {{question}}
\nAnswer:
"""
rag_pipeline = Pipeline()
rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="retriever")
rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
rag_pipeline.add_component(instance=OpenAIGenerator(api_key=OPENAI_API_KEY), name="llm")
rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("llm.meta", "answer_builder.meta")
rag_pipeline.connect("retriever", "answer_builder.documents")

# Ask a question on the data you just added.
question = "Where does Mark live?"
result = rag_pipeline.run(
    {"retriever": {"query": question}, "prompt_builder": {"question": question}, "answer_builder": {"query": question}}
)

# For details, like which documents were used to generate the answer, look into the GeneratedAnswer object
print(result["answer_builder"]["answers"])
