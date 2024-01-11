import os
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.pipeline_utils import build_rag_pipeline

API_KEY = "SET YOUR OPENAI API KEY HERE"

# We support many different databases. Here we load a simple and lightweight in-memory document store.
document_store = InMemoryDocumentStore()

# Create some example documents and add them to the document store.
documents = [
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome."),
]
document_store.write_documents(documents)

# Let's now build a simple RAG pipeline that uses a generative model to answer questions.
rag_pipeline = build_rag_pipeline(llm_api_key=API_KEY, document_store=document_store)
answers = rag_pipeline.run(query="Who lives in Rome?")
print(answers.data)
