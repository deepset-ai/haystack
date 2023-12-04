import os
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipeline_utils import build_rag_pipeline

# We are model agnostic but for default RAG pipeline we'll use an OpenAI model
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable to use this pipeline.")

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
pipe = build_rag_pipeline(document_store=document_store)
answers = pipe.run(query="Who lives in Rome?")
print(answers.data)
