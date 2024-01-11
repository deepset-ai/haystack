from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.pipeline_utils import build_rag_pipeline, build_indexing_pipeline
from haystack.pipeline_utils.indexing import download_files

# We are model agnostic :) In this getting started you can choose any OpenAI or Huggingface TGI generation model
generation_model = "gpt-3.5-turbo"
API_KEY = "sk-..."  # ADD YOUR KEY HERE

# We support many different databases. Here, we load a simple and lightweight in-memory database.
document_store = InMemoryDocumentStore()

# Download example files from web
files = download_files(sources=["http://www.paulgraham.com/superlinear.html"])

# Pipelines are our main abstratcion.
# Here we create a pipeline that can index TXT and HTML. You can also use your own private files.
indexing_pipeline = build_indexing_pipeline(
    document_store=document_store,
    embedding_model="intfloat/e5-base-v2",
    supported_mime_types=["text/plain", "text/html"],  # "application/pdf"
)
indexing_pipeline.run(files=files)  # you can also supply files=[path_to_directory], which is searched recursively

# RAG pipeline with vector-based retriever + LLM
rag_pipeline = build_rag_pipeline(
    document_store=document_store,
    embedding_model="intfloat/e5-base-v2",
    generation_model=generation_model,
    llm_api_key=API_KEY,
)

# For details, like which documents were used to generate the answer, look into the result object
result = rag_pipeline.run(query="What are superlinear returns and why are they important?")
print(result.data)
