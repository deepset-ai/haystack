from typing import List

from haystack.document_stores import InMemoryDocumentStore
from haystack.pipeline_utils import build_rag_pipeline, build_indexing_pipeline
from haystack.pipeline_utils.indexing import download_files

document_store = InMemoryDocumentStore()
indexing_pipeline = build_indexing_pipeline(
    document_store=document_store,
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    supported_mime_types=["text/plain", "text/html"],
)

# before indexing, let's get some documents from the web
files: List[str] = download_files(sources=["http://www.paulgraham.com/superlinear.html"])
indexing_pipeline.run(files=files)

# now do the RAG pipeline on these documents
rag = build_rag_pipeline(
    document_store=document_store,
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    generation_model="gpt-3.5-turbo",
)

# run the query and print the result
result = rag.run(query="What are superlinear returns and why are they important?")
print(result.data)
