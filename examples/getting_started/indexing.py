from pathlib import Path

from haystack.document_stores import InMemoryDocumentStore
from haystack.pipeline_utils import build_indexing_pipeline

# We support many different databases. Here we load a simple and lightweight in-memory document store.
document_store = InMemoryDocumentStore()

# Let's now build indexing pipeline that indexes PDFs and text files from a test folder.
indexing_pipeline = build_indexing_pipeline(
    document_store=document_store, embedding_model="sentence-transformers/all-mpnet-base-v2"
)
result = indexing_pipeline.run(files=list(Path("../../test/test_files").iterdir()))
print(result)
