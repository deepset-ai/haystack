from pathlib import Path

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


# Create components and an indexing pipeline that converts txt and pdf files to documents, cleans and splits them, and
# indexes them for sparse and dense retrieval.
p = Pipeline()
p.add_component(instance=FileTypeRouter(mime_types=["text/plain", "application/pdf"]), name="file_type_router")
p.add_component(instance=TextFileToDocument(), name="text_file_converter")
p.add_component(instance=PyPDFToDocument(), name="pdf_file_converter")
p.add_component(instance=DocumentJoiner(), name="joiner")
p.add_component(instance=DocumentCleaner(), name="cleaner")
p.add_component(instance=DocumentSplitter(split_by="sentence", split_length=250, split_overlap=30), name="splitter")
p.add_component(
    instance=SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"), name="embedder"
)
p.add_component(instance=DocumentWriter(document_store=InMemoryDocumentStore()), name="writer")

p.connect("file_type_router.text/plain", "text_file_converter.sources")
p.connect("file_type_router.application/pdf", "pdf_file_converter.sources")
p.connect("text_file_converter.documents", "joiner.documents")
p.connect("pdf_file_converter.documents", "joiner.documents")
p.connect("joiner.documents", "cleaner.documents")
p.connect("cleaner.documents", "splitter.documents")
p.connect("splitter.documents", "embedder.documents")
p.connect("embedder.documents", "writer.documents")

# Take the current directory as input and run the pipeline
result = p.run({"file_type_router": {"sources": list(Path(".").iterdir())}})
