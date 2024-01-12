from typing import Dict, Any
from pathlib import Path
from datetime import datetime
import os

from haystack import Pipeline
from haystack.dataclasses import ByteStream
from haystack.components.others import Multiplexer
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


document_store = InMemoryDocumentStore()

p = Pipeline()
p.add_component(instance=FileTypeRouter(mime_types=["text/plain", "application/pdf"]), name="file_type_router")
p.add_component(instance=Multiplexer(Dict[str, Any]), name="metadata_multiplexer")
p.add_component(instance=TextFileToDocument(), name="text_file_converter")
p.add_component(instance=PyPDFToDocument(), name="pdf_file_converter")
p.add_component(instance=DocumentJoiner(), name="joiner")
p.add_component(instance=DocumentCleaner(), name="cleaner")
p.add_component(instance=DocumentSplitter(split_by="sentence", split_length=250, split_overlap=30), name="splitter")
p.add_component(instance=DocumentWriter(document_store=document_store), name="writer")

p.connect("file_type_router.text/plain", "text_file_converter.sources")
p.connect("file_type_router.application/pdf", "pdf_file_converter.sources")
p.connect("metadata_multiplexer", "text_file_converter.meta")
p.connect("metadata_multiplexer", "pdf_file_converter.meta")
p.connect("text_file_converter.documents", "joiner.documents")
p.connect("pdf_file_converter.documents", "joiner.documents")
p.connect("joiner.documents", "cleaner.documents")
p.connect("cleaner.documents", "splitter.documents")
p.connect("splitter.documents", "writer.documents")

# Add metadata to your files by using ByteStream
sources = []
for position, path in enumerate(list(Path(".").iterdir())):
    if path.is_file():
        # Create the ByteStream
        source = ByteStream.from_file_path(path)
        # Add the metadata
        source.meta["path"] = path
        source.meta["position"] = position
        sources.append(source)

result = p.run(
    {
        "file_type_router": {"sources": sources},
        "metadata_multiplexer": {"value": {"date_added": datetime.now().isoformat()}},
    }
)

assert all("date_added" in doc.meta for doc in document_store.filter_documents())
