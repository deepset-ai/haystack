import json
from pathlib import Path

from haystack.preview import Pipeline
from haystack.preview.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.preview.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.preview.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.preview.components.routers import FileTypeRouter, DocumentJoiner
from haystack.preview.components.writers import DocumentWriter
from haystack.preview.document_stores import InMemoryDocumentStore
from haystack.preview.components.retrievers import InMemoryEmbeddingRetriever


def test_dense_doc_search_pipeline(tmp_path):
    # Create the indexing pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        instance=FileTypeRouter(mime_types=["text/plain", "application/pdf"]), name="file_type_router"
    )
    indexing_pipeline.add_component(instance=TextFileToDocument(), name="text_file_converter")
    indexing_pipeline.add_component(instance=PyPDFToDocument(), name="pdf_file_converter")
    indexing_pipeline.add_component(instance=DocumentJoiner(), name="joiner")
    indexing_pipeline.add_component(instance=DocumentCleaner(), name="cleaner")
    indexing_pipeline.add_component(
        instance=DocumentSplitter(split_by="sentence", split_length=250, split_overlap=30), name="splitter"
    )
    indexing_pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
        name="embedder",
    )
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=InMemoryDocumentStore()), name="writer")

    indexing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
    indexing_pipeline.connect("file_type_router.application/pdf", "pdf_file_converter.sources")
    indexing_pipeline.connect("text_file_converter.documents", "joiner.documents")
    indexing_pipeline.connect("pdf_file_converter.documents", "joiner.documents")
    indexing_pipeline.connect("joiner.documents", "cleaner.documents")
    indexing_pipeline.connect("cleaner.documents", "splitter.documents")
    indexing_pipeline.connect("splitter.documents", "embedder.documents")
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    # Draw the pipeline
    indexing_pipeline.draw(tmp_path / "test_dense_doc_search_indexing_pipeline.png")

    # Serialize the pipeline to JSON
    with open(tmp_path / "test_dense_doc_search_indexing_pipeline.json", "w") as f:
        print(json.dumps(indexing_pipeline.to_dict(), indent=4))
        json.dump(indexing_pipeline.to_dict(), f)

    # Load the pipeline back
    with open(tmp_path / "test_dense_doc_search_indexing_pipeline.json", "r") as f:
        indexing_pipeline = Pipeline.from_dict(json.load(f))

    query_pipeline = Pipeline()
    query_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(
            document_store=indexing_pipeline.get_component("writer").document_store, top_k=20
        ),
        name="retriever",
    )

    # Draw the pipeline
    query_pipeline.draw(tmp_path / "test_dense_doc_search_query_pipeline.png")

    # Serialize the pipeline to JSON
    with open(tmp_path / "test_dense_doc_search_query_pipeline.json", "w") as f:
        print(json.dumps(query_pipeline.to_dict(), indent=4))
        json.dump(query_pipeline.to_dict(), f)

    # Load the pipeline back
    with open(tmp_path / "test_dense_doc_search_query_pipeline.json", "r") as f:
        query_pipeline = Pipeline.from_dict(json.load(f))

    # TODO Ensure there is a directory with a txt file and a pdf file
    result = indexing_pipeline.run({"file_type_router": {"sources": Path(tmp_path).iterdir()}})

    filled_document_store = indexing_pipeline.get_component("writer").document_store
    assert result["writer"]["documents_written"] == 3
    assert filled_document_store.count_documents() == 3

    result = query_pipeline.run({"retriever": {"query": "Who lives in Rome?"}})
    assert result["retriever"]["documents"][0].text == "My name is Giorgio and I live in Rome."
