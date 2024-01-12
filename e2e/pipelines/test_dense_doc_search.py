import json

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever


def test_dense_doc_search_pipeline(tmp_path, samples_path):
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
        instance=SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"), name="embedder"
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

    # Draw the indexing pipeline
    indexing_pipeline.draw(tmp_path / "test_dense_doc_search_indexing_pipeline.png")

    # Serialize the indexing pipeline to JSON
    with open(tmp_path / "test_dense_doc_search_indexing_pipeline.json", "w") as f:
        print(json.dumps(indexing_pipeline.to_dict(), indent=4))
        json.dump(indexing_pipeline.to_dict(), f)

    # Load the indexing pipeline back
    with open(tmp_path / "test_dense_doc_search_indexing_pipeline.json", "r") as f:
        indexing_pipeline = Pipeline.from_dict(json.load(f))

    indexing_result = indexing_pipeline.run({"file_type_router": {"sources": list(samples_path.iterdir())}})
    filled_document_store = indexing_pipeline.get_component("writer").document_store

    assert indexing_result["writer"]["documents_written"] == 2
    assert filled_document_store.count_documents() == 2

    # Create the querying pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"), name="text_embedder"
    )
    query_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=filled_document_store, top_k=20), name="embedding_retriever"
    )
    query_pipeline.connect("text_embedder", "embedding_retriever")

    querying_result = query_pipeline.run({"text_embedder": {"text": "Who lives in Rome?"}})
    assert querying_result["embedding_retriever"]["documents"][0].content == "My name is Giorgio and I live in Rome."

    # Draw the querying pipeline
    query_pipeline.draw(tmp_path / "test_dense_doc_search_query_pipeline.png")

    # Serialize the querying pipeline to JSON
    with open(tmp_path / "test_dense_doc_search_query_pipeline.json", "w") as f:
        print(json.dumps(query_pipeline.to_dict(), indent=4))
        json.dump(query_pipeline.to_dict(), f)

    # Load the querying pipeline back
    with open(tmp_path / "test_dense_doc_search_query_pipeline.json", "r") as f:
        query_pipeline = Pipeline.from_dict(json.load(f))
