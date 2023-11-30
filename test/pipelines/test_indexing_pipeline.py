import pytest

from haystack.pipeline_utils.indexing import build_indexing_pipeline
from haystack.document_stores import InMemoryDocumentStore


class TestIndexingPipeline:
    #  indexing files without embeddings
    def test_indexing_files_without_embeddings(self, test_files_path):
        file_paths = [test_files_path / "txt" / "doc_1.txt", test_files_path / "txt" / "doc_2.txt"]
        document_store = InMemoryDocumentStore()
        pipeline = build_indexing_pipeline(document_store=document_store)
        result = pipeline.run(files=file_paths)
        assert result == {"documents_written": 2}

    #  indexing files with embeddings
    @pytest.mark.integration
    def test_indexing_files_with_embeddings(self, test_files_path):
        document_store = InMemoryDocumentStore()
        pipeline = build_indexing_pipeline(
            document_store=document_store, embedding_model="sentence-transformers/all-mpnet-base-v2"
        )
        file_paths = [test_files_path / "txt" / "doc_1.txt", test_files_path / "txt" / "doc_2.txt"]
        result = pipeline.run(files=file_paths)
        assert result == {"documents_written": 2}

    @pytest.mark.integration
    def test_indexing_dirs_with_embeddings(self, test_files_path):
        document_store = InMemoryDocumentStore()
        pipeline = build_indexing_pipeline(
            document_store=document_store, embedding_model="sentence-transformers/all-mpnet-base-v2"
        )
        file_paths = [test_files_path / "txt"]
        result = pipeline.run(files=file_paths)
        assert "documents_written" in result
        assert result["documents_written"] >= 3

    #  indexing multiple files
    def test_indexing_multiple_file_types(self, test_files_path):
        document_store = InMemoryDocumentStore()
        pipeline = build_indexing_pipeline(document_store=document_store)
        file_paths = [
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "txt" / "doc_2.txt",
            test_files_path / "pdf" / "sample_pdf_1.pdf",
        ]
        result = pipeline.run(files=file_paths)
        # pdf gets split into 2 documents
        assert result == {"documents_written": 4}

    #  indexing empty list of files
    def test_indexing_empty_list_of_files(self):
        document_store = InMemoryDocumentStore()
        pipeline = build_indexing_pipeline(document_store=document_store)
        result = pipeline.run(files=[])
        assert result == {"documents_written": 0}

    #  document store is not a DocumentStore instance
    def test_document_store_not_instance_of_document_store(self):
        document_store = "hello I am not a DocumentStore instance"
        with pytest.raises(ValueError):
            build_indexing_pipeline(document_store=document_store)

    #  embedding model is not found
    def test_embedding_model_not_found(self):
        document_store = InMemoryDocumentStore()
        with pytest.raises(ValueError, match="Could not find an embedder"):
            build_indexing_pipeline(document_store=document_store, embedding_model="invalid_model")

    @pytest.mark.integration
    def test_open_ai_embedding_model(self):
        document_store = InMemoryDocumentStore()
        pipe = build_indexing_pipeline(document_store=document_store, embedding_model="text-embedding-ada-002")
        # don't run the pipeline and waste credits, just check that it was created correctly
        assert pipe is not None
