import pytest

from haystack import Document, DeserializationError
from haystack.components.retrievers.sentence_window_retrieval import SentenceWindowRetrieval
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentSplitter


class TestSentenceWindowRetrieval:
    def test_init_default(self):
        retrieval = SentenceWindowRetrieval(InMemoryDocumentStore())
        assert retrieval.window_size == 3

    def test_init_with_parameters(self):
        retrieval = SentenceWindowRetrieval(InMemoryDocumentStore(), window_size=5)
        assert retrieval.window_size == 5

    def test_init_with_invalid_window_size_parameter(self):
        with pytest.raises(ValueError):
            SentenceWindowRetrieval(InMemoryDocumentStore(), window_size=-2)

    def test_merge_documents(self):
        docs = [
            {
                "id": "doc_0",
                "content": "This is a text with some words. There is a ",
                "source_id": "c5d7c632affc486d0cfe7b3c0f4dc1d3896ea720da2b538d6d10b104a3df5f99",
                "page_number": 1,
                "split_id": 0,
                "split_idx_start": 0,
                "_split_overlap": [{"doc_id": "doc_1", "range": (0, 22)}],
            },
            {
                "id": "doc_1",
                "content": "some words. There is a second sentence. And there is ",
                "source_id": "c5d7c632affc486d0cfe7b3c0f4dc1d3896ea720da2b538d6d10b104a3df5f99",
                "page_number": 1,
                "split_id": 1,
                "split_idx_start": 21,
                "_split_overlap": [{"doc_id": "doc_0", "range": (20, 42)}, {"doc_id": "doc_2", "range": (0, 29)}],
            },
            {
                "id": "doc_2",
                "content": "second sentence. And there is also a third sentence",
                "source_id": "c5d7c632affc486d0cfe7b3c0f4dc1d3896ea720da2b538d6d10b104a3df5f99",
                "page_number": 1,
                "split_id": 2,
                "split_idx_start": 45,
                "_split_overlap": [{"doc_id": "doc_1", "range": (23, 52)}],
            },
        ]
        merged_text = SentenceWindowRetrieval.merge_documents_text([Document.from_dict(doc) for doc in docs])
        expected = "This is a text with some words. There is a second sentence. And there is also a third sentence"
        assert merged_text == expected

    def test_to_dict(self):
        window_retrieval = SentenceWindowRetrieval(InMemoryDocumentStore())
        data = window_retrieval.to_dict()

        assert data["type"] == "haystack.components.retrievers.sentence_window_retrieval.SentenceWindowRetrieval"
        assert data["init_parameters"]["window_size"] == 3
        assert (
            data["init_parameters"]["document_store"]["type"]
            == "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore"
        )

    def test_from_dict(self):
        data = {
            "type": "haystack.components.retrievers.sentence_window_retrieval.SentenceWindowRetrieval",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    "init_parameters": {},
                },
                "window_size": 5,
            },
        }
        component = SentenceWindowRetrieval.from_dict(data)
        assert isinstance(component.document_store, InMemoryDocumentStore)
        assert component.window_size == 5

    def test_from_dict_without_docstore(self):
        data = {"type": "SentenceWindowRetrieval", "init_parameters": {}}
        with pytest.raises(DeserializationError, match="Missing 'document_store' in serialization data"):
            SentenceWindowRetrieval.from_dict(data)

    def test_from_dict_without_docstore_type(self):
        data = {"type": "SentenceWindowRetrieval", "init_parameters": {"document_store": {"init_parameters": {}}}}
        with pytest.raises(DeserializationError, match="Missing 'type' in document store's serialization data"):
            SentenceWindowRetrieval.from_dict(data)

    def test_from_dict_non_existing_docstore(self):
        data = {
            "type": "SentenceWindowRetrieval",
            "init_parameters": {"document_store": {"type": "Nonexisting.Docstore", "init_parameters": {}}},
        }
        with pytest.raises(DeserializationError):
            SentenceWindowRetrieval.from_dict(data)

    def test_document_without_split_id(self):
        docs = [
            Document(content="This is a text with some words. There is a ", meta={"id": "doc_0"}),
            Document(content="some words. There is a second sentence. And there is ", meta={"id": "doc_1"}),
        ]
        with pytest.raises(ValueError):
            retriever = SentenceWindowRetrieval(document_store=InMemoryDocumentStore(), window_size=3)
            retriever.run(retrieved_documents=docs)

    def test_document_without_source_id(self):
        docs = [
            Document(content="This is a text with some words. There is a ", meta={"id": "doc_0", "split_id": 0}),
            Document(
                content="some words. There is a second sentence. And there is ", meta={"id": "doc_1", "split_id": 1}
            ),
        ]
        with pytest.raises(ValueError):
            retriever = SentenceWindowRetrieval(document_store=InMemoryDocumentStore(), window_size=3)
            retriever.run(retrieved_documents=docs)

    @pytest.mark.integration
    def test_run_with_pipeline(self):
        splitter = DocumentSplitter(split_length=10, split_overlap=5, split_by="word")
        text = (
            "This is a text with some words. There is a second sentence. And there is also a third sentence. "
            "It also contains a fourth sentence. And a fifth sentence. And a sixth sentence. And a seventh sentence"
        )

        doc = Document(content=text)

        docs = splitter.run([doc])
        ds = InMemoryDocumentStore()
        ds.write_documents(docs["documents"])

        retriever = SentenceWindowRetrieval(document_store=ds, window_size=3)
        result = retriever.run(retrieved_documents=[list(ds.storage.values())[3]])
        expected = {
            "context_windows": [
                "This is a text with some words. There is a second sentence. And there is also a third sentence. It "
                "also contains a fourth sentence. And a fifth sentence. And a sixth sentence. And a seventh sentence"
            ]
        }

        assert result == expected
