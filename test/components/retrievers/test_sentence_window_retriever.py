import pytest

from haystack import DeserializationError, Document, Pipeline
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.retrievers.sentence_window_retriever import SentenceWindowRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestSentenceWindowRetriever:
    def test_init_default(self):
        retriever = SentenceWindowRetriever(InMemoryDocumentStore())
        assert retriever.window_size == 3

    def test_init_with_parameters(self):
        retriever = SentenceWindowRetriever(InMemoryDocumentStore(), window_size=5)
        assert retriever.window_size == 5

    def test_init_with_invalid_window_size_parameter(self):
        with pytest.raises(ValueError):
            SentenceWindowRetriever(InMemoryDocumentStore(), window_size=-2)

    def test_merge_documents(self):
        docs = [
            {
                "id": "doc_0",
                "content": "This is a text with some words. There is a ",
                "source_id": "c5d7c632affc486d0cfe7b3c0f4dc1d3896ea720da2b538d6d10b104a3df5f99",
                "page_number": 1,
                "split_id": 0,
                "split_idx_start": 0,
                "_split_overlap": [{"doc_id": "doc_1", "range": (0, 23)}],
            },
            {
                "id": "doc_1",
                "content": "some words. There is a second sentence. And there is ",
                "source_id": "c5d7c632affc486d0cfe7b3c0f4dc1d3896ea720da2b538d6d10b104a3df5f99",
                "page_number": 1,
                "split_id": 1,
                "split_idx_start": 20,
                "_split_overlap": [{"doc_id": "doc_0", "range": (20, 43)}, {"doc_id": "doc_2", "range": (0, 29)}],
            },
            {
                "id": "doc_2",
                "content": "second sentence. And there is also a third sentence",
                "source_id": "c5d7c632affc486d0cfe7b3c0f4dc1d3896ea720da2b538d6d10b104a3df5f99",
                "page_number": 1,
                "split_id": 2,
                "split_idx_start": 43,
                "_split_overlap": [{"doc_id": "doc_1", "range": (23, 52)}],
            },
        ]
        merged_text = SentenceWindowRetriever.merge_documents_text([Document.from_dict(doc) for doc in docs])
        expected = "This is a text with some words. There is a second sentence. And there is also a third sentence"
        assert merged_text == expected

    def test_to_dict(self):
        window_retriever = SentenceWindowRetriever(InMemoryDocumentStore())
        data = window_retriever.to_dict()

        assert data["type"] == "haystack.components.retrievers.sentence_window_retriever.SentenceWindowRetriever"
        assert data["init_parameters"]["window_size"] == 3
        assert (
            data["init_parameters"]["document_store"]["type"]
            == "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore"
        )

    def test_from_dict(self):
        data = {
            "type": "haystack.components.retrievers.sentence_window_retriever.SentenceWindowRetriever",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    "init_parameters": {},
                },
                "window_size": 5,
            },
        }
        component = SentenceWindowRetriever.from_dict(data)
        assert isinstance(component.document_store, InMemoryDocumentStore)
        assert component.window_size == 5

    def test_from_dict_without_docstore(self):
        data = {"type": "SentenceWindowRetriever", "init_parameters": {}}
        with pytest.raises(DeserializationError, match="Missing 'document_store' in serialization data"):
            SentenceWindowRetriever.from_dict(data)

    def test_from_dict_without_docstore_type(self):
        data = {"type": "SentenceWindowRetriever", "init_parameters": {"document_store": {"init_parameters": {}}}}
        with pytest.raises(DeserializationError):
            SentenceWindowRetriever.from_dict(data)

    def test_from_dict_non_existing_docstore(self):
        data = {
            "type": "SentenceWindowRetriever",
            "init_parameters": {"document_store": {"type": "Nonexisting.Docstore", "init_parameters": {}}},
        }
        with pytest.raises(DeserializationError):
            SentenceWindowRetriever.from_dict(data)

    def test_document_without_split_id(self):
        docs = [
            Document(content="This is a text with some words. There is a ", meta={"id": "doc_0"}),
            Document(content="some words. There is a second sentence. And there is ", meta={"id": "doc_1"}),
        ]
        with pytest.raises(ValueError):
            retriever = SentenceWindowRetriever(document_store=InMemoryDocumentStore(), window_size=3)
            retriever.run(retrieved_documents=docs)

    def test_document_without_source_id(self):
        docs = [
            Document(content="This is a text with some words. There is a ", meta={"id": "doc_0", "split_id": 0}),
            Document(
                content="some words. There is a second sentence. And there is ", meta={"id": "doc_1", "split_id": 1}
            ),
        ]
        with pytest.raises(ValueError):
            retriever = SentenceWindowRetriever(document_store=InMemoryDocumentStore(), window_size=3)
            retriever.run(retrieved_documents=docs)

    def test_run_invalid_window_size(self):
        docs = [Document(content="This is a text with some words. There is a ", meta={"id": "doc_0", "split_id": 0})]
        with pytest.raises(ValueError):
            retriever = SentenceWindowRetriever(document_store=InMemoryDocumentStore(), window_size=0)
            retriever.run(retrieved_documents=docs)

    def test_constructor_parameter_does_not_change(self):
        retriever = SentenceWindowRetriever(InMemoryDocumentStore(), window_size=5)
        assert retriever.window_size == 5

        doc = {
            "id": "doc_0",
            "content": "This is a text with some words. There is a ",
            "source_id": "c5d7c632affc486d0cfe7b3c0f4dc1d3896ea720da2b538d6d10b104a3df5f99",
            "page_number": 1,
            "split_id": 0,
            "split_idx_start": 0,
            "_split_overlap": [{"doc_id": "doc_1", "range": (0, 23)}],
        }

        retriever.run(retrieved_documents=[Document.from_dict(doc)], window_size=1)
        assert retriever.window_size == 5

    def test_context_documents_returned_are_ordered_by_split_idx_start(self):
        docs = []
        accumulated_length = 0
        for sent in range(10):
            content = f"Sentence {sent}."
            docs.append(
                Document(
                    content=content,
                    meta={
                        "id": f"doc_{sent}",
                        "split_idx_start": accumulated_length,
                        "source_id": "source1",
                        "split_id": sent,
                    },
                )
            )
            accumulated_length += len(content)

        import random

        random.shuffle(docs)

        doc_store = InMemoryDocumentStore()
        doc_store.write_documents(docs)
        retriever = SentenceWindowRetriever(document_store=doc_store, window_size=3)

        # run the retriever with a document whose content = "Sentence 4."
        result = retriever.run(retrieved_documents=[doc for doc in docs if doc.content == "Sentence 4."])

        # assert that the context documents are in the correct order
        assert len(result["context_documents"]) == 7
        assert [doc.meta["split_idx_start"] for doc in result["context_documents"]] == [11, 22, 33, 44, 55, 66, 77]

    @pytest.mark.integration
    def test_run_with_pipeline(self):
        splitter = DocumentSplitter(split_length=1, split_overlap=0, split_by="period")
        text = (
            "This is a text with some words. There is a second sentence. And there is also a third sentence. "
            "It also contains a fourth sentence. And a fifth sentence. And a sixth sentence. And a seventh sentence"
        )
        doc = Document(content=text)
        docs = splitter.run([doc])
        doc_store = InMemoryDocumentStore()
        doc_store.write_documents(docs["documents"])

        pipe = Pipeline()
        pipe.add_component("bm25_retriever", InMemoryBM25Retriever(doc_store, top_k=1))
        pipe.add_component(
            "sentence_window_retriever", SentenceWindowRetriever(document_store=doc_store, window_size=2)
        )
        pipe.connect("bm25_retriever", "sentence_window_retriever")
        result = pipe.run({"bm25_retriever": {"query": "third"}})

        assert result["sentence_window_retriever"]["context_windows"] == [
            "This is a text with some words. There is a second sentence. And there is also a third sentence. "
            "It also contains a fourth sentence. And a fifth sentence."
        ]
        assert len(result["sentence_window_retriever"]["context_documents"]) == 5

        result = pipe.run({"bm25_retriever": {"query": "third"}, "sentence_window_retriever": {"window_size": 1}})
        assert result["sentence_window_retriever"]["context_windows"] == [
            " There is a second sentence. And there is also a third sentence. It also contains a fourth sentence."
        ]
        assert len(result["sentence_window_retriever"]["context_documents"]) == 3

    @pytest.mark.integration
    def test_serialization_deserialization_in_pipeline(self):
        doc_store = InMemoryDocumentStore()
        pipe = Pipeline()
        pipe.add_component("bm25_retriever", InMemoryBM25Retriever(doc_store, top_k=1))
        pipe.add_component(
            "sentence_window_retriever", SentenceWindowRetriever(document_store=doc_store, window_size=2)
        )
        pipe.connect("bm25_retriever", "sentence_window_retriever")

        serialized = pipe.to_dict()
        deserialized = Pipeline.from_dict(serialized)

        assert deserialized == pipe
