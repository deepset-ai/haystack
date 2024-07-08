from haystack import Document
from haystack.components.retrievers.sentence_window_retrieval import SentenceWindowRetrieval
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestSentenceWindowRetrieval:
    def test_init_default(self):
        retrieval = SentenceWindowRetrieval(InMemoryDocumentStore(), window_size=3)
        assert retrieval.window_size == 3

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
        merged_text = SentenceWindowRetrieval.merge_documents([Document.from_dict(doc) for doc in docs])
        expected = "This is a text with some words. There is a second sentence. And there is also a third sentence"
        assert merged_text == expected
