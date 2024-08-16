from haystack.components.retrievers.sentence_window_retrieval import SentenceWindowRetrieval
from haystack.components.retrievers.sentence_window_retriever import SentenceWindowRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from unittest.mock import patch


class TestSentenceWindowRetrieval:
    def test_init_default(self):
        retriever = SentenceWindowRetrieval(InMemoryDocumentStore())
        assert retriever.window_size == 3

    def test_init_calls_parent(self):
        with patch.object(SentenceWindowRetriever, "__init__", return_value=None) as mock_init:
            document_store = InMemoryDocumentStore()
            retriever = SentenceWindowRetrieval(document_store)
            mock_init.assert_called_once_with(document_store)
