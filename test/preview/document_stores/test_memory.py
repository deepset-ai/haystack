import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from haystack.preview import Document
from haystack.preview.document_stores import DocumentStore, MemoryDocumentStore, DocumentStoreError


from haystack.preview.testing.document_store import DocumentStoreBaseTests


class TestMemoryDocumentStore(DocumentStoreBaseTests):
    """
    Test MemoryDocumentStore's specific features
    """

    @pytest.fixture
    def docstore(self) -> MemoryDocumentStore:
        return MemoryDocumentStore()

    @pytest.mark.unit
    def test_to_dict(self):
        store = MemoryDocumentStore()
        data = store.to_dict()
        assert data == {
            "type": "MemoryDocumentStore",
            "init_parameters": {
                "bm25_tokenization_regex": r"(?u)\b\w\w+\b",
                "bm25_algorithm": "BM25Okapi",
                "bm25_parameters": {},
                "embedding_similarity_function": "dot_product",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        store = MemoryDocumentStore(
            bm25_tokenization_regex="custom_regex",
            bm25_algorithm="BM25Plus",
            bm25_parameters={"key": "value"},
            embedding_similarity_function="cosine",
        )
        data = store.to_dict()
        assert data == {
            "type": "MemoryDocumentStore",
            "init_parameters": {
                "bm25_tokenization_regex": "custom_regex",
                "bm25_algorithm": "BM25Plus",
                "bm25_parameters": {"key": "value"},
                "embedding_similarity_function": "cosine",
            },
        }

    @pytest.mark.unit
    @patch("haystack.preview.document_stores.memory.document_store.re")
    def test_from_dict(self, mock_regex):
        data = {
            "type": "MemoryDocumentStore",
            "init_parameters": {
                "bm25_tokenization_regex": "custom_regex",
                "bm25_algorithm": "BM25Plus",
                "bm25_parameters": {"key": "value"},
            },
        }
        store = MemoryDocumentStore.from_dict(data)
        mock_regex.compile.assert_called_with("custom_regex")
        assert store.tokenizer
        assert store.bm25_algorithm.__name__ == "BM25Plus"
        assert store.bm25_parameters == {"key": "value"}

    @pytest.mark.unit
    def test_bm25_retrieval(self, docstore: DocumentStore):
        docstore = MemoryDocumentStore()
        # Tests if the bm25_retrieval method returns the correct document based on the input query.
        docs = [Document(text="Hello world"), Document(text="Haystack supports multiple languages")]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="What languages?", top_k=1)
        assert len(results) == 1
        assert results[0].text == "Haystack supports multiple languages"

    @pytest.mark.unit
    def test_bm25_retrieval_with_empty_document_store(self, docstore: DocumentStore, caplog):
        caplog.set_level(logging.INFO)
        # Tests if the bm25_retrieval method correctly returns an empty list when there are no documents in the DocumentStore.
        results = docstore.bm25_retrieval(query="How to test this?", top_k=2)
        assert len(results) == 0
        assert "No documents found for BM25 retrieval. Returning empty list." in caplog.text

    @pytest.mark.unit
    def test_bm25_retrieval_empty_query(self, docstore: DocumentStore):
        # Tests if the bm25_retrieval method returns a document when the query is an empty string.
        docs = [Document(text="Hello world"), Document(text="Haystack supports multiple languages")]
        docstore.write_documents(docs)
        with pytest.raises(ValueError, match="Query should be a non-empty string"):
            docstore.bm25_retrieval(query="", top_k=1)

    @pytest.mark.unit
    def test_bm25_retrieval_with_different_top_k(self, docstore: DocumentStore):
        # Tests if the bm25_retrieval method correctly changes the number of returned documents
        # based on the top_k parameter.
        docs = [
            Document(text="Hello world"),
            Document(text="Haystack supports multiple languages"),
            Document(text="Python is a popular programming language"),
        ]
        docstore.write_documents(docs)

        # top_k = 2
        results = docstore.bm25_retrieval(query="languages", top_k=2)
        assert len(results) == 2

        # top_k = 3
        results = docstore.bm25_retrieval(query="languages", top_k=3)
        assert len(results) == 3

    # Test two queries and make sure the results are different
    @pytest.mark.unit
    def test_bm25_retrieval_with_two_queries(self, docstore: DocumentStore):
        # Tests if the bm25_retrieval method returns different documents for different queries.
        docs = [
            Document(text="Javascript is a popular programming language"),
            Document(text="Java is a popular programming language"),
            Document(text="Python is a popular programming language"),
            Document(text="Ruby is a popular programming language"),
            Document(text="PHP is a popular programming language"),
        ]
        docstore.write_documents(docs)

        results = docstore.bm25_retrieval(query="Java", top_k=1)
        assert results[0].text == "Java is a popular programming language"

        results = docstore.bm25_retrieval(query="Python", top_k=1)
        assert results[0].text == "Python is a popular programming language"

    # Test a query, add a new document and make sure results are appropriately updated
    @pytest.mark.unit
    def test_bm25_retrieval_with_updated_docs(self, docstore: DocumentStore):
        # Tests if the bm25_retrieval method correctly updates the retrieved documents when new
        # documents are added to the DocumentStore.
        docs = [Document(text="Hello world")]
        docstore.write_documents(docs)

        results = docstore.bm25_retrieval(query="Python", top_k=1)
        assert len(results) == 1

        docstore.write_documents([Document(text="Python is a popular programming language")])
        results = docstore.bm25_retrieval(query="Python", top_k=1)
        assert len(results) == 1
        assert results[0].text == "Python is a popular programming language"

        docstore.write_documents([Document(text="Java is a popular programming language")])
        results = docstore.bm25_retrieval(query="Python", top_k=1)
        assert len(results) == 1
        assert results[0].text == "Python is a popular programming language"

    @pytest.mark.unit
    def test_bm25_retrieval_with_scale_score(self, docstore: DocumentStore):
        docs = [Document(text="Python programming"), Document(text="Java programming")]
        docstore.write_documents(docs)

        results1 = docstore.bm25_retrieval(query="Python", top_k=1, scale_score=True)
        # Confirm that score is scaled between 0 and 1
        assert 0 <= results1[0].score <= 1

        # Same query, different scale, scores differ when not scaled
        results = docstore.bm25_retrieval(query="Python", top_k=1, scale_score=False)
        assert results[0].score != results1[0].score

    @pytest.mark.unit
    def test_bm25_retrieval_with_table_content(self, docstore: DocumentStore):
        # Tests if the bm25_retrieval method correctly returns a dataframe when the content_type is table.
        table_content = pd.DataFrame({"language": ["Python", "Java"], "use": ["Data Science", "Web Development"]})
        docs = [Document(dataframe=table_content), Document(text="Gardening"), Document(text="Bird watching")]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Java", top_k=1)
        assert len(results) == 1

        df = results[0].dataframe
        assert isinstance(df, pd.DataFrame)
        assert df.equals(table_content)

    @pytest.mark.unit
    def test_bm25_retrieval_with_text_and_table_content(self, docstore: DocumentStore, caplog):
        table_content = pd.DataFrame({"language": ["Python", "Java"], "use": ["Data Science", "Web Development"]})
        document = Document(text="Gardening", dataframe=table_content)
        docs = [
            document,
            Document(text="Python"),
            Document(text="Bird Watching"),
            Document(text="Gardening"),
            Document(text="Java"),
        ]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Gardening", top_k=2)
        assert document in results
        assert "both text and dataframe content" in caplog.text
        results = docstore.bm25_retrieval(query="Python", top_k=2)
        assert document not in results

    @pytest.mark.unit
    def test_bm25_retrieval_default_filter_for_text_and_dataframes(self, docstore: DocumentStore):
        docs = [
            Document(array=np.array([1, 2, 3])),
            Document(text="Gardening", array=np.array([1, 2, 3])),
            Document(text="Bird watching"),
        ]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="doesn't matter, top_k is 10", top_k=10)
        assert len(results) == 2

    @pytest.mark.unit
    def test_bm25_retrieval_with_filters(self, docstore: DocumentStore):
        selected_document = Document(text="Gardening", array=np.array([1, 2, 3]), metadata={"selected": True})
        docs = [Document(array=np.array([1, 2, 3])), selected_document, Document(text="Bird watching")]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Java", top_k=10, filters={"selected": True})
        assert results == [selected_document]

    @pytest.mark.unit
    def test_bm25_retrieval_with_filters_keeps_default_filters(self, docstore: DocumentStore):
        docs = [
            Document(array=np.array([1, 2, 3]), metadata={"selected": True}),
            Document(text="Gardening", array=np.array([1, 2, 3])),
            Document(text="Bird watching"),
        ]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Java", top_k=10, filters={"selected": True})
        assert not len(results)

    @pytest.mark.unit
    def test_bm25_retrieval_with_filters_on_text_or_dataframe(self, docstore: DocumentStore):
        document = Document(dataframe=pd.DataFrame({"language": ["Python", "Java"], "use": ["Data Science", "Web"]}))
        docs = [
            Document(array=np.array([1, 2, 3])),
            Document(text="Gardening"),
            Document(text="Bird watching"),
            document,
        ]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Java", top_k=10, filters={"text": None})
        assert results == [document]

    @pytest.mark.unit
    def test_bm25_retrieval_with_documents_with_mixed_content(self, docstore: DocumentStore):
        double_document = Document(text="Gardening", array=np.array([1, 2, 3]))
        docs = [Document(array=np.array([1, 2, 3])), double_document, Document(text="Bird watching")]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Java", top_k=10, filters={"array": {"$not": None}})
        assert results == [double_document]

    @pytest.mark.unit
    def test_embedding_retrieval(self):
        docstore = MemoryDocumentStore(embedding_similarity_function="cosine")
        # Tests if the embedding retrieval method returns the correct document based on the input query embedding.
        docs = [
            Document(text="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(text="Haystack supports multiple languages", embedding=[1.0, 1.0, 1.0, 1.0]),
        ]
        docstore.write_documents(docs)
        results = docstore.embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, filters={}, scale_score=False
        )
        assert len(results) == 1
        assert results[0].text == "Haystack supports multiple languages"

    @pytest.mark.unit
    def test_embedding_retrieval_invalid_query(self):
        docstore = MemoryDocumentStore()
        with pytest.raises(ValueError, match="query_embedding should be a non-empty list of floats"):
            docstore.embedding_retrieval(query_embedding=[])
        with pytest.raises(ValueError, match="query_embedding should be a non-empty list of floats"):
            docstore.embedding_retrieval(query_embedding=["invalid", "list", "of", "strings"])

    @pytest.mark.unit
    def test_embedding_retrieval_no_embeddings(self, caplog):
        caplog.set_level(logging.WARNING)
        docstore = MemoryDocumentStore()
        docs = [Document(text="Hello world"), Document(text="Haystack supports multiple languages")]
        docstore.write_documents(docs)
        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1])
        assert len(results) == 0
        assert "No Documents found with embeddings. Returning empty list." in caplog.text

    @pytest.mark.unit
    def test_embedding_retrieval_some_documents_wo_embeddings(self, caplog):
        caplog.set_level(logging.INFO)
        docstore = MemoryDocumentStore()
        docs = [
            Document(text="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(text="Haystack supports multiple languages"),
        ]
        docstore.write_documents(docs)
        docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1])
        assert "Skipping some Documents that don't have an embedding." in caplog.text

    @pytest.mark.unit
    def test_embedding_retrieval_documents_different_embedding_sizes(self):
        docstore = MemoryDocumentStore()
        docs = [
            Document(text="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(text="Haystack supports multiple languages", embedding=[1.0, 1.0]),
        ]
        docstore.write_documents(docs)

        with pytest.raises(DocumentStoreError, match="The embedding size of all Documents should be the same."):
            docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1])

    @pytest.mark.unit
    def test_embedding_retrieval_query_documents_different_embedding_sizes(self):
        docstore = MemoryDocumentStore()
        docs = [Document(text="Hello world", embedding=[0.1, 0.2, 0.3, 0.4])]
        docstore.write_documents(docs)

        with pytest.raises(
            DocumentStoreError,
            match="The embedding size of the query should be the same as the embedding size of the Documents.",
        ):
            docstore.embedding_retrieval(query_embedding=[0.1, 0.1])

    @pytest.mark.unit
    def test_embedding_retrieval_with_different_top_k(self):
        docstore = MemoryDocumentStore()
        docs = [
            Document(text="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(text="Haystack supports multiple languages", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(text="Python is a popular programming language", embedding=[0.5, 0.5, 0.5, 0.5]),
        ]
        docstore.write_documents(docs)

        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2)
        assert len(results) == 2

        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=3)
        assert len(results) == 3

    @pytest.mark.unit
    def test_embedding_retrieval_with_scale_score(self):
        docstore = MemoryDocumentStore()
        docs = [
            Document(text="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(text="Haystack supports multiple languages", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(text="Python is a popular programming language", embedding=[0.5, 0.5, 0.5, 0.5]),
        ]
        docstore.write_documents(docs)

        results1 = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, scale_score=True)
        # Confirm that score is scaled between 0 and 1
        assert 0 <= results1[0].score <= 1

        # Same query, different scale, scores differ when not scaled
        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, scale_score=False)
        assert results[0].score != results1[0].score

    @pytest.mark.unit
    def test_embedding_retrieval_return_embedding(self):
        docstore = MemoryDocumentStore(embedding_similarity_function="cosine")
        docs = [
            Document(text="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(text="Haystack supports multiple languages", embedding=[1.0, 1.0, 1.0, 1.0]),
        ]
        docstore.write_documents(docs)

        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, return_embedding=False)
        assert results[0].embedding is None

        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, return_embedding=True)
        assert results[0].embedding == [1.0, 1.0, 1.0, 1.0]

    @pytest.mark.unit
    def test_compute_cosine_similarity_scores(self):
        docstore = MemoryDocumentStore(embedding_similarity_function="cosine")
        docs = [
            Document(text="Document 1", embedding=[1.0, 0.0, 0.0, 0.0]),
            Document(text="Document 2", embedding=[1.0, 1.0, 1.0, 1.0]),
        ]

        scores = docstore._compute_query_embedding_similarity_scores(
            embedding=[0.1, 0.1, 0.1, 0.1], documents=docs, scale_score=False
        )
        assert scores == [0.5, 1.0]

    @pytest.mark.unit
    def test_compute_dot_product_similarity_scores(self):
        docstore = MemoryDocumentStore(embedding_similarity_function="dot_product")
        docs = [
            Document(text="Document 1", embedding=[1.0, 0.0, 0.0, 0.0]),
            Document(text="Document 2", embedding=[1.0, 1.0, 1.0, 1.0]),
        ]

        scores = docstore._compute_query_embedding_similarity_scores(
            embedding=[0.1, 0.1, 0.1, 0.1], documents=docs, scale_score=False
        )
        assert scores == [0.1, 0.4]
