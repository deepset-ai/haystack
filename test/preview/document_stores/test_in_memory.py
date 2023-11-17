import logging
from unittest.mock import patch

import pandas as pd
import pytest

from haystack.preview import Document
from haystack.preview.document_stores import InMemoryDocumentStore, DocumentStoreError


from haystack.preview.testing.document_store import DocumentStoreBaseTests


class TestMemoryDocumentStore(DocumentStoreBaseTests):  # pylint: disable=R0904
    """
    Test InMemoryDocumentStore's specific features
    """

    @pytest.fixture
    def docstore(self) -> InMemoryDocumentStore:
        return InMemoryDocumentStore()

    @pytest.mark.unit
    def test_to_dict(self):
        store = InMemoryDocumentStore()
        data = store.to_dict()
        assert data == {
            "type": "haystack.preview.document_stores.in_memory.document_store.InMemoryDocumentStore",
            "init_parameters": {
                "bm25_tokenization_regex": r"(?u)\b\w\w+\b",
                "bm25_algorithm": "BM25Okapi",
                "bm25_parameters": {},
                "embedding_similarity_function": "dot_product",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        store = InMemoryDocumentStore(
            bm25_tokenization_regex="custom_regex",
            bm25_algorithm="BM25Plus",
            bm25_parameters={"key": "value"},
            embedding_similarity_function="cosine",
        )
        data = store.to_dict()
        assert data == {
            "type": "haystack.preview.document_stores.in_memory.document_store.InMemoryDocumentStore",
            "init_parameters": {
                "bm25_tokenization_regex": "custom_regex",
                "bm25_algorithm": "BM25Plus",
                "bm25_parameters": {"key": "value"},
                "embedding_similarity_function": "cosine",
            },
        }

    @pytest.mark.unit
    @patch("haystack.preview.document_stores.in_memory.document_store.re")
    def test_from_dict(self, mock_regex):
        data = {
            "type": "haystack.preview.document_stores.in_memory.document_store.InMemoryDocumentStore",
            "init_parameters": {
                "bm25_tokenization_regex": "custom_regex",
                "bm25_algorithm": "BM25Plus",
                "bm25_parameters": {"key": "value"},
            },
        }
        store = InMemoryDocumentStore.from_dict(data)
        mock_regex.compile.assert_called_with("custom_regex")
        assert store.tokenizer
        assert store.bm25_algorithm.__name__ == "BM25Plus"
        assert store.bm25_parameters == {"key": "value"}

    @pytest.mark.unit
    def test_bm25_retrieval(self, docstore: InMemoryDocumentStore):
        docstore = InMemoryDocumentStore()
        # Tests if the bm25_retrieval method returns the correct document based on the input query.
        docs = [Document(content="Hello world"), Document(content="Haystack supports multiple languages")]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="What languages?", top_k=1)
        assert len(results) == 1
        assert results[0].content == "Haystack supports multiple languages"

    @pytest.mark.unit
    def test_bm25_retrieval_with_empty_document_store(self, docstore: InMemoryDocumentStore, caplog):
        caplog.set_level(logging.INFO)
        # Tests if the bm25_retrieval method correctly returns an empty list when there are no documents in the DocumentStore.
        results = docstore.bm25_retrieval(query="How to test this?", top_k=2)
        assert len(results) == 0
        assert "No documents found for BM25 retrieval. Returning empty list." in caplog.text

    @pytest.mark.unit
    def test_bm25_retrieval_empty_query(self, docstore: InMemoryDocumentStore):
        # Tests if the bm25_retrieval method returns a document when the query is an empty string.
        docs = [Document(content="Hello world"), Document(content="Haystack supports multiple languages")]
        docstore.write_documents(docs)
        with pytest.raises(ValueError, match="Query should be a non-empty string"):
            docstore.bm25_retrieval(query="", top_k=1)

    @pytest.mark.unit
    def test_bm25_retrieval_with_different_top_k(self, docstore: InMemoryDocumentStore):
        # Tests if the bm25_retrieval method correctly changes the number of returned documents
        # based on the top_k parameter.
        docs = [
            Document(content="Hello world"),
            Document(content="Haystack supports multiple languages"),
            Document(content="Python is a popular programming language"),
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
    def test_bm25_retrieval_with_two_queries(self, docstore: InMemoryDocumentStore):
        # Tests if the bm25_retrieval method returns different documents for different queries.
        docs = [
            Document(content="Javascript is a popular programming language"),
            Document(content="Java is a popular programming language"),
            Document(content="Python is a popular programming language"),
            Document(content="Ruby is a popular programming language"),
            Document(content="PHP is a popular programming language"),
        ]
        docstore.write_documents(docs)

        results = docstore.bm25_retrieval(query="Java", top_k=1)
        assert results[0].content == "Java is a popular programming language"

        results = docstore.bm25_retrieval(query="Python", top_k=1)
        assert results[0].content == "Python is a popular programming language"

    @pytest.mark.skip(reason="Filter is not working properly, see https://github.com/deepset-ai/haystack/issues/6153")
    def test_eq_filter_embedding(self, docstore: InMemoryDocumentStore, filterable_docs):
        pass

    # Test a query, add a new document and make sure results are appropriately updated
    @pytest.mark.unit
    def test_bm25_retrieval_with_updated_docs(self, docstore: InMemoryDocumentStore):
        # Tests if the bm25_retrieval method correctly updates the retrieved documents when new
        # documents are added to the DocumentStore.
        docs = [Document(content="Hello world")]
        docstore.write_documents(docs)

        results = docstore.bm25_retrieval(query="Python", top_k=1)
        assert len(results) == 1

        docstore.write_documents([Document(content="Python is a popular programming language")])
        results = docstore.bm25_retrieval(query="Python", top_k=1)
        assert len(results) == 1
        assert results[0].content == "Python is a popular programming language"

        docstore.write_documents([Document(content="Java is a popular programming language")])
        results = docstore.bm25_retrieval(query="Python", top_k=1)
        assert len(results) == 1
        assert results[0].content == "Python is a popular programming language"

    @pytest.mark.unit
    def test_bm25_retrieval_with_scale_score(self, docstore: InMemoryDocumentStore):
        docs = [Document(content="Python programming"), Document(content="Java programming")]
        docstore.write_documents(docs)

        results1 = docstore.bm25_retrieval(query="Python", top_k=1, scale_score=True)
        # Confirm that score is scaled between 0 and 1
        assert results1[0].score is not None
        assert 0.0 <= results1[0].score <= 1.0

        # Same query, different scale, scores differ when not scaled
        results = docstore.bm25_retrieval(query="Python", top_k=1, scale_score=False)
        assert results[0].score != results1[0].score

    @pytest.mark.unit
    def test_bm25_retrieval_with_table_content(self, docstore: InMemoryDocumentStore):
        # Tests if the bm25_retrieval method correctly returns a dataframe when the content_type is table.
        table_content = pd.DataFrame({"language": ["Python", "Java"], "use": ["Data Science", "Web Development"]})
        docs = [Document(dataframe=table_content), Document(content="Gardening"), Document(content="Bird watching")]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Java", top_k=1)
        assert len(results) == 1

        df = results[0].dataframe
        assert isinstance(df, pd.DataFrame)
        assert df.equals(table_content)

    @pytest.mark.unit
    def test_bm25_retrieval_with_text_and_table_content(self, docstore: InMemoryDocumentStore, caplog):
        table_content = pd.DataFrame({"language": ["Python", "Java"], "use": ["Data Science", "Web Development"]})
        document = Document(content="Gardening", dataframe=table_content)
        docs = [
            document,
            Document(content="Python"),
            Document(content="Bird Watching"),
            Document(content="Gardening"),
            Document(content="Java"),
        ]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Gardening", top_k=2)
        assert document.id in [d.id for d in results]
        assert "both text and dataframe content" in caplog.text
        results = docstore.bm25_retrieval(query="Python", top_k=2)
        assert document.id not in [d.id for d in results]

    @pytest.mark.unit
    def test_bm25_retrieval_default_filter_for_text_and_dataframes(self, docstore: InMemoryDocumentStore):
        docs = [Document(), Document(content="Gardening"), Document(content="Bird watching")]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="doesn't matter, top_k is 10", top_k=10)
        assert len(results) == 2

    @pytest.mark.unit
    def test_bm25_retrieval_with_filters(self, docstore: InMemoryDocumentStore):
        selected_document = Document(content="Gardening", meta={"selected": True})
        docs = [Document(), selected_document, Document(content="Bird watching")]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Java", top_k=10, filters={"selected": True})
        assert len(results) == 1
        assert results[0].id == selected_document.id

    @pytest.mark.unit
    def test_bm25_retrieval_with_filters_keeps_default_filters(self, docstore: InMemoryDocumentStore):
        docs = [Document(meta={"selected": True}), Document(content="Gardening"), Document(content="Bird watching")]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Java", top_k=10, filters={"selected": True})
        assert len(results) == 0

    @pytest.mark.unit
    def test_bm25_retrieval_with_filters_on_text_or_dataframe(self, docstore: InMemoryDocumentStore):
        document = Document(dataframe=pd.DataFrame({"language": ["Python", "Java"], "use": ["Data Science", "Web"]}))
        docs = [Document(), Document(content="Gardening"), Document(content="Bird watching"), document]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Java", top_k=10, filters={"content": None})
        assert len(results) == 1
        assert results[0].id == document.id

    @pytest.mark.unit
    def test_bm25_retrieval_with_documents_with_mixed_content(self, docstore: InMemoryDocumentStore):
        double_document = Document(content="Gardening", embedding=[1.0, 2.0, 3.0])
        docs = [Document(embedding=[1.0, 2.0, 3.0]), double_document, Document(content="Bird watching")]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Java", top_k=10, filters={"embedding": {"$not": None}})
        assert len(results) == 1
        assert results[0].id == double_document.id

    @pytest.mark.unit
    def test_embedding_retrieval(self):
        docstore = InMemoryDocumentStore(embedding_similarity_function="cosine")
        # Tests if the embedding retrieval method returns the correct document based on the input query embedding.
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Haystack supports multiple languages", embedding=[1.0, 1.0, 1.0, 1.0]),
        ]
        docstore.write_documents(docs)
        results = docstore.embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, filters={}, scale_score=False
        )
        assert len(results) == 1
        assert results[0].content == "Haystack supports multiple languages"

    @pytest.mark.unit
    def test_embedding_retrieval_invalid_query(self):
        docstore = InMemoryDocumentStore()
        with pytest.raises(ValueError, match="query_embedding should be a non-empty list of floats"):
            docstore.embedding_retrieval(query_embedding=[])
        with pytest.raises(ValueError, match="query_embedding should be a non-empty list of floats"):
            docstore.embedding_retrieval(query_embedding=["invalid", "list", "of", "strings"])  # type: ignore

    @pytest.mark.unit
    def test_embedding_retrieval_no_embeddings(self, caplog):
        caplog.set_level(logging.WARNING)
        docstore = InMemoryDocumentStore()
        docs = [Document(content="Hello world"), Document(content="Haystack supports multiple languages")]
        docstore.write_documents(docs)
        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1])
        assert len(results) == 0
        assert "No Documents found with embeddings. Returning empty list." in caplog.text

    @pytest.mark.unit
    def test_embedding_retrieval_some_documents_wo_embeddings(self, caplog):
        caplog.set_level(logging.INFO)
        docstore = InMemoryDocumentStore()
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Haystack supports multiple languages"),
        ]
        docstore.write_documents(docs)
        docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1])
        assert "Skipping some Documents that don't have an embedding." in caplog.text

    @pytest.mark.unit
    def test_embedding_retrieval_documents_different_embedding_sizes(self):
        docstore = InMemoryDocumentStore()
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Haystack supports multiple languages", embedding=[1.0, 1.0]),
        ]
        docstore.write_documents(docs)

        with pytest.raises(DocumentStoreError, match="The embedding size of all Documents should be the same."):
            docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1])

    @pytest.mark.unit
    def test_embedding_retrieval_query_documents_different_embedding_sizes(self):
        docstore = InMemoryDocumentStore()
        docs = [Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4])]
        docstore.write_documents(docs)

        with pytest.raises(
            DocumentStoreError,
            match="The embedding size of the query should be the same as the embedding size of the Documents.",
        ):
            docstore.embedding_retrieval(query_embedding=[0.1, 0.1])

    @pytest.mark.unit
    def test_embedding_retrieval_with_different_top_k(self):
        docstore = InMemoryDocumentStore()
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Haystack supports multiple languages", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="Python is a popular programming language", embedding=[0.5, 0.5, 0.5, 0.5]),
        ]
        docstore.write_documents(docs)

        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2)
        assert len(results) == 2

        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=3)
        assert len(results) == 3

    @pytest.mark.unit
    def test_embedding_retrieval_with_scale_score(self):
        docstore = InMemoryDocumentStore()
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Haystack supports multiple languages", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="Python is a popular programming language", embedding=[0.5, 0.5, 0.5, 0.5]),
        ]
        docstore.write_documents(docs)

        results1 = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, scale_score=True)
        # Confirm that score is scaled between 0 and 1
        assert results1[0].score is not None
        assert 0.0 <= results1[0].score <= 1.0

        # Same query, different scale, scores differ when not scaled
        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, scale_score=False)
        assert results[0].score != results1[0].score

    @pytest.mark.unit
    def test_embedding_retrieval_return_embedding(self):
        docstore = InMemoryDocumentStore(embedding_similarity_function="cosine")
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Haystack supports multiple languages", embedding=[1.0, 1.0, 1.0, 1.0]),
        ]
        docstore.write_documents(docs)

        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, return_embedding=False)
        assert results[0].embedding is None

        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, return_embedding=True)
        assert results[0].embedding == [1.0, 1.0, 1.0, 1.0]

    @pytest.mark.unit
    def test_compute_cosine_similarity_scores(self):
        docstore = InMemoryDocumentStore(embedding_similarity_function="cosine")
        docs = [
            Document(content="Document 1", embedding=[1.0, 0.0, 0.0, 0.0]),
            Document(content="Document 2", embedding=[1.0, 1.0, 1.0, 1.0]),
        ]

        scores = docstore._compute_query_embedding_similarity_scores(
            embedding=[0.1, 0.1, 0.1, 0.1], documents=docs, scale_score=False
        )
        assert scores == [0.5, 1.0]

    @pytest.mark.unit
    def test_compute_dot_product_similarity_scores(self):
        docstore = InMemoryDocumentStore(embedding_similarity_function="dot_product")
        docs = [
            Document(content="Document 1", embedding=[1.0, 0.0, 0.0, 0.0]),
            Document(content="Document 2", embedding=[1.0, 1.0, 1.0, 1.0]),
        ]

        scores = docstore._compute_query_embedding_similarity_scores(
            embedding=[0.1, 0.1, 0.1, 0.1], documents=docs, scale_score=False
        )
        assert scores == [0.1, 0.4]
