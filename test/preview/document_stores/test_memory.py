import logging

import pandas as pd
import pytest

from haystack.preview import Document
from haystack.preview.document_stores import MemoryDocumentStore

from test.preview.document_stores._base import DocumentStoreBaseTests


class TestMemoryDocumentStore(DocumentStoreBaseTests):
    """
    Test MemoryDocumentStore's specific features
    """

    @pytest.fixture
    def docstore(self) -> MemoryDocumentStore:
        return MemoryDocumentStore()

    @pytest.mark.unit
    def test_bm25_retrieval(self, docstore):
        docstore = MemoryDocumentStore()
        # Tests if the bm25_retrieval method returns the correct document based on the input query.
        docs = [Document(content="Hello world"), Document(content="Haystack supports multiple languages")]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="What languages?", top_k=1, filters={})
        assert len(results) == 1
        assert results[0].content == "Haystack supports multiple languages"

    @pytest.mark.unit
    def test_bm25_retrieval_with_empty_document_store(self, docstore, caplog):
        caplog.set_level(logging.INFO)
        # Tests if the bm25_retrieval method correctly returns an empty list when there are no documents in the store.
        results = docstore.bm25_retrieval(query="How to test this?", top_k=2)
        assert len(results) == 0
        assert "No documents found for BM25 retrieval. Returning empty list." in caplog.text

    @pytest.mark.unit
    def test_bm25_retrieval_empty_query(self, docstore):
        # Tests if the bm25_retrieval method returns a document when the query is an empty string.
        docs = [Document(content="Hello world"), Document(content="Haystack supports multiple languages")]
        docstore.write_documents(docs)
        with pytest.raises(ValueError, match=r"Query should be a non-empty string"):
            docstore.bm25_retrieval(query="", top_k=1)

    @pytest.mark.unit
    def test_bm25_retrieval_filter_only_one_allowed_doc_type_as_string(self, docstore):
        # Tests if the bm25_retrieval method returns a document when the query is an empty string.
        docs = [
            Document.from_dict({"content": pd.DataFrame({"language": ["Python", "Java"]}), "content_type": "table"}),
            Document(content="Haystack supports multiple languages"),
        ]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Python", top_k=3, filters={"content_type": "text"})
        assert len(results) == 1
        assert results[0].content == "Haystack supports multiple languages"

    @pytest.mark.unit
    def test_bm25_retrieval_filter_only_one_allowed_doc_type_as_list(self, docstore):
        # Tests if the bm25_retrieval method returns a document when the query is an empty string.
        docs = [
            Document.from_dict({"content": pd.DataFrame({"language": ["Python", "Java"]}), "content_type": "table"}),
            Document(content="Haystack supports multiple languages"),
        ]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Python", top_k=3, filters={"content_type": ["text"]})
        assert len(results) == 1
        assert results[0].content == "Haystack supports multiple languages"

    @pytest.mark.unit
    def test_bm25_retrieval_filter_two_allowed_doc_type_as_list(self, docstore):
        # Tests if the bm25_retrieval method returns a document when the query is an empty string.
        docs = [
            Document.from_dict({"content": pd.DataFrame({"language": ["Python", "Java"]}), "content_type": "table"}),
            Document(content="Haystack supports multiple languages"),
        ]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Python", top_k=3, filters={"content_type": ["text", "table"]})
        assert len(results) == 2

    @pytest.mark.unit
    def test_bm25_retrieval_filter_only_one_not_allowed_doc_type_as_string(self, docstore):
        # Tests if the bm25_retrieval method returns a document when the query is an empty string.
        docs = [
            Document(content=pd.DataFrame({"language": ["Python", "Java"]}), content_type="table"),
            Document(content="Haystack supports multiple languages"),
        ]
        docstore.write_documents(docs)
        with pytest.raises(
            ValueError, match="MemoryDocumentStore can do BM25 retrieval on no other document type than text or table."
        ):
            docstore.bm25_retrieval(query="Python", top_k=3, filters={"content_type": "audio"})

    @pytest.mark.unit
    def test_bm25_retrieval_filter_only_one_not_allowed_doc_type_as_list(self, docstore):
        # Tests if the bm25_retrieval method returns a document when the query is an empty string.
        docs = [
            Document(content=pd.DataFrame({"language": ["Python", "Java"]}), content_type="table"),
            Document(content="Haystack supports multiple languages"),
        ]
        docstore.write_documents(docs)
        with pytest.raises(
            ValueError, match="MemoryDocumentStore can do BM25 retrieval on no other document type than text or table."
        ):
            docstore.bm25_retrieval(query="Python", top_k=3, filters={"content_type": ["audio"]})

    @pytest.mark.unit
    def test_bm25_retrieval_filter_two_not_all_allowed_doc_type_as_list(self, docstore):
        # Tests if the bm25_retrieval method returns a document when the query is an empty string.
        docs = [
            Document.from_dict({"content": pd.DataFrame({"language": ["Python", "Java"]}), "content_type": "table"}),
            Document(content="Haystack supports multiple languages"),
        ]
        docstore.write_documents(docs)
        with pytest.raises(
            ValueError, match="MemoryDocumentStore can do BM25 retrieval on no other document type than text or table."
        ):
            docstore.bm25_retrieval(query="Python", top_k=3, filters={"content_type": ["text", "audio"]})

    @pytest.mark.unit
    def test_bm25_retrieval_with_different_top_k(self, docstore):
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
    def test_bm25_retrieval_with_two_queries(self, docstore):
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

    # Test a query, add a new document and make sure results are appropriately updated
    @pytest.mark.unit
    def test_bm25_retrieval_with_updated_docs(self, docstore):
        # Tests if the bm25_retrieval method correctly updates the retrieved documents when new
        # documents are added to the store.
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
    def test_bm25_retrieval_with_scale_score(self, docstore):
        docs = [Document(content="Python programming"), Document(content="Java programming")]
        docstore.write_documents(docs)

        results1 = docstore.bm25_retrieval(query="Python", top_k=1, scale_score=True)
        # Confirm that score is scaled between 0 and 1
        assert 0 <= results1[0].score <= 1

        # Same query, different scale, scores differ when not scaled
        results = docstore.bm25_retrieval(query="Python", top_k=1, scale_score=False)
        assert results[0].score != results1[0].score

    @pytest.mark.unit
    def test_bm25_retrieval_with_table_content(self, docstore):
        # Tests if the bm25_retrieval method correctly returns a dataframe when the content_type is table.
        table_content = pd.DataFrame({"language": ["Python", "Java"], "use": ["Data Science", "Web Development"]})
        docs = [
            Document(content=table_content, content_type="table"),
            Document(content="Gardening", content_type="text"),
            Document(content="Bird watching", content_type="text"),
        ]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="Java", top_k=1)
        assert len(results) == 1
        df = results[0].content
        assert isinstance(df, pd.DataFrame)
        assert df.equals(table_content)
