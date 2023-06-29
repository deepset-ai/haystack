import logging
from copy import deepcopy

import pandas as pd
import pytest
from rank_bm25 import BM25
import numpy as np

from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack.schema import Document
from haystack.testing import DocumentStoreBaseTestAbstract


class TestInMemoryDocumentStore(DocumentStoreBaseTestAbstract):
    @pytest.fixture
    def ds(self):
        return InMemoryDocumentStore(return_embedding=True, use_bm25=True)

    @pytest.mark.integration
    def test_delete_index(self, ds, documents):
        """Contrary to other Document Stores, this doesn't raise if the index is empty"""
        ds.write_documents(documents, index="custom_index")
        assert ds.get_document_count(index="custom_index") == len(documents)
        ds.delete_index(index="custom_index")
        assert ds.get_document_count(index="custom_index") == 0

    @pytest.mark.integration
    def test_ne_filters(self, ds, documents):
        """
        InMemory doesn't include documents if the field is missing,
        so we customize this test
        """
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"year": {"$ne": "2020"}})
        assert len(result) == 3

    @pytest.mark.integration
    def test_get_documents_by_id(self, ds, documents):
        """
        The base test uses the batch_size param that's not supported
        here, so we override the test case
        """
        ds.write_documents(documents)
        ids = [doc.id for doc in documents]
        result = {doc.id for doc in ds.get_documents_by_id(ids)}
        assert set(ids) == result

    @pytest.mark.integration
    def test_update_bm25(self, ds, documents):
        ds.write_documents(documents)
        bm25_representation = ds.bm25[ds.index]
        assert isinstance(bm25_representation, BM25)
        assert bm25_representation.corpus_size == ds.get_document_count()

    @pytest.mark.integration
    def test_update_bm25_table(self, ds):
        table_doc = Document(
            content=pd.DataFrame(columns=["id", "text"], data=[[0, "This is a test"], ["2", "This is another test"]]),
            content_type="table",
        )
        ds.write_documents([table_doc])
        bm25_representation = ds.bm25[ds.index]
        assert isinstance(bm25_representation, BM25)
        assert bm25_representation.corpus_size == ds.get_document_count()

    @pytest.mark.integration
    def test_memory_query(self, ds, documents):
        ds.write_documents(documents)
        query_text = "Bar"
        docs = ds.query(query=query_text, top_k=1)
        assert len(docs) == 1
        assert "A Bar Document" in docs[0].content

    @pytest.mark.integration
    def test_memory_query_batch(self, ds, documents):
        ds.write_documents(documents)
        query_texts = ["Foo", "Bar"]
        docs = ds.query_batch(queries=query_texts, top_k=5)
        assert len(docs) == 2
        assert len(docs[0]) == 5
        assert "A Foo Document" in docs[0][0].content
        assert len(docs[1]) == 5
        assert "A Bar Document" in docs[1][0].content

    @pytest.mark.integration
    def test_memory_query_by_embedding_batch(self, ds, documents):
        documents = [doc for doc in documents if doc.embedding is not None]
        ds.write_documents(documents)
        query_embs = [doc.embedding for doc in documents]
        docs_batch = ds.query_by_embedding_batch(query_embs=query_embs, top_k=5)
        assert len(docs_batch) == 6
        for docs, query_emb in zip(docs_batch, query_embs):
            assert len(docs) == 5
            assert (docs[0].embedding == query_emb).all()

    @pytest.mark.integration
    def test_memory_query_by_embedding_docs_wo_embeddings(self, ds, caplog):
        # write document but don't update embeddings
        ds.write_documents([Document(content="test Document")])

        query_embedding = np.random.rand(768).astype(np.float32)

        with caplog.at_level(logging.WARNING):
            docs = ds.query_by_embedding(query_emb=query_embedding, top_k=1)
            assert "Skipping some of your documents that don't have embeddings" in caplog.text
        assert len(docs) == 0

    @pytest.mark.integration
    def test_bm25_scores_not_changing_across_queries(self, ds, documents):
        """Test that computed scores which are returned to the user should not change when running multiple queries."""
        ds.write_documents(documents)
        retriever = BM25Retriever(ds, scale_score=False)
        queries = ["What is a Foo Document?", "What is a Bar Document?", "Tell me about a document without embeddings"]
        results_direct = []
        results_direct = [retriever.retrieve(query) for query in queries]
        results_copied = [deepcopy(retriever.retrieve(query)) for query in queries]
        scores_direct = [rd.score for rds in results_direct for rd in rds]
        scores_copied = [rc.score for rcs in results_copied for rc in rcs]

        assert scores_direct == scores_copied
