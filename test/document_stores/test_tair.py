from typing import List, Union, Dict, Any
from random import choice, randint, random
from sentence_transformers import SentenceTransformer
# from haystack.retriever.dense import EmbeddingRetriever
import torch

import os
import numpy as np
from inspect import getmembers, isclass, isfunction
from unittest.mock import MagicMock, ANY

import pytest

from haystack.document_stores.tairvector import tair
from haystack.document_stores.tairvector import TairDocumentStore
from haystack.schema import Document
from haystack.errors import FilterError, TairDocumentStoreError
from haystack.testing import DocumentStoreBaseTestAbstract
from haystack.nodes.retriever import DensePassageRetriever

# Set metadata fields used during testing for TairDocumentStore meta_config
META_FIELDS = ["meta_field", "name", "date", "numeric_field", "odd_document"]


class TestTairDocumentStore(DocumentStoreBaseTestAbstract):
    # Fixtures
    index="haystack_tests"
    url = "redis://vector:Vectortest123@r-bp1wvb0pgh7x58zoctpd.redis.rds.aliyuncs.com:6379/0"
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')

    @pytest.fixture
    def ds(self):
        return TairDocumentStore(
            url=self.url,
            embedding_dim=512,
            embedding_field="embedding",
            index=self.index,
            similarity="IP",
            index_type="HNSW",
            recreate_index=True,
        )

    @pytest.fixture
    def doc_store_with_docs(self, ds: TairDocumentStore, documents: List[Document]) -> TairDocumentStore:
        """
        This fixture provides a pre-populated document store and takes care of cleaning up after each test
        """
        ds.write_documents(documents, self.index)
        return ds

    @pytest.fixture
    def docs_all_formats(self) -> List[Union[Document, Dict[str, Any]]]:
        return [
            # metafield at the top level for backward compatibility
            {
                "content": "My name is Paul and I live in New York",
                "meta_field": "test-1",
                "name": "file_1.txt",
                "date": "2019-10-01",
                "numeric_field": 5.0,
                "odd_document": True,
                "year": "2021",
                "month": "02",
                "embedding": self.model.encode("My name is Paul and I live in New York")
            },
            # "dict" format
            {
                "content": "My name is Carla and I live in Berlin",
                "meta": {
                    "meta_field": "test-2",
                    "name": "file_2.txt",
                    "date": "2020-03-01",
                    "numeric_field": 5.5,
                    "odd_document": False,
                    "year": "2021",
                    "month": "02",
                },
                "embedding": self.model.encode("My name is Carla and I live in Berlin")
            },
            # Document object
            Document(
                content="My name is Christelle and I live in Paris",
                meta={
                    "meta_field": "test-3",
                    "name": "file_3.txt",
                    "date": "2018-10-01",
                    "numeric_field": 4.5,
                    "odd_document": True,
                    "year": "2020",
                    "month": "02",
                },
                embedding=self.model.encode("My name is Christelle and I live in Paris")
            ),
            Document(
                content="My name is Camila and I live in Madrid",
                meta={
                    "meta_field": "test-4",
                    "name": "file_4.txt",
                    "date": "2021-02-01",
                    "numeric_field": 3.0,
                    "odd_document": False,
                    "year": "2020",
                },
                embedding=self.model.encode("My name is Camila and I live in Madrid")
            ),
            Document(
                content="My name is Matteo and I live in Rome",
                meta={
                    "meta_field": "test-5",
                    "name": "file_5.txt",
                    "date": "2019-01-01",
                    "numeric_field": 0.0,
                    "odd_document": True,
                    "year": "2020",
                },
                embedding=self.model.encode("My name is Matteo and I live in Rome")
            ),
            Document(
                content="My name is Adele and I live in London",
                meta={
                    "meta_field": "test-5",
                    "name": "file_5.txt",
                    "date": "2019-01-01",
                    "numeric_field": 0.0,
                    "odd_document": True,
                    "year": "2021",
                },
                embedding=self.model.encode("My name is Adele and I live in London")
            ),
            # Without meta
            Document(content="My name is Ahmed and I live in Cairo", embedding=self.model.encode("My name is Ahmed and I live in Cairo")),
            Document(content="My name is Bruce and I live in Gotham", embedding=self.model.encode("My name is Bruce and I live in Gotham")),
            Document(content="My name is Peter and I live in Quahog", embedding=self.model.encode("My name is Peter and I live in  Qindao")),
        ]

    @pytest.fixture
    def documents(self, docs_all_formats: List[Union[Document, Dict[str, Any]]]) -> List[Document]:
        return [Document.from_dict(doc) if isinstance(doc, dict) else doc for doc in docs_all_formats]

    #
    #  Tests
    #
    @pytest.mark.integration
    def test_write_documents(self, ds, documents):
        ds.write_documents(documents, self.index)

    @pytest.mark.integration
    def test_ne_filters(self, ds, documents):
        ds.write_documents(documents, self.index)

        # result = ds.get_all_documents(index=self.index)
        # assert len(result) == 9
        result = ds.get_all_documents(index=self.index, filters={"year": {"$ne": "2020"}})
        assert len(result) == 3

    @pytest.mark.integration
    def test_get_label_count(self, ds, labels):
        with pytest.raises(NotImplementedError):
            ds.get_label_count()

    @pytest.mark.integration
    def test_nin_filters(self, ds, documents):
        ds.write_documents(documents, self.index)

        result = ds.get_all_documents(filters={"year": {"$nin": ["2020", "2021", "n.a."]}})
        assert len(result) == 0

    @pytest.mark.integration
    def test_update_embeddings(self, ds, documents):
        ds.write_documents(documents, self.index)

        retriever = DensePassageRetriever()
        ds.update_embeddings(retriever=retriever, index=self.index)

    @pytest.mark.integration
    def test_query_by_embedding(self, ds, documents):
        ds.write_documents(documents, self.index)

        # retriever = DensePassageRetriever()
        # ds.update_embeddings(retriever=retriever, index=self.index)
        query_context = "My name is Adele and I live in London"
        query_emb = self.model.encode(query_context)
        # query_emb.resize((1, ds.embedding_dim))
        # query_emb = np.random.rand(ds.embedding_dim).astype(np.float32)
        # query_emb = np.array(query_emb)
        ds.query_by_embedding(query_emb=query_emb, filters=None, top_k=3, index=self.index)

    @pytest.mark.integration
    def test_delete_index(self, ds, documents):
        ds.write_documents(documents, self.index)
        ds.delete_index(self.index)

    @pytest.mark.integration
    def test_delete_documents(self, ds, documents):
        ds.write_documents(documents, self.index)
        ids = ["b5872cc8d37f0acc19c88b6f028f97b2", "8a94dfce37493094390eb12b3088a20c", "598dc9f8cc824fe847ce4c9161f94eb9"]
        ds.delete_documents(self.index, ids)

    @pytest.mark.integration
    def test_get_embedding_count(self, ds, documents):
        """
        We expect 9 docs with embeddings because all documents in the documents fixture for this class contain
        embeddings.
        """
        ds.write_documents(documents)
        assert ds.get_embedding_count() == 9

    @pytest.mark.unit
    def test_get_all_labels_legacy_document_id(self, ds, monkeypatch):
        monkeypatch.setattr(
            ds,
            "get_all_documents",
            MagicMock(
                return_value=[
                    Document.from_dict(
                        {
                            "content": "My name is Carla and I live in Berlin",
                            "content_type": "text",
                            "score": None,
                            "meta": {
                                "label-id": "d9256445-7b8a-4a33-a558-402ec84d6881",
                                "query": "query_1",
                                "label-is-correct-answer": False,
                                "label-is-correct-document": True,
                                "label-document-content": "My name is Carla and I live in Berlin",
                                "label-document-id": "a0747b83aea0b60c4b114b15476dd32d",
                                "label-no-answer": False,
                                "label-origin": "user-feedback",
                                "label-created-at": "2023-02-07 14:46:54",
                                "label-updated-at": None,
                                "label-pipeline-id": None,
                                "label-document-meta-meta_field": "test-2",
                                "label-document-meta-name": "file_2.txt",
                                "label-document-meta-date": "2020-03-01",
                                "label-document-meta-numeric_field": 5.5,
                                "label-document-meta-odd_document": False,
                                "label-document-meta-year": "2021",
                                "label-document-meta-month": "02",
                                "label-meta-name": "label_1",
                                "label-meta-year": "2021",
                                "label-answer-answer": "the answer is 1",
                                "label-answer-type": "extractive",
                                "label-answer-score": None,
                                "label-answer-context": None,
                                # legacy document_id answer
                                "label-answer-document-id": "a0747b83aea0b60c4b114b15476dd32d",
                                "label-answer-offsets-in-document-start": None,
                                "label-answer-offsets-in-document-end": None,
                                "label-answer-offsets-in-context-start": None,
                                "label-answer-offsets-in-context-end": None,
                            },
                            "id_hash_keys": ["content"],
                            "embedding": None,
                            "id": "d9256445-7b8a-4a33-a558-402ec84d6881",
                        }
                    )
                ]
            ),
        )

        labels = ds.get_all_labels()

        assert labels[0].answer.document_ids == ["a0747b83aea0b60c4b114b15476dd32d"]



