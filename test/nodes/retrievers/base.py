import re
from typing import List

import os
from uuid import UUID
from abc import ABC, abstractmethod

import pytest

from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores import (
    BaseDocumentStore,
    InMemoryDocumentStore,
    ElasticsearchDocumentStore,
    FAISSDocumentStore,
    OpenSearchDocumentStore,
    MilvusDocumentStore,
    WeaviateDocumentStore,
    PineconeDocumentStore,
)
from haystack.utils.doc_store import launch_es, launch_milvus, launch_opensearch, launch_weaviate
from test.conftest import get_sql_url, META_FIELDS as PINECONE_META_FIELDS, mock_pinecone


class RunOnAllDocstores:
    """
    Infrastructure class to parametrize children suites on all docstores.

    Select which docstores to run the tests on with `--document_store_type=<comma separated list>`

    Set defaults by manually adding `@pytest.mark.skip` to the child suite.
    """

    @pytest.fixture(params=["memory", "elasticsearch", "opensearch", "faiss", "milvus", "weaviate", "pinecone"])
    def docstore(self, unsupported_docstores, request, monkeypatch):
        if request.param in unsupported_docstores:
            pytest.skip(reason=f"{request.param} is not supported by this suite.")

        # if request.param == "elasticsearch":
        #     launch_es()
        # if request.param == "openasearch":
        #     launch_opensearch()
        # elif request.param == "milvus":
        #     launch_milvus()
        # elif request.param == "weaviate":
        #     launch_weaviate()
        # el
        if request.param == "pinecone":
            mock_pinecone(monkeypatch)
        return request.getfixturevalue(request.param)

    @pytest.fixture
    def unsupported_docstores(self) -> List[str]:
        """
        Override this method to specify which docstores are unsupported by your suite.
        """
        return []

    #
    # InMemory
    #
    @pytest.fixture
    def memory(self, docs: List[Document]):
        docstore = InMemoryDocumentStore()
        docstore.write_documents(docs)
        yield docstore

    #
    # Elasticsearch
    #
    @pytest.fixture
    def elasticsearch(self, docs: List[Document]):
        docstore = ElasticsearchDocumentStore(
            index="haystack_test",
            return_embedding=True,
            embedding_dim=768,
            embedding_field="embedding",
            similarity="cosine",
            recreate_index=True,
        )
        docstore.write_documents(docs)
        yield docstore
        docstore.delete_documents()

    #
    # Opensearch
    #
    @pytest.fixture
    def opensearch(self, docs: List[Document]):
        docstore = OpenSearchDocumentStore(
            index="haystack_test",
            return_embedding=True,
            embedding_dim=768,
            embedding_field="embedding",
            similarity="cosine",
            recreate_index=True,
        )
        docstore.write_documents(docs)
        yield docstore
        docstore.delete_documents()

    #
    # FAISS
    #
    @pytest.fixture
    def faiss(self, tmp_path, docs: List[Document]):
        docstore = FAISSDocumentStore(
            sql_url=get_sql_url(tmp_path), return_embedding=True, isolation_level="AUTOCOMMIT"
        )
        docstore.write_documents(docs)
        yield docstore

    #
    # Milvus
    #
    @pytest.fixture
    def milvus(self, tmp_path, docs: List[Document]):
        docstore = MilvusDocumentStore(
            index="haystack_test", sql_url=get_sql_url(tmp_path), return_embedding=True, isolation_level="AUTOCOMMIT"
        )
        docstore.write_documents(docs)
        yield docstore
        docstore.delete_documents()

    #
    # Weaviate
    #
    @pytest.fixture
    def weaviate(self, docs: List[Document]):
        docstore = WeaviateDocumentStore(index="haystack_test")
        docstore.write_documents(docs)
        yield docstore
        docstore.delete_documents()

    #
    # Pinecone
    #
    @pytest.fixture
    def pinecone(self, docs: List[Document]):
        docstore: BaseDocumentStore = PineconeDocumentStore(
            index="haystack_test",
            api_key=os.environ.get("PINECONE_API_KEY") or "fake-haystack-test-key",
            metadata_config={"indexed": PINECONE_META_FIELDS},
        )
        docstore.write_documents(docs)
        yield docstore
        docstore.delete_documents()


class ABC_TestRetriever(RunOnAllDocstores, ABC):
    """
    Base class for the suites of all Retrievers, including multimodal ones.
    """


class ABC_TestTextRetriever(ABC_TestRetriever, ABC):
    """
    Base class for the suites of all Retrievers that can handle text and implement filtering.
    """

    QUESTION = "Who lives in Berlin?"
    ANSWER = "My name is Carla and I live in Berlin"

    @abstractmethod
    @pytest.fixture
    def retriever(self, docstore):
        raise NotImplementedError("Abstract method, use a subclass")

    @pytest.fixture
    def text_docs(self) -> List[Document]:
        return [
            Document(
                content="My name is Paul and I live in New York",
                meta={
                    "meta_field": "test2",
                    "name": "filename2",
                    "date_field": "2019-10-01",
                    "numeric_field": 5.0,
                    "odd_field": 0,
                },
            ),
            Document(
                content="My name is Carla and I live in Berlin",
                meta={
                    "meta_field": "test1",
                    "name": "filename1",
                    "date_field": "2020-03-01",
                    "numeric_field": 5.5,
                    "odd_field": 1,
                },
            ),
            Document(
                content="My name is Christelle and I live in Paris",
                meta={
                    "meta_field": "test3",
                    "name": "filename3",
                    "date_field": "2018-10-01",
                    "numeric_field": 4.5,
                    "odd_field": 1,
                },
            ),
            Document(
                content="My name is Camila and I live in Madrid",
                meta={
                    "meta_field": "test4",
                    "name": "filename4",
                    "date_field": "2021-02-01",
                    "numeric_field": 3.0,
                    "odd_field": 0,
                },
            ),
            Document(
                content="My name is Matteo and I live in Rome",
                meta={
                    "meta_field": "test5",
                    "name": "filename5",
                    "date_field": "2019-01-01",
                    "numeric_field": 0.0,
                    "odd_field": 1,
                },
            ),
        ]

    @pytest.fixture
    def text_docs_with_ids(self, text_docs) -> List[Document]:
        # Should be already sorted
        uuids = [
            UUID("190a2421-7e48-4a49-a639-35a86e202dfb"),
            UUID("20ff1706-cb55-4704-8ae8-a3459774c8dc"),
            UUID("5078722f-07ae-412d-8ccb-b77224c4bacb"),
            UUID("81d8ca45-fad1-4d1c-8028-d818ef33d755"),
            UUID("f985789f-1673-4d8f-8d5f-2b8d3a9e8e23"),
        ]
        uuids.sort()
        for doc, uuid in zip(text_docs, uuids):
            doc.id = str(uuid)
        return text_docs

    #
    # Tests
    #

    def test_retrieval(self, retriever: BaseRetriever):
        res = retriever.retrieve(query="Who lives in Berlin?")
        assert len(res) > 0
        assert res[0].content == "My name is Carla and I live in Berlin"

    def test_retrieval_with_single_filter(self, retriever: BaseRetriever):
        result = retriever.retrieve(query="Who lives in Berlin?", filters={"name": ["filename1"]}, top_k=5)
        assert len(result) == 1
        assert result[0].content == "My name is Carla and I live in Berlin"
        assert result[0].meta["name"] == "filename1"

    def test_retrieval_with_many_filter(self, retriever: BaseRetriever):
        result = retriever.retrieve(
            query="Who lives in Berlin?",
            filters={"name": ["filename5", "filename1"], "meta_field": ["test1", "test2"]},
            top_k=5,
        )
        assert len(result) == 1
        assert result[0].meta["name"] == "filename1"

    def test_retrieval_with_many_filter_no_match(self, retriever: BaseRetriever):
        result = retriever.retrieve(
            query="Who lives in Berlin?", filters={"name": ["filename1"], "meta_field": ["test2", "test3"]}, top_k=5
        )
        assert len(result) == 0

    def test_batch_retrieval(self, retriever: BaseRetriever):
        res = retriever.retrieve_batch(queries=["Who lives in Berlin?", "Who lives in Madrid?"], top_k=5)

        assert len(res) == 2  # 2 queries
        assert len(res[0]) == 5  # top_k = 5
        assert res[0][0].content == "My name is Carla and I live in Berlin"
        assert res[1][0].content == "My name is Camila and I live in Madrid"
