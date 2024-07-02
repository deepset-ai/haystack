# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Any, List

import pytest

from haystack import Pipeline, DeserializationError
from haystack.testing.factory import document_store_class
from haystack.components.retrievers.similar_documents_retriever import SimilarDocumentsRetriever
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever


@pytest.fixture()
def sample_docs():
    programming_docs = [
        Document(content="Javascript is a popular programming language"),
        Document(content="Python is a popular programming language"),
    ]
    dna_docs = [
        Document(content="A chromosome is a package of DNA"),
        Document(content="DNA carries genetic information"),
    ]
    all_docs = programming_docs + dna_docs
    return {"programming_docs": programming_docs, "dna_docs": dna_docs, "all_docs": all_docs}


@pytest.fixture()
def sample_document_store(sample_docs):
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(sample_docs["all_docs"])
    return doc_store


@pytest.fixture()
def mock_retrievers(sample_document_store):
    MyFakeRetriever1 = document_store_class("MyFakeRetriever1", bases=(InMemoryBM25Retriever,))
    retriever1 = MyFakeRetriever1(sample_document_store)
    retriever1.to_dict = lambda: {"type": "MyFakeRetriever1", "init_parameters": {}}

    MyFakeRetriever2 = document_store_class("MyFakeRetriever2", bases=(InMemoryBM25Retriever,))
    retriever2 = MyFakeRetriever2(sample_document_store)
    retriever2.to_dict = lambda: {"type": "MyFakeRetriever2", "init_parameters": {}}

    return [retriever1, retriever2]


class TestSimilarDocumentsRetriever:
    @classmethod
    def _documents_equal(cls, docs1: List[Document], docs2: List[Document]) -> bool:
        # Order doesn't matter; we sort before comparing
        docs1.sort(key=lambda x: x.id)
        docs2.sort(key=lambda x: x.id)
        return docs1 == docs2

    def test_init_empty_retrievers(self):
        with pytest.raises(ValueError):
            SimilarDocumentsRetriever(retrievers=[])

    def test_to_dict(self, mock_retrievers):
        component = SimilarDocumentsRetriever(retrievers=mock_retrievers)

        data = component.to_dict()
        assert data == {
            "type": "haystack.components.retrievers.similar_documents_retriever.SimilarDocumentsRetriever",
            "init_parameters": {
                "retrievers": [
                    {"type": "MyFakeRetriever1", "init_parameters": {}},
                    {"type": "MyFakeRetriever2", "init_parameters": {}},
                ]
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.retrievers.similar_documents_retriever.SimilarDocumentsRetriever",
            "init_parameters": {
                "retrievers": [
                    {
                        "type": "haystack.components.retrievers.in_memory.bm25_retriever.InMemoryBM25Retriever",
                        "init_parameters": {
                            "document_store": {
                                "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore"
                            }
                        },
                    },
                    {
                        "type": "haystack.components.retrievers.in_memory.bm25_retriever.InMemoryBM25Retriever",
                        "init_parameters": {
                            "document_store": {
                                "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore"
                            }
                        },
                    },
                ]
            },
        }
        component = SimilarDocumentsRetriever.from_dict(data)
        assert isinstance(component.retrievers[0], InMemoryBM25Retriever)
        assert len(component.retrievers) == 2

    def test_from_dict_without_retrievers(self):
        data = {
            "type": "haystack.components.retrievers.similar_documents_retriever.SimilarDocumentsRetriever",
            "init_parameters": {},
        }
        with pytest.raises(DeserializationError, match="Missing 'retrievers' in serialization data"):
            SimilarDocumentsRetriever.from_dict(data)

    def test_from_dict_without_retriever_type(self):
        data = {
            "type": "haystack.components.retrievers.similar_documents_retriever.SimilarDocumentsRetriever",
            "init_parameters": {"retrievers": [{"init_parameters": {}}]},
        }
        with pytest.raises(DeserializationError, match="Missing 'type' in retriever's serialization data"):
            SimilarDocumentsRetriever.from_dict(data)

    def test_from_dict_nonexisting_retriever(self):
        data = {
            "type": "haystack.components.retrievers.similar_documents_retriever.SimilarDocumentsRetriever",
            "init_parameters": {"retrievers": [{"type": "Nonexisting.Retriever", "init_parameters": {}}]},
        }
        with pytest.raises(DeserializationError):
            SimilarDocumentsRetriever.from_dict(data)

    @pytest.mark.parametrize("r1_top_k, r2_top_k", [(1, 1), (2, 1), (2, 2)])
    def test_retriever_valid_run(self, sample_document_store, r1_top_k, r2_top_k):
        retriever1 = InMemoryBM25Retriever(document_store=sample_document_store, top_k=r1_top_k)
        retriever2 = InMemoryBM25Retriever(document_store=sample_document_store, top_k=r2_top_k)
        similar_docs_retriever = SimilarDocumentsRetriever(retrievers=[retriever1, retriever2])

        result = similar_docs_retriever.run(
            [Document(content="A document about DNA"), Document(content="Java is a popular programming language")]
        )

        assert "document_lists" in result
        doc_lists = result["document_lists"]

        # [retriever1-doc1, retriever1-doc2, retriever2-doc1, retriever2-doc2]
        assert len(doc_lists) == 4

        assert len(doc_lists[0]) == r1_top_k
        assert all("DNA" in d.content for d in doc_lists[0])

        assert len(doc_lists[1]) == r1_top_k
        assert all("programming" in d.content for d in doc_lists[1])

        assert len(doc_lists[2]) == r2_top_k
        assert all("DNA" in d.content for d in doc_lists[2])

        assert len(doc_lists[3]) == r2_top_k
        assert all("programming" in d.content for d in doc_lists[3])

    @pytest.mark.integration
    @pytest.mark.parametrize("r1_top_k, r2_top_k", [(1, 1), (2, 1), (2, 2)])
    def test_retriever_in_pipeline_valid_run(self, sample_document_store, r1_top_k, r2_top_k):
        retriever1 = InMemoryBM25Retriever(document_store=sample_document_store, top_k=r1_top_k)
        retriever2 = InMemoryBM25Retriever(document_store=sample_document_store, top_k=r2_top_k)
        similar_docs_retriever = SimilarDocumentsRetriever(retrievers=[retriever1, retriever2])

        pipeline = Pipeline()
        pipeline.add_component("sim_docs_retriever", similar_docs_retriever)

        result: Dict[str, Any] = pipeline.run(
            data={
                "documents": [
                    Document(content="A document about DNA"),
                    Document(content="Java is a popular programming language"),
                ]
            }
        )

        assert result
        assert "sim_docs_retriever" in result

        sim_docs_result = result["sim_docs_retriever"]
        assert "document_lists" in sim_docs_result
        doc_lists = sim_docs_result["document_lists"]

        # [retriever1-doc1, retriever1-doc2, retriever2-doc1, retriever2-doc2]
        assert len(doc_lists) == 4

        assert len(doc_lists[0]) == r1_top_k
        assert all("DNA" in d.content for d in doc_lists[0])

        assert len(doc_lists[1]) == r1_top_k
        assert all("programming" in d.content for d in doc_lists[1])

        assert len(doc_lists[2]) == r2_top_k
        assert all("DNA" in d.content for d in doc_lists[2])

        assert len(doc_lists[3]) == r2_top_k
        assert all("programming" in d.content for d in doc_lists[3])
