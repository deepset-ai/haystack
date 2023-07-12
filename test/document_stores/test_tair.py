from typing import List, Union, Dict, Any
from sentence_transformers import SentenceTransformer

import os
import numpy as np

import pytest

from haystack.document_stores.tairvector import TairDocumentStore
from haystack.schema import Document, Label, Answer, Span
from haystack.testing import DocumentStoreBaseTestAbstract

class TestTairDocumentStore(DocumentStoreBaseTestAbstract):
    # Fixtures
    index = "haystack_tests"
    url = os.environ.get("TAIR_VECTOR")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    @pytest.fixture
    def ds(self):
        return TairDocumentStore(
            url=self.url,
            embedding_dim=384,
            embedding_field="embedding",
            index=self.index,
            similarity="COSINE",
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
            Document(content="My name is Ahmed and I live in Cairo",
                     embedding=self.model.encode("My name is Ahmed and I live in Cairo")),
            Document(content="My name is Bruce and I live in Gotham",
                     embedding=self.model.encode("My name is Bruce and I live in Gotham")),
            Document(content="My name is Peter and I live in Quahog",
                     embedding=self.model.encode("My name is Peter and I live in  Qindao")),
        ]

    @pytest.fixture
    def documents(self, docs_all_formats: List[Union[Document, Dict[str, Any]]]) -> List[Document]:
        return [Document.from_dict(doc) if isinstance(doc, dict) else doc for doc in docs_all_formats]
    @pytest.fixture
    def labels(self, documents):
        labels = []
        for i, d in enumerate(documents):
            labels.append(
                Label(
                    query=f"query_{i}",
                    document=d,
                    is_correct_document=True,
                    is_correct_answer=False,
                    # create a mix set of labels
                    origin="user-feedback" if i % 2 else "gold-label",
                    answer=None if not i else Answer(f"the answer is {i}", document_ids=[d.id]),
                    meta={"name": f"label_{i}", "year": f"{2020 + i}"},
                )
            )
        return labels

    #
    #  Tests
    #
    @pytest.mark.integration
    def test_ne_filters(self, ds, documents):
        ds.write_documents(documents, self.index)

        result = ds.get_all_documents(index=self.index, filters={"year": {"$ne": "2020"}})
        assert len(result) == 3

    @pytest.mark.integration
    def test_eq_filters(self, ds, documents):
        ds.write_documents(documents, self.index)

        result = ds.get_all_documents(index=self.index)
        assert len(result) == 9
        result = ds.get_all_documents(index=self.index, filters={"year": {"$eq": "2020"}})
        assert len(result) == 3
        result = ds.get_all_documents(index=self.index, filters={"numeric_field": {"$eq": 5.0}})
        assert len(result) == 1

    @pytest.mark.integration
    def test_gte_lt_filters(self, ds, documents):
        ds.write_documents(documents, self.index)

        result = ds.get_all_documents(filters={"numeric_field": {"$gte": 5.0}})
        assert len(result) == 2

        result = ds.get_all_documents(filters={"numeric_field": {"$lt": 5.0}})
        assert len(result) == 4

    @pytest.mark.integration
    def test_lte_gt_filters(self, ds, documents):
        ds.write_documents(documents, self.index)

        result = ds.get_all_documents(filters={"numeric_field": {"$lte": 5.0}})
        assert len(result) == 5

        result = ds.get_all_documents(filters={"numeric_field": {"$gt": 5.0}})
        assert len(result) == 1

    def test_and_or_not_filters(self, ds, documents):
        ds.write_documents(documents, self.index)

        filters = {
            "$not": {
                "$or": {
                    "$and": {"numeric_field": {"$gt": 3.0}, "meta_field": {"$ne": "test-3"}},
                    "$not": {"date": {"$ne": "2019-01-01"}},
                }
            }
        }
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 2

    @pytest.mark.integration
    def test_nin_filters(self, ds, documents):
        ds.write_documents(documents, self.index)

        result = ds.get_all_documents(filters={"year": {"$nin": ["2020", "2021"]}})
        assert len(result) == 0

        result = ds.get_all_documents(filters={"year": {"$in": ["2020", "2021"]}})
        assert len(result) == 6

    @pytest.mark.integration
    def test_get_label_count(self, ds, labels):
        with pytest.raises(NotImplementedError):
            ds.get_label_count()

    @pytest.mark.integration
    def test_query_by_embedding(self, ds, documents):
        ds.write_documents(documents, self.index)

        query_context = "Which city does Adele live in?"
        query_emb = self.model.encode(query_context)

        result = ds.query_by_embedding(query_emb=query_emb, filters=None, top_k=3, index=self.index)
        assert len(result) == 3
        assert result[0].content == "My name is Adele and I live in London"

    @pytest.mark.integration
    def test_get_embedding_count(self, ds, documents):
        """
        We expect 9 docs with embeddings because all documents in the documents fixture for this class contain
        embeddings.
        """
        ds.write_documents(documents)
        assert ds.get_embedding_count() == 9

    @pytest.mark.integration
    def test_write_labels(self, ds, labels):
        ds.write_labels(labels, self.index)
        ds.get_all_labels()
        assert len(ds.get_all_labels()) == len(labels)

    @pytest.mark.integration
    def test_write_with_duplicate_doc_ids(self, ds):
        duplicate_documents = [
            Document(content="Doc1", id_hash_keys=["content"], meta={"key1": "value1"}),
            Document(content="Doc1", id_hash_keys=["content"], meta={"key1": "value1"}),
        ]
        ds.write_documents(duplicate_documents, duplicate_documents="skip")
        results = ds.get_all_documents()
        assert len(results) == 1
        assert results[0] == duplicate_documents[0]
        with pytest.raises(Exception):
            ds.write_documents(duplicate_documents, duplicate_documents="fail")

    @pytest.mark.skip
    @pytest.mark.integration
    def test_get_all_documents_without_filters(self, ds, documents):
        ds.write_documents(documents)
        out = ds.get_all_documents()
        assert out == documents

    @pytest.mark.integration
    def test_comparison_filters(self, ds, documents):
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"numeric_field": {"$gt": 0.0}})
        assert len(result) == 4

        result = ds.get_all_documents(filters={"numeric_field": {"$gte": -2.0}})
        assert len(result) == 6

        result = ds.get_all_documents(filters={"numeric_field": {"$lt": 0.0}})
        assert len(result) == 0

        result = ds.get_all_documents(filters={"numeric_field": {"$lte": 2.0}})
        assert len(result) == 2

    @pytest.mark.integration
    def test_compound_filters(self, ds, documents):
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"numeric_field": {"$lte": 5.5, "$gte": 0.0}})
        assert len(result) == 6

    @pytest.mark.integration
    def test_simplified_filters(self, ds, documents):
        ds.write_documents(documents)

        filters = {"$and": {"numeric_field": {"$lte": 5.5, "$gte": 3.0}, "name": {"$in": ["file_2.txt", "file_5.txt"]}}}
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 1

        filters_simplified = {"numeric_field": {"$lte": 5.5, "$gte": 3.0}, "name": ["file_2.txt", "file_5.txt"]}
        result = ds.get_all_documents(filters=filters_simplified)
        assert len(result) == 1

    @pytest.mark.integration
    def test_nested_condition_filters(self, ds, documents):
        ds.write_documents(documents)
        filters = {
            "$and": {
                "numeric_field": {"$lte": 5.0, "$gte": 3.0},
                "$or": {"name": {"$in": ["file_2.txt", "file_3.txt"]}, "year": {"$eq": "2020"}},
            }
        }
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 2

        filters_simplified = {
            "numeric_field": {"$lte": 5.0, "$gte": 3.0},
            "$or": {"name": {"$in": ["file_2.txt", "file_3.txt"]}, "year": {"$eq": "2020"}},
        }
        result = ds.get_all_documents(filters=filters_simplified)
        assert len(result) == 2

        filters = {
            "$and": {
                "numeric_field": {"$lte": 5.0, "$gte": 3.0},
                "$or": {
                    "name": {"$in": ["file_2.txt", "file_4.txt"]},
                    "$and": {"year": {"$eq": "2020"}, "$not": {"meta_field": {"$eq": "test-3"}}},
                },
            }
        }
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 1

        filters_simplified = {
            "numeric_field": {"$lte": 5.0, "$gte": 3.0},
            "$or": {"name": ["file_2.txt", "file_4.txt"], "$and": {"year": {"$eq": "2020"}, "$not": {"meta_field": {"$eq": "test-3"}}}},
        }
        result = ds.get_all_documents(filters=filters_simplified)
        assert len(result) == 1

    @pytest.mark.integration
    def test_nested_condition_not_filters(self, ds, documents):
        """
        Test nested logical operations within "$not", important as we apply De Morgan's laws in WeaviateDocumentstore
        """
        ds.write_documents(documents)
        filters = {
            "$not": {
                "$or": {
                    "$and": {"numeric_field": {"$lt": 5.0}, "date": {"$ne": "2018-10-01"}},
                    "$not": {"year": {"$eq": "2021"}},
                }
            }
        }
        result = ds.get_all_documents(filters=filters)
        docs_meta = [doc.meta["meta_field"] for doc in result]
        assert len(result) == 2
        assert "test-1" in docs_meta
        assert "test-4" not in docs_meta

        # Test same logical operator twice on same level

        filters = {
            "$or": [
                {"$and": {"name": {"$in": ["file_1.txt", "file_2.txt"]}, "year": {"$eq": "2021"}}},
                {"$and": {"name": {"$in": ["file_3.txt", "file_4.txt"]}, "year": {"$ne": "2021"}}},
            ]
        }
        result = ds.get_all_documents(filters=filters)
        docs_meta = [doc.meta["meta_field"] for doc in result]
        assert len(result) == 4
        assert "test-1" in docs_meta
        assert "test-5" not in docs_meta

    @pytest.mark.integration
    def test_get_document_count(self, ds, documents):
        ds.write_documents(documents)
        assert ds.get_document_count() == len(documents)
        assert ds.get_document_count(filters={"year": ["2020"]}) == 3
        assert ds.get_document_count(filters={"month": ["02"]}) == 3

    @pytest.mark.integration
    def test_delete_documents_with_filters(self, ds, documents):
        ds.write_documents(documents)
        ds.delete_documents(filters={"year": ["2020", "2021"]})
        documents = ds.get_all_documents()
        assert ds.get_document_count() == 3

    @pytest.mark.integration
    def test_delete_documents_by_id(self, ds, documents):
        ds.write_documents(documents)
        docs_to_delete = ds.get_all_documents(filters={"year": ["2020"]})
        ds.delete_documents(ids=[doc.id for doc in docs_to_delete])
        assert ds.get_document_count() == 6

    @pytest.mark.integration
    def test_delete_documents_by_id_with_filters(self, ds, documents):
        ds.write_documents(documents)
        docs_to_delete = ds.get_all_documents(filters={"year": ["2020"]})
        # this should delete only 1 document out of the 3 ids passed
        ds.delete_documents(ids=[doc.id for doc in docs_to_delete], filters={"name": ["file_5.txt"]})
        assert ds.get_document_count() == 8

    @pytest.mark.integration
    def test_write_get_all_labels(self, ds, labels):
        ds.write_labels(labels)
        ds.write_labels(labels[:3], index="custom_index")
        assert len(ds.get_all_labels()) == 9
        assert len(ds.get_all_labels(index="custom_index")) == 3
        # remove the index we created in this test
        ds.delete_index("custom_index")

    @pytest.mark.integration
    def test_delete_labels(self, ds, labels):
        ds.write_labels(labels)
        ds.write_labels(labels[:3], index="custom_index")
        ds.delete_labels()
        ds.delete_labels(index="custom_index")
        assert len(ds.get_all_labels()) == 0
        assert len(ds.get_all_labels(index="custom_index")) == 0
        # remove the index we created in this test
        ds.delete_index("custom_index")

    @pytest.mark.integration
    def test_write_labels_duplicate(self, ds, labels):
        # create a duplicate
        dupe = Label.from_dict(labels[0].to_dict())

        ds.write_labels(labels + [dupe])

        # ensure the duplicate was discarded
        assert len(ds.get_all_labels()) == len(labels)

    @pytest.mark.integration
    def test_delete_labels_by_id(self, ds, labels):
        ds.write_labels(labels)
        ds.delete_labels(ids=[labels[0].id])
        assert len(ds.get_all_labels()) == len(labels) - 1

    @pytest.mark.integration
    def test_delete_labels_by_filter(self, ds, labels):
        ds.write_labels(labels)
        ds.delete_labels(filters={"query": "query_0"})
        assert len(ds.get_all_labels()) == len(labels) - 1

    @pytest.mark.integration
    def test_delete_labels_by_filter_id(self, ds, labels):
        ds.write_labels(labels)

        # ids and filters are ANDed, the following should have no effect
        ds.delete_labels(ids=[labels[0].id], filters={"query": "query_9"})
        assert len(ds.get_all_labels()) == len(labels)

        #
        ds.delete_labels(ids=[labels[0].id], filters={"query": "query_0"})
        assert len(ds.get_all_labels()) == len(labels) - 1

    @pytest.mark.integration
    def test_update_meta(self, ds, documents):
        ds.write_documents(documents)
        doc = documents[0]
        ds.update_document_meta(doc.id, meta={"year": "2099", "month": "12"})
        doc = ds.get_document_by_id(doc.id)
        assert doc.meta["year"] == "2099"
        assert doc.meta["month"] == "12"

    @pytest.mark.integration
    def test_labels_with_long_texts(self, ds, documents):
        label = Label(
            query="question1",
            answer=Answer(
                answer="answer",
                type="extractive",
                score=0.0,
                context="something " * 10_000,
                offsets_in_document=[Span(start=12, end=14)],
                offsets_in_context=[Span(start=12, end=14)],
            ),
            is_correct_answer=True,
            is_correct_document=True,
            document=Document(content="something " * 10_000, id="123"),
            origin="gold-label",
        )
        ds.write_labels(labels=[label])
        labels = ds.get_all_labels()
        assert len(labels) == 1
        assert label == labels[0]

    @pytest.mark.integration
    def test_get_all_documents_large_quantities(self, ds):
        # Test to exclude situations like Weaviate not returning more than 100 docs by default
        #   https://github.com/deepset-ai/haystack/issues/1893
        docs_to_write = [
            {"meta": {"name": f"name_{i}"}, "content": f"text_{i}", "embedding": np.random.rand(384).astype(np.float32)}
            for i in range(1000)
        ]
        ds.write_documents(docs_to_write)
        documents = ds.get_all_documents()
        assert all(isinstance(d, Document) for d in documents)
        assert len(documents) == len(docs_to_write)

    @pytest.mark.integration
    def test_custom_embedding_field(self, ds):
        ds.embedding_field = "custom_embedding_field"
        doc_to_write = {"content": "test", "custom_embedding_field": np.random.rand(384).astype(np.float32)}
        ds.write_documents([doc_to_write])
        documents = ds.get_all_documents(return_embedding=True)
        assert len(documents) == 1
        assert documents[0].content == "test"
        # Some document stores normalize the embedding on save, let's just compare the length
        assert doc_to_write["custom_embedding_field"].shape == documents[0].embedding.shape

    @pytest.mark.integration
    def test_delete_index(self, ds, documents):
        ds.write_documents(documents, index=self.index)
        assert ds.get_document_count(index=self.index) == len(documents)
        ds.delete_index(index=self.index)
        with pytest.raises(Exception):
            ds.get_document_count(index=self.index)
