# pylint: disable=too-many-public-methods
import sys

import pytest
import numpy as np

from haystack.schema import Document, Label, Answer, Span
from haystack.errors import DuplicateDocumentError
from haystack.document_stores import BaseDocumentStore


@pytest.mark.document_store
class DocumentStoreBaseTestAbstract:
    """
    This is a base class to test abstract methods from DocumentStoreBase to be inherited by any Document Store
    testsuite. It doesn't have the `Test` prefix in the name so that its methods won't be collected for this
    class but only for its subclasses.
    """

    @pytest.fixture
    def documents(self):
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={"name": f"name_{i}", "year": "2020", "month": "01", "numbers": [2, 4]},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={"name": f"name_{i}", "year": "2021", "month": "02", "numbers": [-2, -4]},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    content=f"Document {i} without embeddings",
                    meta={"name": f"name_{i}", "no_embedding": True, "month": "03"},
                )
            )

        return documents

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
    # Integration tests
    #

    @pytest.mark.integration
    def test_write_documents(self, ds, documents):
        ds.write_documents(documents)
        docs = ds.get_all_documents()
        assert len(docs) == len(documents)
        expected_ids = set(doc.id for doc in documents)
        ids = set(doc.id for doc in docs)
        assert ids == expected_ids

    @pytest.mark.integration
    def test_write_labels(self, ds, labels):
        ds.write_labels(labels)
        assert ds.get_all_labels() == labels

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

    @pytest.mark.integration
    def test_get_embedding_count(self, ds, documents):
        """
        We expect 6 docs with embeddings because only 6 documents in the documents fixture for this class contain
        embeddings.
        """
        ds.write_documents(documents)
        assert ds.get_embedding_count() == 6

    @pytest.mark.skip
    @pytest.mark.integration
    def test_get_all_documents_without_filters(self, ds, documents):
        ds.write_documents(documents)
        out = ds.get_all_documents()
        assert out == documents

    @pytest.mark.integration
    def test_get_all_documents_without_embeddings(self, ds, documents):
        ds.write_documents(documents)
        out = ds.get_all_documents(return_embedding=False)
        for doc in out:
            assert doc.embedding is None

    @pytest.mark.integration
    def test_get_all_document_filter_duplicate_text_value(self, ds):
        documents = [
            Document(content="duplicated", meta={"meta_field": "0"}, id_hash_keys=["meta"]),
            Document(content="duplicated", meta={"meta_field": "1", "name": "file.txt"}, id_hash_keys=["meta"]),
            Document(content="Doc2", meta={"name": "file_2.txt"}, id_hash_keys=["meta"]),
        ]
        ds.write_documents(documents)
        documents = ds.get_all_documents(filters={"meta_field": ["1"]})
        assert len(documents) == 1
        assert documents[0].content == "duplicated"
        assert documents[0].meta["name"] == "file.txt"

        documents = ds.get_all_documents(filters={"meta_field": ["0"]})
        assert len(documents) == 1
        assert documents[0].content == "duplicated"
        assert documents[0].meta.get("name") is None

        documents = ds.get_all_documents(filters={"name": ["file_2.txt"]})
        assert len(documents) == 1
        assert documents[0].content == "Doc2"
        assert documents[0].meta.get("meta_field") is None

    @pytest.mark.integration
    def test_get_all_documents_with_correct_filters(self, ds, documents):
        ds.write_documents(documents)
        result = ds.get_all_documents(filters={"year": ["2020"]})
        assert len(result) == 3

        documents = ds.get_all_documents(filters={"year": ["2020", "2021"]})
        assert len(documents) == 6

    @pytest.mark.integration
    def test_get_all_documents_with_incorrect_filter_name(self, ds, documents):
        ds.write_documents(documents)
        result = ds.get_all_documents(filters={"non_existing_meta_field": ["whatever"]})
        assert len(result) == 0

    @pytest.mark.integration
    def test_get_all_documents_with_incorrect_filter_value(self, ds, documents):
        ds.write_documents(documents)
        result = ds.get_all_documents(filters={"year": ["nope"]})
        assert len(result) == 0

    @pytest.mark.integration
    def test_eq_filters(self, ds, documents):
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"year": {"$eq": "2020"}})
        assert len(result) == 3
        result = ds.get_all_documents(filters={"year": "2020"})
        assert len(result) == 3

    @pytest.mark.integration
    def test_in_filters(self, ds, documents):
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"year": {"$in": ["2020", "2021", "n.a."]}})
        assert len(result) == 6
        result = ds.get_all_documents(filters={"year": ["2020", "2021", "n.a."]})
        assert len(result) == 6

    @pytest.mark.integration
    def test_ne_filters(self, ds, documents):
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"year": {"$ne": "2020"}})
        assert len(result) == 6

    @pytest.mark.integration
    def test_nin_filters(self, ds, documents):
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"year": {"$nin": ["2020", "2021", "n.a."]}})
        assert len(result) == 3

    @pytest.mark.integration
    def test_comparison_filters(self, ds, documents):
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"numbers": {"$gt": 0.0}})
        assert len(result) == 3

        result = ds.get_all_documents(filters={"numbers": {"$gte": -2.0}})
        assert len(result) == 6

        result = ds.get_all_documents(filters={"numbers": {"$lt": 0.0}})
        assert len(result) == 3

        result = ds.get_all_documents(filters={"numbers": {"$lte": 2.0}})
        assert len(result) == 6

    @pytest.mark.integration
    def test_compound_filters(self, ds, documents):
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"year": {"$lte": "2021", "$gte": "2020"}})
        assert len(result) == 6

    @pytest.mark.integration
    def test_simplified_filters(self, ds, documents):
        ds.write_documents(documents)

        filters = {"$and": {"year": {"$lte": "2021", "$gte": "2020"}, "name": {"$in": ["name_0", "name_1"]}}}
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 4

        filters_simplified = {"year": {"$lte": "2021", "$gte": "2020"}, "name": ["name_0", "name_1"]}
        result = ds.get_all_documents(filters=filters_simplified)
        assert len(result) == 4

    @pytest.mark.integration
    def test_nested_condition_filters(self, ds, documents):
        ds.write_documents(documents)
        filters = {
            "$and": {
                "year": {"$lte": "2021", "$gte": "2020"},
                "$or": {"name": {"$in": ["name_0", "name_1"]}, "numbers": {"$lt": 5.0}},
            }
        }
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 6

        filters_simplified = {
            "year": {"$lte": "2021", "$gte": "2020"},
            "$or": {"name": {"$in": ["name_0", "name_2"]}, "numbers": {"$lt": 5.0}},
        }
        result = ds.get_all_documents(filters=filters_simplified)
        assert len(result) == 6

        filters = {
            "$and": {
                "year": {"$lte": "2021", "$gte": "2020"},
                "$or": {
                    "name": {"$in": ["name_0", "name_1"]},
                    "$and": {"numbers": {"$lt": 5.0}, "$not": {"month": {"$eq": "01"}}},
                },
            }
        }
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 5

        filters_simplified = {
            "year": {"$lte": "2021", "$gte": "2020"},
            "$or": {"name": ["name_0", "name_1"], "$and": {"numbers": {"$lt": 5.0}, "$not": {"month": {"$eq": "01"}}}},
        }
        result = ds.get_all_documents(filters=filters_simplified)
        assert len(result) == 5

    @pytest.mark.integration
    def test_nested_condition_not_filters(self, ds, documents):
        """
        Test nested logical operations within "$not", important as we apply De Morgan's laws in WeaviateDocumentstore
        """
        ds.write_documents(documents)
        filters = {
            "$not": {
                "$or": {
                    "$and": {"numbers": {"$lt": 5.0}, "month": {"$ne": "01"}},
                    "$not": {"year": {"$lte": "2021", "$gte": "2020"}},
                }
            }
        }
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 3

        docs_meta = result[0].meta["numbers"]
        assert [2, 4] == docs_meta

        # Test same logical operator twice on same level

        filters = {
            "$or": [
                {"$and": {"name": {"$in": ["name_0", "name_1"]}, "year": {"$gte": "2020"}}},
                {"$and": {"name": {"$in": ["name_0", "name_1"]}, "year": {"$lt": "2021"}}},
            ]
        }
        result = ds.get_all_documents(filters=filters)
        docs_meta = [doc.meta["name"] for doc in result]
        assert len(result) == 4
        assert "name_0" in docs_meta
        assert "name_2" not in docs_meta

    @pytest.mark.integration
    def test_get_document_by_id(self, ds, documents):
        ds.write_documents(documents)
        doc = ds.get_document_by_id(documents[0].id)
        assert doc.id == documents[0].id
        assert doc.content == documents[0].content

    @pytest.mark.integration
    def test_get_documents_by_id(self, ds, documents):
        ds.write_documents(documents)
        ids = [doc.id for doc in documents]
        result = {doc.id for doc in ds.get_documents_by_id(ids, batch_size=2)}
        assert set(ids) == result

    @pytest.mark.integration
    def test_get_document_count(self, ds, documents):
        ds.write_documents(documents)
        assert ds.get_document_count() == len(documents)
        assert ds.get_document_count(filters={"year": ["2020"]}) == 3
        assert ds.get_document_count(filters={"month": ["02"]}) == 3

    @pytest.mark.integration
    def test_get_all_documents_generator(self, ds, documents):
        ds.write_documents(documents)
        assert len(list(ds.get_all_documents_generator(batch_size=2))) == 9

    @pytest.mark.integration
    def test_duplicate_documents_skip(self, ds, documents):
        ds.write_documents(documents)

        updated_docs = []
        for d in documents:
            updated_d = Document.from_dict(d.to_dict())
            updated_d.meta["name"] = "Updated"
            updated_docs.append(updated_d)

        ds.write_documents(updated_docs, duplicate_documents="skip")
        for d in ds.get_all_documents():
            assert d.meta.get("name") != "Updated"

    @pytest.mark.integration
    def test_duplicate_documents_overwrite(self, ds, documents):
        ds.write_documents(documents)

        updated_docs = []
        for d in documents:
            updated_d = Document.from_dict(d.to_dict())
            updated_d.meta["name"] = "Updated"
            updated_docs.append(updated_d)

        ds.write_documents(updated_docs, duplicate_documents="overwrite")
        for doc in ds.get_all_documents():
            assert doc.meta["name"] == "Updated"

    @pytest.mark.integration
    def test_duplicate_documents_fail(self, ds, documents):
        ds.write_documents(documents)

        updated_docs = []
        for d in documents:
            updated_d = Document.from_dict(d.to_dict())
            updated_d.meta["name"] = "Updated"
            updated_docs.append(updated_d)

        with pytest.raises(DuplicateDocumentError):
            ds.write_documents(updated_docs, duplicate_documents="fail")

    @pytest.mark.integration
    def test_write_document_meta(self, ds):
        ds.write_documents(
            [
                {"content": "dict_without_meta", "id": "1"},
                {"content": "dict_with_meta", "meta_field": "test2", "id": "2"},
                Document(content="document_object_without_meta", id="3"),
                Document(content="document_object_with_meta", meta={"meta_field": "test4"}, id="4"),
            ]
        )
        assert not ds.get_document_by_id("1").meta
        assert ds.get_document_by_id("2").meta["meta_field"] == "test2"
        assert not ds.get_document_by_id("3").meta
        assert ds.get_document_by_id("4").meta["meta_field"] == "test4"

    @pytest.mark.integration
    def test_delete_documents(self, ds, documents):
        ds.write_documents(documents)
        ds.delete_documents()
        assert ds.get_document_count() == 0

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
        ds.delete_documents(ids=[doc.id for doc in docs_to_delete], filters={"name": ["name_0"]})
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
        ds.delete_labels(filters={"query": "query_1"})
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
    def test_get_label_count(self, ds, labels):
        ds.write_labels(labels)
        assert ds.get_label_count() == len(labels)

    @pytest.mark.integration
    def test_delete_index(self, ds, documents):
        ds.write_documents(documents, index="custom_index")
        assert ds.get_document_count(index="custom_index") == len(documents)
        ds.delete_index(index="custom_index")
        with pytest.raises(Exception):
            ds.get_document_count(index="custom_index")

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
    @pytest.mark.skipif(sys.platform == "win32", reason="_get_documents_meta() fails with 'too many SQL variables'")
    def test_get_all_documents_large_quantities(self, ds):
        # Test to exclude situations like Weaviate not returning more than 100 docs by default
        #   https://github.com/deepset-ai/haystack/issues/1893
        docs_to_write = [
            {"meta": {"name": f"name_{i}"}, "content": f"text_{i}", "embedding": np.random.rand(768).astype(np.float32)}
            for i in range(1000)
        ]
        ds.write_documents(docs_to_write)
        documents = ds.get_all_documents()
        assert all(isinstance(d, Document) for d in documents)
        assert len(documents) == len(docs_to_write)

    @pytest.mark.integration
    def test_custom_embedding_field(self, ds):
        ds.embedding_field = "custom_embedding_field"
        doc_to_write = {"content": "test", "custom_embedding_field": np.random.rand(768).astype(np.float32)}
        ds.write_documents([doc_to_write])
        documents = ds.get_all_documents(return_embedding=True)
        assert len(documents) == 1
        assert documents[0].content == "test"
        # Some document stores normalize the embedding on save, let's just compare the length
        assert doc_to_write["custom_embedding_field"].shape == documents[0].embedding.shape

    #
    # Unit tests
    #

    @pytest.mark.unit
    def test_normalize_embeddings_diff_shapes(self):
        VEC_1 = np.array([0.1, 0.2, 0.3], dtype="float32")
        BaseDocumentStore.normalize_embedding(VEC_1)
        assert np.linalg.norm(VEC_1) - 1 < 0.01

        VEC_1 = np.array([0.1, 0.2, 0.3], dtype="float32").reshape(1, -1)
        BaseDocumentStore.normalize_embedding(VEC_1)
        assert np.linalg.norm(VEC_1) - 1 < 0.01
