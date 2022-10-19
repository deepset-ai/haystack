import pytest
import numpy as np

from haystack.schema import Document, Label, Answer


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
                    query="query",
                    document=d,
                    is_correct_document=True,
                    is_correct_answer=False,
                    # create a mix set of labels
                    origin="user-feedback" if i % 2 else "gold-label",
                    answer=None if not i else Answer(f"the answer is {i}"),
                )
            )
        return labels

    @pytest.mark.integration
    def test_write_documents(self, ds, documents):
        ds.write_documents(documents)
        docs = ds.get_all_documents()
        assert len(docs) == len(documents)
        for i, doc in enumerate(docs):
            expected = documents[i]
            assert doc.id == expected.id

    @pytest.mark.integration
    def test_write_labels(self, ds, labels):
        ds.write_labels(labels)
        assert ds.get_all_labels() == labels

    @pytest.mark.integration
    def test_write_with_duplicate_doc_ids(self, ds):
        duplicate_documents = [
            Document(content="Doc1", id_hash_keys=["content"]),
            Document(content="Doc1", id_hash_keys=["content"]),
        ]
        ds.write_documents(duplicate_documents, duplicate_documents="skip")
        assert len(ds.get_all_documents()) == 1
        with pytest.raises(Exception):
            ds.write_documents(duplicate_documents, duplicate_documents="fail")

    @pytest.mark.skip
    @pytest.mark.integration
    def test_get_all_documents_without_filters(self, ds, documents):
        ds.write_documents(documents)
        out = ds.get_all_documents()
        assert out == documents

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
    def test_extended_filter(self, ds, documents):
        ds.write_documents(documents)

        # Test comparison operators individually

        result = ds.get_all_documents(filters={"year": {"$eq": "2020"}})
        assert len(result) == 3
        result = ds.get_all_documents(filters={"year": "2020"})
        assert len(result) == 3

        result = ds.get_all_documents(filters={"year": {"$in": ["2020", "2021", "n.a."]}})
        assert len(result) == 6
        result = ds.get_all_documents(filters={"year": ["2020", "2021", "n.a."]})
        assert len(result) == 6

        result = ds.get_all_documents(filters={"year": {"$ne": "2020"}})
        assert len(result) == 6

        result = ds.get_all_documents(filters={"year": {"$nin": ["2020", "2021", "n.a."]}})
        assert len(result) == 3

        result = ds.get_all_documents(filters={"numbers": {"$gt": 0}})
        assert len(result) == 3

        result = ds.get_all_documents(filters={"numbers": {"$gte": -2}})
        assert len(result) == 6

        result = ds.get_all_documents(filters={"numbers": {"$lt": 0}})
        assert len(result) == 3

        result = ds.get_all_documents(filters={"numbers": {"$lte": 2.0}})
        assert len(result) == 6

        # Test compound filters

        result = ds.get_all_documents(filters={"year": {"$lte": "2021", "$gte": "2020"}})
        assert len(result) == 6

        filters = {"$and": {"year": {"$lte": "2021", "$gte": "2020"}, "name": {"$in": ["name_0", "name_1"]}}}
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 4

        filters_simplified = {"year": {"$lte": "2021", "$gte": "2020"}, "name": ["name_0", "name_1"]}
        result = ds.get_all_documents(filters=filters_simplified)
        assert len(result) == 4

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

        # Test nested logical operations within "$not", important as we apply De Morgan's laws in WeaviateDocumentstore

        filters = {
            "$not": {
                "$or": {
                    "$and": {"numbers": {"$lt": 5.0}, "month": {"$ne": "01"}},
                    "$not": {"year": {"$lte": "2021", "$gte": "2020"}},
                }
            }
        }
        result = ds.get_all_documents(filters=filters)
        docs_meta = result[0].meta["numbers"]
        assert len(result) == 3
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

    # get_all_documents_generator
    # get_all_labels
    # get_document_by_id
    # get_document_count
    # query_by_embedding
    # get_label_count
    # write_labels
    # delete_documents
    # delete_labels
    # delete_index
    # _create_document_field_map
    # get_documents_by_id
    # update_document_meta
