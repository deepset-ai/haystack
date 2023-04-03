import pytest

import numpy as np

from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import MissingDocumentError, DuplicateDocumentError


class DocumentStoreBaseTests:
    @pytest.fixture
    def docstore(self):
        raise NotImplementedError()

    def direct_access(self, docstore, doc_id):
        """
        Bypass `filter_documents()`
        """
        raise NotImplementedError()

    def direct_write(self, docstore, documents):
        """
        Bypass `write_documents()`
        """
        raise NotImplementedError()

    def direct_delete(self, docstore, ids):
        """
        Bypass `delete_documents()`
        """
        raise NotImplementedError()

    @pytest.fixture
    def filterable_docs(self):
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    metadata={"name": f"name_{i}", "year": "2020", "month": "01", "number": 2},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )
            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    metadata={"name": f"name_{i}", "year": "2021", "month": "02", "number": -2},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )
            documents.append(
                Document(
                    content=f"Document {i} without embedding",
                    metadata={"name": f"name_{i}", "no_embedding": True, "month": "03"},
                )
            )

        return documents

    def test_count_empty(self, docstore):
        assert docstore.count_documents() == 0

    def test_count_not_empty(self, docstore):
        self.direct_write(
            docstore, [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert docstore.count_documents() == 3

    def test_no_filter_empty(self, docstore):
        assert docstore.filter_documents() == []
        assert docstore.filter_documents(filters={}) == []

    def test_no_filter_not_empty(self, docstore):
        docs = [Document(content="test doc")]
        self.direct_write(docstore, docs)
        assert docstore.filter_documents() == docs
        assert docstore.filter_documents(filters={}) == docs

    def test_simple_correct_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": ["2020"]})
        assert len(result) == 3
        documents = docstore.filter_documents(filters={"year": ["2020", "2021"]})
        assert len(documents) == 6

    def test_incorrect_filter_name(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"non_existing_meta_field": ["whatever"]})
        assert len(result) == 0

    def test_incorrect_filter_value(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": ["nope"]})
        assert len(result) == 0

    def test_eq_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": {"$eq": "2020"}})
        assert len(result) == 3
        result = docstore.filter_documents(filters={"year": "2020"})
        assert len(result) == 3

    def test_in_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": {"$in": ["2020", "2021", "n.a."]}})
        assert len(result) == 6
        result = docstore.filter_documents(filters={"year": ["2020", "2021", "n.a."]})
        assert len(result) == 6

    def test_ne_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": {"$ne": "2020"}})
        assert len(result) == 6

    def test_nin_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": {"$nin": ["2020", "2021", "n.a."]}})
        assert len(result) == 3

    def test_gt_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$gt": 0.0}})
        assert len(result) == 3

    def test_gte_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$gte": -2.0}})
        assert len(result) == 6

    def test_lt_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lt": 0.0}})
        assert len(result) == 3

    def test_lte_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lte": 2.0}})
        assert len(result) == 6

    def test_compound_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)

        result = docstore.filter_documents(filters={"year": {"$lte": "2021", "$gte": "2020"}})
        assert len(result) == 6

    def test_simplified_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)

        filters = {"$and": {"year": {"$lte": "2021", "$gte": "2020"}, "name": {"$in": ["name_0", "name_1"]}}}
        result = docstore.filter_documents(filters=filters)
        assert len(result) == 4

        filters_simplified = {"year": {"$lte": "2021", "$gte": "2020"}, "name": ["name_0", "name_1"]}
        result = docstore.filter_documents(filters=filters_simplified)
        assert len(result) == 4

    def test_nested_condition_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters = {
            "$and": {
                "year": {"$lte": "2021", "$gte": "2020"},
                "$or": {"name": {"$in": ["name_0", "name_1"]}, "numbers": {"$lt": 5.0}},
            }
        }
        result = docstore.filter_documents(filters=filters)
        for doc in result:
            print(repr(doc))
        assert len(result) == 4

    def test_simplified_nested_condition_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters_simplified = {
            "year": {"$lte": "2021", "$gte": "2020"},
            "$or": {"name": {"$in": ["name_0", "name_2"]}, "numbers": {"$lt": 5.0}},
        }
        result = docstore.filter_documents(filters=filters_simplified)
        assert len(result) == 4

    def test_nested_condition_and_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters = {
            "$and": {
                "year": {"$lte": "2021", "$gte": "2020"},
                "$or": {
                    "name": {"$in": ["name_0", "name_1"]},
                    "$and": {"number": {"$lt": 5.0}, "$not": {"month": {"$eq": "01"}}},
                },
            }
        }
        result = docstore.filter_documents(filters=filters)
        assert len(result) == 5

    def test_nested_condition_or_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters_simplified = {
            "year": {"$lte": "2021", "$gte": "2020"},
            "$or": {"name": ["name_0", "name_1"], "$and": {"number": {"$lt": 5.0}, "$not": {"month": {"$eq": "01"}}}},
        }
        result = docstore.filter_documents(filters=filters_simplified)
        assert len(result) == 5

    def test_nested_condition_not_filter(self, docstore, filterable_docs):
        """
        Test nested logical operations within "$not".
        Important as we apply De Morgan's laws in WeaviateDocumentstore
        """
        self.direct_write(docstore, filterable_docs)
        filters = {
            "$not": {
                "$or": {
                    "$and": {"number": {"$lt": 5.0}, "month": {"$ne": "01"}},
                    "$not": {"year": {"$lte": "2021", "$gte": "2020"}},
                }
            }
        }
        result = docstore.filter_documents(filters=filters)
        assert len(result) == 3

        docs_meta = result[0].metadata["number"]
        assert 2 == docs_meta

    def test_nested_condition_same_operator_same_level(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters = {
            "$or": [
                {"$and": {"name": {"$in": ["name_0", "name_1"]}, "year": {"$gte": "2020"}}},
                {"$and": {"name": {"$in": ["name_0", "name_1"]}, "year": {"$lt": "2021"}}},
            ]
        }
        result = docstore.filter_documents(filters=filters)
        docs_meta = [doc.metadata["name"] for doc in result]
        assert len(result) == 4
        assert "name_0" in docs_meta
        assert "name_2" not in docs_meta

    def test_write(self, docstore):
        doc = Document(content="test doc")
        docstore.write_documents(documents=[doc])
        assert self.direct_access(docstore, doc_id=doc.id) == doc

    def test_write_duplicate_fail(self, docstore):
        doc = Document(content="test doc")
        self.direct_write(docstore, [doc])
        with pytest.raises(DuplicateDocumentError, match=f"ID '{doc.id}' already exists."):
            docstore.write_documents(documents=[doc])
        assert self.direct_access(docstore, doc_id=doc.id) == doc

    def test_write_duplicate_skip(self, docstore):
        doc = Document(content="test doc")
        self.direct_write(docstore, [doc])
        docstore.write_documents(documents=[doc], duplicates="skip")
        assert self.direct_access(docstore, doc_id=doc.id) == doc

    def test_write_duplicate_overwrite(self, docstore):
        doc1 = Document(content="test doc 1")
        doc2 = Document(content="test doc 2")
        object.__setattr__(doc2, "id", doc1.id)  # Make two docs with different content but same ID

        self.direct_write(docstore, [doc2])
        assert self.direct_access(docstore, doc_id=doc1.id) == doc2
        docstore.write_documents(documents=[doc1], duplicates="overwrite")
        assert self.direct_access(docstore, doc_id=doc1.id) == doc1

    def test_write_not_docs(self, docstore):
        with pytest.raises(ValueError, match="Please provide a list of Documents"):
            docstore.write_documents(["not a document for sure"])

    def test_write_not_list(self, docstore):
        with pytest.raises(ValueError, match="Please provide a list of Documents"):
            docstore.write_documents("not a list actually")

    def test_delete_empty(self, docstore):
        with pytest.raises(MissingDocumentError):
            docstore.delete_documents(["test"])

    def test_delete_not_empty(self, docstore):
        doc = Document(content="test doc")
        self.direct_write(docstore, [doc])

        docstore.delete_documents([doc.id])

        with pytest.raises(Exception):
            assert self.direct_access(docstore, doc_id=doc.id)

    def test_delete_not_empty_nonexisting(self, docstore):
        doc = Document(content="test doc")
        self.direct_write(docstore, [doc])

        with pytest.raises(MissingDocumentError):
            docstore.delete_documents(["non_existing"])

        assert self.direct_access(docstore, doc_id=doc.id) == doc
