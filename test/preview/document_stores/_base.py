import pytest

import numpy as np

from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import MissingDocumentError, DuplicateDocumentError


class DocumentStoreBaseTests:
    @pytest.fixture
    def docstore(self):
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
                    content=f"A Foobar Document {i}",
                    metadata={"name": f"name_{i}", "year": "2000", "month": "03", "number": -10},
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

    def test_filter_simple_value(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": "2020"})
        assert len(result) == 3

    def test_filter_simple_list(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": ["2020"]})
        assert all(doc.metadata["year"] == "2020" for doc in result)
        result = docstore.filter_documents(filters={"year": ["2020", "2021"]})
        assert all(doc.metadata["year"] in ["2020", "2021"] for doc in result)

    def test_incorrect_filter_name(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"non_existing_meta_field": ["whatever"]})
        assert len(result) == 0

    def test_incorrect_filter_type(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        with pytest.raises(ValueError, match="dictionaries or lists"):
            docstore.filter_documents(filters="something odd")

    def test_incorrect_filter_value(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": ["nope"]})
        assert len(result) == 0

    def test_incorrect_filter_nesting(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        with pytest.raises(ValueError, match="malformed"):
            docstore.filter_documents(filters={"number": {"year": "2020"}})
        with pytest.raises(ValueError, match="malformed"):
            docstore.filter_documents(filters={"number": {"year": {"month": "01"}}})

    def test_eq_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": {"$eq": "2020"}})
        assert all(doc.metadata["year"] == "2020" for doc in result)
        result = docstore.filter_documents(filters={"year": "2020"})
        assert all(doc.metadata["year"] == "2020" for doc in result)

    def test_in_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": {"$in": ["2020", "2021", "n.a."]}})
        assert all(doc.metadata["year"] in ["2020", "2021"] for doc in result)
        result = docstore.filter_documents(filters={"year": ["2020", "2021", "n.a."]})
        assert all(doc.metadata["year"] in ["2020", "2021"] for doc in result)

    def test_ne_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": {"$ne": "2020"}})
        assert all(doc.metadata.get("year", None) != "2020" for doc in result)

    def test_nin_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": {"$nin": ["2020", "2021", "n.a."]}})
        assert all(doc.metadata.get("year", None) not in ["2020", "2021"] for doc in result)

    def test_gt_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$gt": 0.0}})
        assert all(doc.metadata["number"] > 0 for doc in result)

    def test_gte_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$gte": -2.0}})
        assert all(doc.metadata["number"] >= -2.0 for doc in result)

    def test_lt_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lt": 0.0}})
        assert all(doc.metadata["number"] < 0 for doc in result)

    def test_lte_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lte": 2.0}})
        assert all(doc.metadata["number"] <= 2.0 for doc in result)

    def test_filter_simple_explicit_and(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": {"$and": {"$lte": "2021", "$gte": "2020"}}})
        assert all(int(doc.metadata["year"]) >= 2020 and int(doc.metadata["year"]) <= 2021 for doc in result)
        result = docstore.filter_documents(filters={"year": {"$and": [{"$lte": "2021"}, {"$gte": "2020"}]}})
        assert all(int(doc.metadata["year"]) >= 2020 and int(doc.metadata["year"]) <= 2021 for doc in result)

    def test_filter_simple_implicit_and(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"year": {"$lte": "2021", "$gte": "2020"}})
        assert all(int(doc.metadata["year"]) >= 2020 and int(doc.metadata["year"]) <= 2021 for doc in result)

    def test_filter_nested_explicit_and(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters = {"$and": {"year": {"$and": {"$lte": "2021", "$gte": "2020"}}, "name": {"$in": ["name_0", "name_1"]}}}
        result = docstore.filter_documents(filters=filters)
        assert all(
            int(doc.metadata["year"]) >= 2020
            and int(doc.metadata["year"]) <= 2021
            and doc.metadata["name"] in ["name_0", "name_1"]
            for doc in result
        )

    def test_filter_nested_implicit_and(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters_simplified = {"year": {"$lte": "2021", "$gte": "2020"}, "name": ["name_0", "name_1"]}
        result = docstore.filter_documents(filters=filters_simplified)
        assert all(
            int(doc.metadata["year"]) >= 2020
            and int(doc.metadata["year"]) <= 2021
            and doc.metadata["name"] in ["name_0", "name_1"]
            for doc in result
        )

    def test_filter_simple_or(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters = {"$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}}}
        result = docstore.filter_documents(filters=filters)
        assert all(doc.metadata["name"] in ["name_0", "name_1"] or doc.metadata["number"] < 1.0 for doc in result)

    def test_filter_nested_or(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters = {"$or": {"name": {"$or": [{"$eq": "name_0"}, {"$eq": "name_1"}]}, "number": {"$lt": 1.0}}}
        result = docstore.filter_documents(filters=filters)
        assert all(doc.metadata["name"] in ["name_0", "name_1"] or doc.metadata["number"] < 1.0 for doc in result)

    def test_filter_nested_and_or(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters_simplified = {
            "year": {"$lte": "2021", "$gte": "2020"},
            "$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}},
        }
        result = docstore.filter_documents(filters=filters_simplified)
        assert all(
            (int(doc.metadata["year"]) >= 2020 and int(doc.metadata["year"]) <= 2021)
            and (doc.metadata["name"] in ["name_0", "name_1"] or doc.metadata["number"] < 1.0)
            for doc in result
        )

    def test_filter_nested_or_and(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters_simplified = {
            "$or": {
                "number": {"$lt": 1.0},
                "$and": {"name": {"$in": ["name_0", "name_1"]}, "$not": {"month": {"$eq": "01"}}},
            }
        }
        result = docstore.filter_documents(filters=filters_simplified)
        assert all(
            doc.metadata.get("number", 2) < 1.0
            or (doc.metadata["name"] in ["name_0", "name_1"] and doc.metadata["month"] != "01")
            for doc in result
        )

    def test_filter_nested_multiple_identical_operators_same_level(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters = {
            "$or": [
                {"$and": {"name": {"$in": ["name_0", "name_1"]}, "year": {"$gte": "2020"}}},
                {"$and": {"name": {"$in": ["name_0", "name_1"]}, "year": {"$lt": "2021"}}},
            ]
        }
        result = docstore.filter_documents(filters=filters)
        assert all(doc.metadata["name"] in ["name_0", "name_1"] for doc in result)

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
