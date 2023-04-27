from math import inf

import pytest
import numpy as np
import pandas as pd

from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import StoreError
from haystack.preview.document_stores import MissingDocumentError, DuplicateDocumentError


class DocumentStoreBaseTests:
    @pytest.fixture
    def docstore(self):
        raise NotImplementedError()

    @pytest.fixture
    def filterable_docs(self):
        embedding_zero = np.zeros([768, 1]).astype(np.float32)
        embedding_one = np.ones([768, 1]).astype(np.float32)

        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    metadata={"name": f"name_{i}", "page": "100", "chapter": "intro", "number": 2},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )
            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    metadata={"name": f"name_{i}", "page": "123", "chapter": "abstract", "number": -2},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )
            documents.append(
                Document(
                    content=f"A Foobar Document {i}",
                    metadata={"name": f"name_{i}", "page": "90", "chapter": "conclusion", "number": -10},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )
            documents.append(
                Document(
                    content=f"Document {i} without embedding",
                    metadata={"name": f"name_{i}", "no_embedding": True, "chapter": "conclusion"},
                )
            )
            documents.append(
                Document(content=pd.DataFrame([i]), content_type="table", metadata={"name": f"table_doc_{i}"})
            )
            documents.append(
                Document(content=f"Doc {i} with zeros emb", metadata={"name": f"zeros_doc"}, embedding=embedding_zero)
            )
            documents.append(
                Document(content=f"Doc {i} with ones emb", metadata={"name": f"ones_doc"}, embedding=embedding_one)
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

    def test_filter_simple_metadata_value(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"page": "100"})
        assert len(result) == 3

    def test_filter_document_content(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"content": "A Foo Document 1"})
        assert len(result) > 0
        assert all(doc.content == "A Foo Document 1" for doc in result)

    def test_filter_document_type(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"content_type": "table"})
        assert len(result) > 0
        assert all(doc.content_type == "table" for doc in result)

    def test_filter_simple_list(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"page": ["100"]})
        assert all(doc.metadata["page"] == "100" for doc in result)
        result = docstore.filter_documents(filters={"page": ["100", "123"]})
        assert all(doc.metadata["page"] in ["100", "123"] for doc in result)

    def test_incorrect_filter_name(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"non_existing_meta_field": ["whatever"]})
        assert len(result) == 0

    def test_incorrect_filter_type(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        with pytest.raises(StoreError, match="dictionaries or lists"):
            docstore.filter_documents(filters="something odd")

    def test_incorrect_filter_value(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"page": ["nope"]})
        assert len(result) == 0

    def test_incorrect_filter_nesting(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        with pytest.raises(StoreError, match="malformed"):
            docstore.filter_documents(filters={"number": {"page": "100"}})
        with pytest.raises(StoreError, match="malformed"):
            docstore.filter_documents(filters={"number": {"page": {"chapter": "intro"}}})

    def test_eq_filter_explicit(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$eq": "100"}})
        assert all(doc.metadata["page"] == "100" for doc in result)

    def test_eq_filter_implicit(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"page": "100"})
        assert all(doc.metadata["page"] == "100" for doc in result)

    def test_eq_filter_date(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"content": pd.DataFrame([1])})
        assert len(result) > 0
        assert all(doc.content.equals(pd.DataFrame([1])) for doc in result)

    def test_eq_filter_table(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"content": pd.DataFrame([1])})
        assert len(result) > 0
        assert all(doc.content.equals(pd.DataFrame([1])) for doc in result)

    def test_eq_filter_tensor(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        embedding = np.zeros([768, 1]).astype(np.float32)
        result = docstore.filter_documents(filters={"embedding": embedding})
        assert len(result) > 0
        assert all(np.array_equal(embedding, doc.embedding) for doc in result)

    def test_in_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$in": ["100", "123", "n.a."]}})
        assert all(doc.metadata["page"] in ["100", "123"] for doc in result)
        result = docstore.filter_documents(filters={"page": ["100", "123", "n.a."]})
        assert all(doc.metadata["page"] in ["100", "123"] for doc in result)

    def test_in_filter_table(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"$in": {"content": [pd.DataFrame([1]), pd.DataFrame([2])]}})
        assert len(result) > 0
        assert all(doc.content.equals(pd.DataFrame([1])) or doc.content.equals(pd.DataFrame([2])) for doc in result)

    def test_in_filter_tensor(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        embedding_zero = np.zeros([768, 1]).astype(np.float32)
        embedding_one = np.ones([768, 1]).astype(np.float32)
        result = docstore.filter_documents(filters={"$in": {"embedding": [embedding_zero, embedding_one]}})
        assert len(result) > 0
        assert all(
            np.array_equal(embedding_zero, doc.embedding) or np.array_equal(embedding_one, doc.embedding)
            for doc in result
        )

    def test_ne_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$ne": "100"}})
        assert all(doc.metadata.get("page", None) != "100" for doc in result)

    def test_ne_filter_table(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"content": {"$ne": pd.DataFrame([1])}})
        assert len(result) > 0
        assert all(
            not isinstance(doc.content, pd.DataFrame) or not doc.content.equals(pd.DataFrame([1])) for doc in result
        )

    def test_ne_filter_tensor(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        embedding = np.zeros([768, 1]).astype(np.float32)
        result = docstore.filter_documents(filters={"embedding": {"$ne": embedding}})
        assert len(result) > 0
        assert all(
            not isinstance(doc.content, np.ndarray) or not np.array_equal(embedding, doc.embedding) for doc in result
        )

    def test_nin_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$nin": ["100", "123", "n.a."]}})
        assert all(doc.metadata.get("page", None) not in ["100", "123"] for doc in result)

    def test_nin_filter_table(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"content": {"$nin": [pd.DataFrame([1]), pd.DataFrame([0])]}})
        assert len(result) > 0
        assert all(
            not isinstance(doc.content, pd.DataFrame)
            or (not doc.content.equals(pd.DataFrame([1])) and not doc.content.equals(pd.DataFrame([0])))
            for doc in result
        )

    def test_nin_filter_tensor(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        embedding_zeros = np.zeros([768, 1]).astype(np.float32)
        embedding_ones = np.zeros([768, 1]).astype(np.float32)
        result = docstore.filter_documents(filters={"embedding": {"$nin": [embedding_ones, embedding_zeros]}})
        assert len(result) > 0
        assert all(
            not isinstance(doc.content, np.ndarray)
            or (
                not np.array_equal(embedding_zeros, doc.embedding) and not np.array_equal(embedding_ones, doc.embedding)
            )
            for doc in result
        )

    def test_gt_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$gt": 0.0}})
        assert all(doc.metadata["number"] > 0 for doc in result)

    def test_gt_filter_non_numeric(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        with pytest.raises(StoreError, match="Can't evaluate"):
            docstore.filter_documents(filters={"page": {"$gt": "100"}})

    def test_gt_filter_table(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        with pytest.raises(StoreError, match="Can't evaluate"):
            docstore.filter_documents(filters={"content": {"$gt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    def test_gt_filter_tensor(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        embedding_zeros = np.zeros([768, 1]).astype(np.float32)
        with pytest.raises(StoreError, match="Can't evaluate"):
            docstore.filter_documents(filters={"embedding": {"$gt": embedding_zeros}})

    def test_gte_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$gte": -2.0}})
        assert all(doc.metadata["number"] >= -2.0 for doc in result)

    def test_gte_filter_non_numeric(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        with pytest.raises(StoreError, match="Can't evaluate"):
            docstore.filter_documents(filters={"page": {"$gte": "100"}})

    def test_gte_filter_table(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        with pytest.raises(StoreError, match="Can't evaluate"):
            docstore.filter_documents(filters={"content": {"$gte": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    def test_gte_filter_tensor(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        embedding_zeros = np.zeros([768, 1]).astype(np.float32)
        with pytest.raises(StoreError, match="Can't evaluate"):
            docstore.filter_documents(filters={"embedding": {"$gte": embedding_zeros}})

    def test_lt_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lt": 0.0}})
        assert all(doc.metadata["number"] < 0.0 for doc in result)

    def test_lt_filter_non_numeric(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        with pytest.raises(StoreError, match="Can't evaluate"):
            docstore.filter_documents(filters={"page": {"$lt": "100"}})

    def test_lt_filter_table(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        with pytest.raises(StoreError, match="Can't evaluate"):
            docstore.filter_documents(filters={"content": {"$lt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    def test_lt_filter_tensor(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        embedding_ones = np.ones([768, 1]).astype(np.float32)
        with pytest.raises(StoreError, match="Can't evaluate"):
            docstore.filter_documents(filters={"embedding": {"$lt": embedding_ones}})

    def test_lte_filter(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lte": 2.0}})
        assert all(doc.metadata["number"] <= 2.0 for doc in result)

    def test_lte_filter_non_numeric(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        with pytest.raises(StoreError, match="Can't evaluate"):
            docstore.filter_documents(filters={"page": {"$lte": "100"}})

    def test_lte_filter_table(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        with pytest.raises(StoreError, match="Can't evaluate"):
            docstore.filter_documents(filters={"content": {"$lte": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    def test_lte_filter_tensor(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        embedding_ones = np.ones([768, 1]).astype(np.float32)
        with pytest.raises(StoreError, match="Can't evaluate"):
            docstore.filter_documents(filters={"embedding": {"$lte": embedding_ones}})

    def test_filter_simple_explicit_and_multi_key_dict(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$and": {"$lte": 2.0, "$gte": 0.0}}})
        assert all(int(doc.metadata["number"]) <= 2.0 and int(doc.metadata["number"]) >= 0.0 for doc in result)

    def test_filter_simple_explicit_and_sibling_dicts(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$and": [{"$lte": 2.0}, {"$gte": 0.0}]}})
        assert all(int(doc.metadata["number"]) <= 2.0 and int(doc.metadata["number"]) >= 0.0 for doc in result)

    def test_filter_simple_implicit_and(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lte": 2.0, "$gte": 0.0}})
        assert all(int(doc.metadata["number"]) >= 0.0 and int(doc.metadata["number"]) <= 2.0 for doc in result)

    def test_filter_nested_explicit_and(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters = {"$and": {"page": {"$and": {"$lte": "123", "$gte": "100"}}, "name": {"$in": ["name_0", "name_1"]}}}
        result = docstore.filter_documents(filters=filters)
        assert all(
            int(doc.metadata["page"]) >= 100
            and int(doc.metadata["page"]) <= 123
            and doc.metadata["name"] in ["name_0", "name_1"]
            for doc in result
        )

    def test_filter_nested_implicit_and(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters_simplified = {"page": {"$lte": "123", "$gte": "100"}, "name": ["name_0", "name_1"]}
        result = docstore.filter_documents(filters=filters_simplified)
        assert all(
            int(doc.metadata["page"]) >= 100
            and int(doc.metadata["page"]) <= 123
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
            "page": {"$lte": "123", "$gte": "100"},
            "$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}},
        }
        result = docstore.filter_documents(filters=filters_simplified)
        assert all(
            (int(doc.metadata["page"]) >= 100 and int(doc.metadata["page"]) <= 123)
            and (doc.metadata["name"] in ["name_0", "name_1"] or doc.metadata["number"] < 1.0)
            for doc in result
        )

    def test_filter_nested_or_and(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters_simplified = {
            "$or": {
                "number": {"$lt": 1.0},
                "$and": {"name": {"$in": ["name_0", "name_1"]}, "$not": {"chapter": {"$eq": "intro"}}},
            }
        }
        result = docstore.filter_documents(filters=filters_simplified)
        assert all(
            doc.metadata.get("number", 2) < 1.0
            or (doc.metadata["name"] in ["name_0", "name_1"] and doc.metadata["chapter"] != "intro")
            for doc in result
        )

    def test_filter_nested_multiple_identical_operators_same_level(self, docstore, filterable_docs):
        self.direct_write(docstore, filterable_docs)
        filters = {
            "$or": [
                {"$and": {"name": {"$in": ["name_0", "name_1"]}, "page": {"$gte": "100"}}},
                {"$and": {"name": {"$in": ["name_0", "name_1"]}, "page": {"$lt": "123"}}},
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
