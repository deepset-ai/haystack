# pylint: disable=too-many-public-methods
from typing import List

import pytest
import numpy as np
import pandas as pd

from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import DocumentStore, DuplicatePolicy
from haystack.preview.document_stores.errors import FilterError, MissingDocumentError, DuplicateDocumentError


class DocumentStoreBaseTests:
    @pytest.fixture
    def docstore(self) -> DocumentStore:
        raise NotImplementedError()

    @pytest.fixture
    def filterable_docs(self) -> List[Document]:
        embedding_zero = np.zeros(768).astype(np.float32)
        embedding_one = np.ones(768).astype(np.float32)

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
                Document(content=f"Doc {i} with zeros emb", metadata={"name": "zeros_doc"}, embedding=embedding_zero)
            )
            documents.append(
                Document(content=f"Doc {i} with ones emb", metadata={"name": "ones_doc"}, embedding=embedding_one)
            )
        return documents

    def contains_same_docs(self, first_list: List[Document], second_list: List[Document]) -> bool:
        """
        Utility to compare two lists of documents for equality regardless of the order od the documents.
        """
        return (
            len(first_list) > 0
            and len(second_list) > 0
            and first_list.sort(key=lambda d: d.id) == second_list.sort(key=lambda d: d.id)
        )

    @pytest.mark.unit
    def test_count_empty(self, docstore: DocumentStore):
        assert docstore.count_documents() == 0

    @pytest.mark.unit
    def test_count_not_empty(self, docstore: DocumentStore):
        docstore.write_documents(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert docstore.count_documents() == 3

    @pytest.mark.unit
    def test_no_filter_empty(self, docstore: DocumentStore):
        assert docstore.filter_documents() == []
        assert docstore.filter_documents(filters={}) == []

    @pytest.mark.unit
    def test_no_filter_not_empty(self, docstore: DocumentStore):
        docs = [Document(content="test doc")]
        docstore.write_documents(docs)
        assert docstore.filter_documents() == docs
        assert docstore.filter_documents(filters={}) == docs

    @pytest.mark.unit
    def test_filter_simple_metadata_value(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": "100"})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.metadata.get("page") == "100"])

    @pytest.mark.unit
    def test_filter_simple_list_single_element(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": ["100"]})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.metadata.get("page") == "100"])

    @pytest.mark.unit
    def test_filter_document_content(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"content": "A Foo Document 1"})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if doc.content_type == "text" and doc.content == "A Foo Document 1"]
        )

    @pytest.mark.unit
    def test_filter_document_type(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"content_type": "table"})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.content_type == "table"])

    @pytest.mark.unit
    def test_filter_simple_list_one_value(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": ["100"]})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.metadata.get("page") in ["100"]])

    @pytest.mark.unit
    def test_filter_simple_list(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": ["100", "123"]})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if doc.metadata.get("page") in ["100", "123"]]
        )

    @pytest.mark.unit
    def test_incorrect_filter_name(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"non_existing_meta_field": ["whatever"]})
        assert len(result) == 0

    @pytest.mark.unit
    def test_incorrect_filter_type(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters="something odd")  # type: ignore

    @pytest.mark.unit
    def test_incorrect_filter_value(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": ["nope"]})
        assert len(result) == 0

    @pytest.mark.unit
    def test_incorrect_filter_nesting(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"number": {"page": "100"}})

    @pytest.mark.unit
    def test_deeper_incorrect_filter_nesting(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"number": {"page": {"chapter": "intro"}}})

    @pytest.mark.unit
    def test_eq_filter_explicit(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$eq": "100"}})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.metadata.get("page") == "100"])

    @pytest.mark.unit
    def test_eq_filter_implicit(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": "100"})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.metadata.get("page") == "100"])

    @pytest.mark.unit
    def test_eq_filter_table(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"content": pd.DataFrame([1])})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if isinstance(doc.content, pd.DataFrame) and doc.content.equals(pd.DataFrame([1]))
            ],
        )

    @pytest.mark.unit
    def test_eq_filter_embedding(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding = np.zeros(768).astype(np.float32)
        result = docstore.filter_documents(filters={"embedding": embedding})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if np.array_equal(embedding, doc.embedding)]  # type: ignore
        )

    @pytest.mark.unit
    def test_in_filter_explicit(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$in": ["100", "123", "n.a."]}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if doc.metadata.get("page") in ["100", "123"]]
        )

    @pytest.mark.unit
    def test_in_filter_implicit(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": ["100", "123", "n.a."]})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if doc.metadata.get("page") in ["100", "123"]]
        )

    @pytest.mark.unit
    def test_in_filter_table(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"content": {"$in": [pd.DataFrame([1]), pd.DataFrame([2])]}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if isinstance(doc.content, pd.DataFrame)
                and (doc.content.equals(pd.DataFrame([1])) or doc.content.equals(pd.DataFrame([2])))
            ],
        )

    @pytest.mark.unit
    def test_in_filter_embedding(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding_zero = np.zeros(768, np.float32)
        embedding_one = np.ones(768, np.float32)
        result = docstore.filter_documents(filters={"embedding": {"$in": [embedding_zero, embedding_one]}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if isinstance(doc.embedding, np.ndarray)
                and (np.array_equal(embedding_zero, doc.embedding) or np.array_equal(embedding_one, doc.embedding))
            ],
        )

    @pytest.mark.unit
    def test_ne_filter(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$ne": "100"}})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.metadata.get("page") != "100"])

    @pytest.mark.unit
    def test_ne_filter_table(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"content": {"$ne": pd.DataFrame([1])}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if not isinstance(doc.content, pd.DataFrame) or not doc.content.equals(pd.DataFrame([1]))
            ],
        )

    @pytest.mark.unit
    def test_ne_filter_embedding(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding = np.zeros([768, 1]).astype(np.float32)
        result = docstore.filter_documents(filters={"embedding": {"$ne": embedding}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if not isinstance(doc.content, np.ndarray) or not np.array_equal(embedding, doc.embedding)  # type: ignore
            ],
        )

    @pytest.mark.unit
    def test_nin_filter_table(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"content": {"$nin": [pd.DataFrame([1]), pd.DataFrame([0])]}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if not isinstance(doc.content, pd.DataFrame)
                or (not doc.content.equals(pd.DataFrame([1])) and not doc.content.equals(pd.DataFrame([0])))
            ],
        )

    @pytest.mark.unit
    def test_nin_filter_embedding(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding_zeros = np.zeros([768, 1]).astype(np.float32)
        embedding_ones = np.zeros([768, 1]).astype(np.float32)
        result = docstore.filter_documents(filters={"embedding": {"$nin": [embedding_ones, embedding_zeros]}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if not isinstance(doc.content, np.ndarray)
                or (
                    not np.array_equal(embedding_zeros, doc.embedding)  # type: ignore
                    and not np.array_equal(embedding_ones, doc.embedding)  # type: ignore
                )
            ],
        )

    @pytest.mark.unit
    def test_nin_filter(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$nin": ["100", "123", "n.a."]}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if doc.metadata.get("page") not in ["100", "123"]]
        )

    @pytest.mark.unit
    def test_gt_filter(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$gt": 0.0}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if "number" in doc.metadata and doc.metadata["number"] > 0]
        )

    @pytest.mark.unit
    def test_gt_filter_non_numeric(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"page": {"$gt": "100"}})

    @pytest.mark.unit
    def test_gt_filter_table(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"content": {"$gt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    @pytest.mark.unit
    def test_gt_filter_embedding(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding_zeros = np.zeros([768, 1]).astype(np.float32)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"embedding": {"$gt": embedding_zeros}})

    @pytest.mark.unit
    def test_gte_filter(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$gte": -2.0}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if "number" in doc.metadata and doc.metadata["number"] >= -2.0]
        )

    @pytest.mark.unit
    def test_gte_filter_non_numeric(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"page": {"$gte": "100"}})

    @pytest.mark.unit
    def test_gte_filter_table(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"content": {"$gte": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    @pytest.mark.unit
    def test_gte_filter_embedding(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding_zeros = np.zeros([768, 1]).astype(np.float32)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"embedding": {"$gte": embedding_zeros}})

    @pytest.mark.unit
    def test_lt_filter(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lt": 0.0}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if "number" in doc.metadata and doc.metadata["number"] < 0]
        )

    @pytest.mark.unit
    def test_lt_filter_non_numeric(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"page": {"$lt": "100"}})

    @pytest.mark.unit
    def test_lt_filter_table(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"content": {"$lt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    @pytest.mark.unit
    def test_lt_filter_embedding(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding_ones = np.ones([768, 1]).astype(np.float32)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"embedding": {"$lt": embedding_ones}})

    @pytest.mark.unit
    def test_lte_filter(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lte": 2.0}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if "number" in doc.metadata and doc.metadata["number"] <= 2.0]
        )

    @pytest.mark.unit
    def test_lte_filter_non_numeric(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"page": {"$lte": "100"}})

    @pytest.mark.unit
    def test_lte_filter_table(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"content": {"$lte": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    @pytest.mark.unit
    def test_lte_filter_embedding(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding_ones = np.ones([768, 1]).astype(np.float32)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"embedding": {"$lte": embedding_ones}})

    @pytest.mark.unit
    def test_filter_simple_implicit_and_with_multi_key_dict(
        self, docstore: DocumentStore, filterable_docs: List[Document]
    ):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lte": 2.0, "$gte": 0.0}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.metadata and doc.metadata["number"] >= 0.0 and doc.metadata["number"] <= 2.0
            ],
        )

    @pytest.mark.unit
    def test_filter_simple_explicit_and_with_multikey_dict(
        self, docstore: DocumentStore, filterable_docs: List[Document]
    ):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$and": {"$lte": 0, "$gte": -2}}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.metadata and doc.metadata["number"] >= 0.0 and doc.metadata["number"] <= 2.0
            ],
        )

    @pytest.mark.unit
    def test_filter_simple_explicit_and_with_list(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$and": [{"$lte": 2}, {"$gte": 0}]}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.metadata and doc.metadata["number"] <= 2.0 and doc.metadata["number"] >= 0.0
            ],
        )

    @pytest.mark.unit
    def test_filter_simple_implicit_and(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lte": 2.0, "$gte": 0}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.metadata and doc.metadata["number"] <= 2.0 and doc.metadata["number"] >= 0.0
            ],
        )

    @pytest.mark.unit
    def test_filter_nested_explicit_and(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters = {"$and": {"number": {"$and": {"$lte": 2, "$gte": 0}}, "name": {"$in": ["name_0", "name_1"]}}}
        result = docstore.filter_documents(filters=filters)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    "number" in doc.metadata
                    and doc.metadata["number"] >= 0
                    and doc.metadata["number"] <= 2
                    and doc.metadata["name"] in ["name_0", "name_1"]
                )
            ],
        )

    @pytest.mark.unit
    def test_filter_nested_implicit_and(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters_simplified = {"number": {"$lte": 2, "$gte": 0}, "name": ["name_0", "name_1"]}
        result = docstore.filter_documents(filters=filters_simplified)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    "number" in doc.metadata
                    and doc.metadata["number"] <= 2
                    and doc.metadata["number"] >= 0
                    and doc.metadata.get("name") in ["name_0", "name_1"]
                )
            ],
        )

    @pytest.mark.unit
    def test_filter_simple_or(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters = {"$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}}}
        result = docstore.filter_documents(filters=filters)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    ("number" in doc.metadata and doc.metadata["number"] < 1)
                    or doc.metadata.get("name") in ["name_0", "name_1"]
                )
            ],
        )

    @pytest.mark.unit
    def test_filter_nested_or(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters = {"$or": {"name": {"$or": [{"$eq": "name_0"}, {"$eq": "name_1"}]}, "number": {"$lt": 1.0}}}
        result = docstore.filter_documents(filters=filters)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.metadata.get("name") in ["name_0", "name_1"]
                    or ("number" in doc.metadata and doc.metadata["number"] < 1)
                )
            ],
        )

    @pytest.mark.unit
    def test_filter_nested_and_or_explicit(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters_simplified = {
            "$and": {"page": {"$eq": "123"}, "$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}}}
        }
        result = docstore.filter_documents(filters=filters_simplified)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.metadata.get("page") in ["123"]
                    and (
                        doc.metadata.get("name") in ["name_0", "name_1"]
                        or ("number" in doc.metadata and doc.metadata["number"] < 1)
                    )
                )
            ],
        )

    @pytest.mark.unit
    def test_filter_nested_and_or_implicit(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters_simplified = {
            "page": {"$eq": "123"},
            "$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}},
        }
        result = docstore.filter_documents(filters=filters_simplified)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.metadata.get("page") in ["123"]
                    and (
                        doc.metadata.get("name") in ["name_0", "name_1"]
                        or ("number" in doc.metadata and doc.metadata["number"] < 1)
                    )
                )
            ],
        )

    @pytest.mark.unit
    def test_filter_nested_or_and(self, docstore: DocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters_simplified = {
            "$or": {
                "number": {"$lt": 1.0},
                "$and": {"name": {"$in": ["name_0", "name_1"]}, "$not": {"chapter": {"$eq": "intro"}}},
            }
        }
        result = docstore.filter_documents(filters=filters_simplified)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    ("number" in doc.metadata and doc.metadata["number"] < 1)
                    or (
                        doc.metadata.get("name") in ["name_0", "name_1"]
                        or ("chapter" in doc.metadata and doc.metadata["chapter"] != "intro")
                    )
                )
            ],
        )

    @pytest.mark.unit
    def test_filter_nested_multiple_identical_operators_same_level(
        self, docstore: DocumentStore, filterable_docs: List[Document]
    ):
        docstore.write_documents(filterable_docs)
        filters = {
            "$or": [
                {"$and": {"name": {"$in": ["name_0", "name_1"]}, "page": "100"}},
                {"$and": {"chapter": {"$in": ["intro", "abstract"]}, "page": "123"}},
            ]
        }
        result = docstore.filter_documents(filters=filters)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    (doc.metadata.get("name") in ["name_0", "name_1"] and doc.metadata.get("page") == "100")
                    or (doc.metadata.get("chapter") in ["intro", "abstract"] and doc.metadata.get("page") == "100")
                )
            ],
        )

    @pytest.mark.unit
    def test_write(self, docstore: DocumentStore):
        doc = Document(content="test doc")
        docstore.write_documents([doc])
        assert docstore.filter_documents(filters={"id": doc.id}) == [doc]

    @pytest.mark.unit
    def test_write_duplicate_fail(self, docstore: DocumentStore):
        doc = Document(content="test doc")
        docstore.write_documents([doc])
        with pytest.raises(DuplicateDocumentError, match=f"ID '{doc.id}' already exists."):
            docstore.write_documents(documents=[doc], policy=DuplicatePolicy.FAIL)
        assert docstore.filter_documents(filters={"id": doc.id}) == [doc]

    @pytest.mark.unit
    def test_write_duplicate_skip(self, docstore: DocumentStore):
        doc = Document(content="test doc")
        docstore.write_documents([doc])
        docstore.write_documents(documents=[doc], policy=DuplicatePolicy.SKIP)
        assert docstore.filter_documents(filters={"id": doc.id}) == [doc]

    @pytest.mark.unit
    def test_write_duplicate_overwrite(self, docstore: DocumentStore):
        doc1 = Document(content="test doc 1")
        doc2 = Document(content="test doc 2")
        object.__setattr__(doc2, "id", doc1.id)  # Make two docs with different content but same ID

        docstore.write_documents([doc2])
        assert docstore.filter_documents(filters={"id": doc1.id}) == [doc2]
        docstore.write_documents(documents=[doc1], policy=DuplicatePolicy.OVERWRITE)
        assert docstore.filter_documents(filters={"id": doc1.id}) == [doc1]

    @pytest.mark.unit
    def test_write_not_docs(self, docstore: DocumentStore):
        with pytest.raises(ValueError):
            docstore.write_documents(["not a document for sure"])  # type: ignore

    @pytest.mark.unit
    def test_write_not_list(self, docstore: DocumentStore):
        with pytest.raises(ValueError):
            docstore.write_documents("not a list actually")  # type: ignore

    @pytest.mark.unit
    def test_delete_empty(self, docstore: DocumentStore):
        with pytest.raises(MissingDocumentError):
            docstore.delete_documents(["test"])

    @pytest.mark.unit
    def test_delete_not_empty(self, docstore: DocumentStore):
        doc = Document(content="test doc")
        docstore.write_documents([doc])

        docstore.delete_documents([doc.id])

        with pytest.raises(Exception):
            assert docstore.filter_documents(filters={"id": doc.id})

    @pytest.mark.unit
    def test_delete_not_empty_nonexisting(self, docstore: DocumentStore):
        doc = Document(content="test doc")
        docstore.write_documents([doc])

        with pytest.raises(MissingDocumentError):
            docstore.delete_documents(["non_existing"])

        assert docstore.filter_documents(filters={"id": doc.id}) == [doc]
