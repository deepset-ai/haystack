from copy import deepcopy
import logging
import math
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import json
import responses
from responses import matchers
from unittest.mock import Mock
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError

from ..conftest import (
    deepset_cloud_fixture,
    get_document_store,
    ensure_ids_are_correct_uuids,
    MOCK_DC,
    DC_API_ENDPOINT,
    DC_API_KEY,
    DC_TEST_INDEX,
    SAMPLES_PATH,
)
from haystack.document_stores import (
    WeaviateDocumentStore,
    DeepsetCloudDocumentStore,
    InMemoryDocumentStore,
    MilvusDocumentStore,
    FAISSDocumentStore,
    ElasticsearchDocumentStore,
    OpenSearchDocumentStore,
)

from haystack.document_stores.base import BaseDocumentStore
from haystack.document_stores.es_converter import elasticsearch_index_to_document_store
from haystack.errors import DuplicateDocumentError
from haystack.schema import Document, Label, Answer, Span
from haystack.nodes import EmbeddingRetriever, PreProcessor
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import DeepsetCloudError


DOCUMENTS = [
    {
        "meta": {"name": "name_1", "year": "2020", "month": "01"},
        "content": "text_1",
        "embedding": np.random.rand(768).astype(np.float32),
    },
    {
        "meta": {"name": "name_2", "year": "2020", "month": "02"},
        "content": "text_2",
        "embedding": np.random.rand(768).astype(np.float32),
    },
    {
        "meta": {"name": "name_3", "year": "2020", "month": "03"},
        "content": "text_3",
        "embedding": np.random.rand(768).astype(np.float64),
    },
    {
        "meta": {"name": "name_4", "year": "2021", "month": "01"},
        "content": "text_4",
        "embedding": np.random.rand(768).astype(np.float32),
    },
    {
        "meta": {"name": "name_5", "year": "2021", "month": "02"},
        "content": "text_5",
        "embedding": np.random.rand(768).astype(np.float32),
    },
    {
        "meta": {"name": "name_6", "year": "2021", "month": "03"},
        "content": "text_6",
        "embedding": np.random.rand(768).astype(np.float64),
    },
]


@pytest.mark.elasticsearch
def test_init_elastic_client():
    # defaults
    _ = ElasticsearchDocumentStore()

    # list of hosts + single port
    _ = ElasticsearchDocumentStore(host=["localhost", "127.0.0.1"], port=9200)

    # list of hosts + list of ports (wrong)
    with pytest.raises(Exception):
        _ = ElasticsearchDocumentStore(host=["localhost", "127.0.0.1"], port=[9200])

    # list of hosts + list
    _ = ElasticsearchDocumentStore(host=["localhost", "127.0.0.1"], port=[9200, 9200])

    # only api_key
    with pytest.raises(Exception):
        _ = ElasticsearchDocumentStore(host=["localhost"], port=[9200], api_key="test")

    # api_key +  id
    _ = ElasticsearchDocumentStore(host=["localhost"], port=[9200], api_key="test", api_key_id="test")


@pytest.mark.elasticsearch
def test_init_elastic_doc_store_with_index_recreation():
    index_name = "test_index_recreation"
    label_index_name = "test_index_recreation_labels"

    document_store = ElasticsearchDocumentStore(index=index_name, label_index=label_index_name)
    documents = [Document(content="Doc1")]
    labels = [
        Label(
            query="query",
            document=documents[0],
            is_correct_document=True,
            is_correct_answer=False,
            origin="user-feedback",
            answer=None,
        )
    ]
    document_store.write_documents(documents, index=index_name)
    document_store.write_labels(labels, index=label_index_name)

    document_store = ElasticsearchDocumentStore(index=index_name, label_index=label_index_name, recreate_index=True)
    docs = document_store.get_all_documents(index=index_name)
    labels = document_store.get_all_labels(index=label_index_name)

    assert len(docs) == 0
    assert len(labels) == 0


@pytest.mark.elasticsearch
def test_elasticsearch_eq_filter():
    documents = [
        {"content": "some text", "id": "1", "keyword_field": ["x", "y", "z"], "number_field": [1, 2, 3, 4]},
        {"content": "some text", "id": "2", "keyword_field": ["x", "y", "w"], "number_field": [1, 2, 3]},
        {"content": "some text", "id": "3", "keyword_field": ["x", "z"], "number_field": [2, 4]},
        {"content": "some text", "id": "4", "keyword_field": ["z", "x"], "number_field": [5, 6]},
        {"content": "some text", "id": "5", "keyword_field": ["x", "y"], "number_field": [2, 3]},
    ]

    index = "test_elasticsearch_eq_filter"
    document_store = ElasticsearchDocumentStore(index=index, recreate_index=True)
    document_store.write_documents(documents)

    filter = {"keyword_field": {"$eq": ["z", "x"]}}
    filtered_docs = document_store.get_all_documents(index=index, filters=filter)
    assert len(filtered_docs) == 2
    for doc in filtered_docs:
        assert set(doc.meta["keyword_field"]) == {"x", "z"}

    filter = {"number_field": {"$eq": [2, 3]}}
    filtered_docs = document_store.query(query=None, index=index, filters=filter)
    assert len(filtered_docs) == 1
    assert filtered_docs[0].meta["number_field"] == [2, 3]
    assert filtered_docs[0].id == "5"


def test_write_with_duplicate_doc_ids(document_store: BaseDocumentStore):
    duplicate_documents = [
        Document(content="Doc1", id_hash_keys=["content"]),
        Document(content="Doc1", id_hash_keys=["content"]),
    ]
    document_store.write_documents(duplicate_documents, duplicate_documents="skip")
    assert len(document_store.get_all_documents()) == 1
    with pytest.raises(Exception):
        document_store.write_documents(duplicate_documents, duplicate_documents="fail")


@pytest.mark.parametrize(
    "document_store", ["elasticsearch", "faiss", "memory", "milvus1", "weaviate", "pinecone"], indirect=True
)
def test_write_with_duplicate_doc_ids_custom_index(document_store: BaseDocumentStore):
    duplicate_documents = [
        Document(content="Doc1", id_hash_keys=["content"]),
        Document(content="Doc1", id_hash_keys=["content"]),
    ]
    document_store.delete_index(index="haystack_custom_test")
    document_store.write_documents(duplicate_documents, index="haystack_custom_test", duplicate_documents="skip")
    assert len(document_store.get_all_documents(index="haystack_custom_test")) == 1
    with pytest.raises(DuplicateDocumentError):
        document_store.write_documents(duplicate_documents, index="haystack_custom_test", duplicate_documents="fail")

    # Weaviate manipulates document objects in-place when writing them to an index.
    # It generates a uuid based on the provided id and the index name where the document is added to.
    # We need to get rid of these generated uuids for this test and therefore reset the document objects.
    # As a result, the documents will receive a fresh uuid based on their id_hash_keys and a different index name.
    if isinstance(document_store, WeaviateDocumentStore):
        duplicate_documents = [
            Document(content="Doc1", id_hash_keys=["content"]),
            Document(content="Doc1", id_hash_keys=["content"]),
        ]
    # writing to the default, empty index should still work
    document_store.write_documents(duplicate_documents, duplicate_documents="fail")


def test_get_all_documents_without_filters(document_store_with_docs):
    print("hey!")
    documents = document_store_with_docs.get_all_documents()
    assert all(isinstance(d, Document) for d in documents)
    assert len(documents) == 5
    assert {d.meta["name"] for d in documents} == {"filename1", "filename2", "filename3", "filename4", "filename5"}
    assert {d.meta["meta_field"] for d in documents} == {"test1", "test2", "test3", "test4", "test5"}


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Test fails on Windows with an SQLite exception")
def test_get_all_documents_large_quantities(document_store: BaseDocumentStore):
    # Test to exclude situations like Weaviate not returning more than 100 docs by default
    #   https://github.com/deepset-ai/haystack/issues/1893
    docs_to_write = [
        {"meta": {"name": f"name_{i}"}, "content": f"text_{i}", "embedding": np.random.rand(768).astype(np.float32)}
        for i in range(1000)
    ]
    document_store.write_documents(docs_to_write)
    documents = document_store.get_all_documents()
    assert all(isinstance(d, Document) for d in documents)
    assert len(documents) == len(docs_to_write)


def test_get_all_document_filter_duplicate_text_value(document_store: BaseDocumentStore):
    documents = [
        Document(content="Doc1", meta={"meta_field": "0"}, id_hash_keys=["meta"]),
        Document(content="Doc1", meta={"meta_field": "1", "name": "file.txt"}, id_hash_keys=["meta"]),
        Document(content="Doc2", meta={"name": "file_2.txt"}, id_hash_keys=["meta"]),
    ]
    document_store.write_documents(documents)
    documents = document_store.get_all_documents(filters={"meta_field": ["1"]})
    assert documents[0].content == "Doc1"
    assert len(documents) == 1
    assert {d.meta["name"] for d in documents} == {"file.txt"}

    documents = document_store.get_all_documents(filters={"meta_field": ["0"]})
    assert documents[0].content == "Doc1"
    assert len(documents) == 1
    assert documents[0].meta.get("name") is None

    documents = document_store.get_all_documents(filters={"name": ["file_2.txt"]})
    assert documents[0].content == "Doc2"
    assert len(documents) == 1
    assert documents[0].meta.get("meta_field") is None


def test_get_all_documents_with_correct_filters(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents(filters={"meta_field": ["test2"]})
    assert len(documents) == 1
    assert documents[0].meta["name"] == "filename2"

    documents = document_store_with_docs.get_all_documents(filters={"meta_field": ["test1", "test3"]})
    assert len(documents) == 2
    assert {d.meta["name"] for d in documents} == {"filename1", "filename3"}
    assert {d.meta["meta_field"] for d in documents} == {"test1", "test3"}


def test_get_all_documents_with_correct_filters_legacy_sqlite(docs, tmp_path):
    document_store_with_docs = get_document_store("sql", tmp_path)
    document_store_with_docs.write_documents(docs)

    document_store_with_docs.use_windowed_query = False
    documents = document_store_with_docs.get_all_documents(filters={"meta_field": ["test2"]})
    assert len(documents) == 1
    assert documents[0].meta["name"] == "filename2"

    documents = document_store_with_docs.get_all_documents(filters={"meta_field": ["test1", "test3"]})
    assert len(documents) == 2
    assert {d.meta["name"] for d in documents} == {"filename1", "filename3"}
    assert {d.meta["meta_field"] for d in documents} == {"test1", "test3"}


def test_get_all_documents_with_incorrect_filter_name(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents(filters={"incorrect_meta_field": ["test2"]})
    assert len(documents) == 0


def test_get_all_documents_with_incorrect_filter_value(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents(filters={"meta_field": ["incorrect_value"]})
    assert len(documents) == 0


# See test_pinecone.py
@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch", "sql", "weaviate", "memory"], indirect=True)
def test_extended_filter(document_store_with_docs):
    # Test comparison operators individually
    documents = document_store_with_docs.get_all_documents(filters={"meta_field": {"$eq": "test1"}})
    assert len(documents) == 1
    documents = document_store_with_docs.get_all_documents(filters={"meta_field": "test1"})
    assert len(documents) == 1

    documents = document_store_with_docs.get_all_documents(filters={"meta_field": {"$in": ["test1", "test2", "n.a."]}})
    assert len(documents) == 2
    documents = document_store_with_docs.get_all_documents(filters={"meta_field": ["test1", "test2", "n.a."]})
    assert len(documents) == 2

    documents = document_store_with_docs.get_all_documents(filters={"meta_field": {"$ne": "test1"}})
    assert len(documents) == 4

    documents = document_store_with_docs.get_all_documents(filters={"meta_field": {"$nin": ["test1", "test2", "n.a."]}})
    assert len(documents) == 3

    documents = document_store_with_docs.get_all_documents(filters={"numeric_field": {"$gt": 3.0}})
    assert len(documents) == 3

    documents = document_store_with_docs.get_all_documents(filters={"numeric_field": {"$gte": 3.0}})
    assert len(documents) == 4

    documents = document_store_with_docs.get_all_documents(filters={"numeric_field": {"$lt": 3.0}})
    assert len(documents) == 1

    documents = document_store_with_docs.get_all_documents(filters={"numeric_field": {"$lte": 3.0}})
    assert len(documents) == 2

    # Test compound filters
    filters = {"date_field": {"$lte": "2020-12-31", "$gte": "2019-01-01"}}
    documents = document_store_with_docs.get_all_documents(filters=filters)
    assert len(documents) == 3

    filters = {
        "$and": {
            "date_field": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
            "name": {"$in": ["filename5", "filename3"]},
        }
    }
    documents = document_store_with_docs.get_all_documents(filters=filters)
    assert len(documents) == 1
    filters_simplified = {
        "date_field": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
        "name": ["filename5", "filename3"],
    }
    documents_simplified_filter = document_store_with_docs.get_all_documents(filters=filters_simplified)
    # Order of returned documents might differ
    assert len(documents) == len(documents_simplified_filter) and all(
        doc in documents_simplified_filter for doc in documents
    )

    filters = {
        "$and": {
            "date_field": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
            "$or": {"name": {"$in": ["filename5", "filename3"]}, "numeric_field": {"$lte": 5.0}},
        }
    }
    documents = document_store_with_docs.get_all_documents(filters=filters)
    assert len(documents) == 2
    filters_simplified = {
        "date_field": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
        "$or": {"name": ["filename5", "filename3"], "numeric_field": {"$lte": 5.0}},
    }
    documents_simplified_filter = document_store_with_docs.get_all_documents(filters=filters_simplified)
    assert len(documents) == len(documents_simplified_filter) and all(
        doc in documents_simplified_filter for doc in documents
    )

    filters = {
        "$and": {
            "date_field": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
            "$or": {
                "name": {"$in": ["filename5", "filename3"]},
                "$and": {"numeric_field": {"$lte": 5.0}, "$not": {"meta_field": {"$eq": "test2"}}},
            },
        }
    }
    documents = document_store_with_docs.get_all_documents(filters=filters)
    assert len(documents) == 1
    filters_simplified = {
        "date_field": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
        "$or": {
            "name": ["filename5", "filename3"],
            "$and": {"numeric_field": {"$lte": 5.0}, "$not": {"meta_field": "test2"}},
        },
    }
    documents_simplified_filter = document_store_with_docs.get_all_documents(filters=filters_simplified)
    assert len(documents) == len(documents_simplified_filter) and all(
        doc in documents_simplified_filter for doc in documents
    )

    # Test nested logical operations within "$not", important as we apply De Morgan's laws in WeaviateDocumentstore
    filters = {
        "$not": {
            "$or": {
                "$and": {"numeric_field": {"$gt": 3.0}, "meta_field": {"$ne": "test3"}},
                "$not": {"date_field": {"$lt": "2020-01-01"}},
            }
        }
    }
    documents = document_store_with_docs.get_all_documents(filters=filters)
    docs_meta = [doc.meta["meta_field"] for doc in documents]
    assert len(documents) == 2
    assert "test3" in docs_meta
    assert "test5" in docs_meta

    # Test same logical operator twice on same level
    filters = {
        "$or": [
            {"$and": {"meta_field": {"$in": ["test1", "test2"]}, "date_field": {"$gte": "2020-01-01"}}},
            {"$and": {"meta_field": {"$in": ["test3", "test4"]}, "date_field": {"$lt": "2020-01-01"}}},
        ]
    }
    documents = document_store_with_docs.get_all_documents(filters=filters)
    docs_meta = [doc.meta["meta_field"] for doc in documents]
    assert len(documents) == 2
    assert "test1" in docs_meta
    assert "test3" in docs_meta


def test_get_document_by_id(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents()
    doc = document_store_with_docs.get_document_by_id(documents[0].id)
    assert doc.id == documents[0].id
    assert doc.content == documents[0].content


def test_get_documents_by_id(document_store: BaseDocumentStore):
    # generate more documents than the elasticsearch default query size limit of 10
    docs_to_generate = 15
    documents = [{"content": "doc-" + str(i)} for i in range(docs_to_generate)]
    document_store.write_documents(documents)

    all_docs = document_store.get_all_documents()
    all_ids = [doc.id for doc in all_docs]

    retrieved_by_id = document_store.get_documents_by_id(all_ids)
    retrieved_ids = [doc.id for doc in retrieved_by_id]

    # all documents in the index should be retrieved when passing all document ids in the index
    assert set(retrieved_ids) == set(all_ids)


def test_get_document_count(document_store: BaseDocumentStore):
    documents = [
        {"content": "text1", "id": "1", "meta_field_for_count": "c"},
        {"content": "text2", "id": "2", "meta_field_for_count": "b"},
        {"content": "text3", "id": "3", "meta_field_for_count": "b"},
        {"content": "text4", "id": "4", "meta_field_for_count": "b"},
    ]
    document_store.write_documents(documents)
    assert document_store.get_document_count() == 4
    assert document_store.get_document_count(filters={"meta_field_for_count": ["c"]}) == 1
    assert document_store.get_document_count(filters={"meta_field_for_count": ["b"]}) == 3


def test_get_all_documents_generator(document_store: BaseDocumentStore):
    documents = [
        {"content": "text1", "id": "1", "meta_field_for_count": "a"},
        {"content": "text2", "id": "2", "meta_field_for_count": "b"},
        {"content": "text3", "id": "3", "meta_field_for_count": "b"},
        {"content": "text4", "id": "4", "meta_field_for_count": "b"},
        {"content": "text5", "id": "5", "meta_field_for_count": "b"},
    ]

    document_store.write_documents(documents)
    assert len(list(document_store.get_all_documents_generator(batch_size=2))) == 5


@pytest.mark.parametrize("update_existing_documents", [True, False])
def test_update_existing_documents(document_store, update_existing_documents):
    original_docs = [{"content": "text1_orig", "id": "1", "meta_field_for_count": "a"}]

    updated_docs = [{"content": "text1_new", "id": "1", "meta_field_for_count": "a"}]

    document_store.write_documents(original_docs)
    assert document_store.get_document_count() == 1

    if update_existing_documents:
        document_store.write_documents(updated_docs, duplicate_documents="overwrite")
    else:
        with pytest.raises(Exception):
            document_store.write_documents(updated_docs, duplicate_documents="fail")

    stored_docs = document_store.get_all_documents()
    assert len(stored_docs) == 1
    if update_existing_documents:
        assert stored_docs[0].content == updated_docs[0]["content"]
    else:
        assert stored_docs[0].content == original_docs[0]["content"]


def test_write_document_meta(document_store: BaseDocumentStore):
    documents = [
        {"content": "dict_without_meta", "id": "1"},
        {"content": "dict_with_meta", "meta_field": "test2", "name": "filename2", "id": "2"},
        Document(content="document_object_without_meta", id="3"),
        Document(content="document_object_with_meta", meta={"meta_field": "test4", "name": "filename3"}, id="4"),
    ]
    document_store.write_documents(documents)
    documents_in_store = document_store.get_all_documents()
    assert len(documents_in_store) == 4

    assert not document_store.get_document_by_id("1").meta
    assert document_store.get_document_by_id("2").meta["meta_field"] == "test2"
    assert not document_store.get_document_by_id("3").meta
    assert document_store.get_document_by_id("4").meta["meta_field"] == "test4"


@pytest.mark.parametrize("document_store", ["sql"], indirect=True)
def test_sql_write_document_invalid_meta(document_store: BaseDocumentStore):
    documents = [
        {
            "content": "dict_with_invalid_meta",
            "valid_meta_field": "test1",
            "invalid_meta_field": [1, 2, 3],
            "name": "filename1",
            "id": "1",
        },
        Document(
            content="document_object_with_invalid_meta",
            meta={"valid_meta_field": "test2", "invalid_meta_field": [1, 2, 3], "name": "filename2"},
            id="2",
        ),
    ]
    document_store.write_documents(documents)
    documents_in_store = document_store.get_all_documents()
    assert len(documents_in_store) == 2

    assert document_store.get_document_by_id("1").meta == {"name": "filename1", "valid_meta_field": "test1"}
    assert document_store.get_document_by_id("2").meta == {"name": "filename2", "valid_meta_field": "test2"}


@pytest.mark.parametrize("document_store", ["sql"], indirect=True)
def test_sql_write_different_documents_same_vector_id(document_store: BaseDocumentStore):
    doc1 = {"content": "content 1", "name": "doc1", "id": "1", "vector_id": "vector_id"}
    doc2 = {"content": "content 2", "name": "doc2", "id": "2", "vector_id": "vector_id"}

    document_store.write_documents([doc1], index="index1")
    documents_in_index1 = document_store.get_all_documents(index="index1")
    assert len(documents_in_index1) == 1
    document_store.write_documents([doc2], index="index2")
    documents_in_index2 = document_store.get_all_documents(index="index2")
    assert len(documents_in_index2) == 1

    document_store.write_documents([doc1], index="index3")
    with pytest.raises(Exception, match=r"(?i)unique"):
        document_store.write_documents([doc2], index="index3")


def test_write_document_index(document_store: BaseDocumentStore):
    document_store.delete_index("haystack_test_one")
    document_store.delete_index("haystack_test_two")
    documents = [{"content": "text1", "id": "1"}, {"content": "text2", "id": "2"}]
    document_store.write_documents([documents[0]], index="haystack_test_one")
    assert len(document_store.get_all_documents(index="haystack_test_one")) == 1

    document_store.write_documents([documents[1]], index="haystack_test_two")
    assert len(document_store.get_all_documents(index="haystack_test_two")) == 1

    assert len(document_store.get_all_documents(index="haystack_test_one")) == 1
    assert len(document_store.get_all_documents()) == 0


@pytest.mark.parametrize(
    "document_store", ["elasticsearch", "faiss", "memory", "milvus1", "milvus", "weaviate"], indirect=True
)
def test_document_with_embeddings(document_store: BaseDocumentStore):
    documents = [
        {"content": "text1", "id": "1", "embedding": np.random.rand(768).astype(np.float32)},
        {"content": "text2", "id": "2", "embedding": np.random.rand(768).astype(np.float64)},
        {"content": "text3", "id": "3", "embedding": np.random.rand(768).astype(np.float32).tolist()},
        {"content": "text4", "id": "4", "embedding": np.random.rand(768).astype(np.float32)},
    ]
    document_store.write_documents(documents)
    assert len(document_store.get_all_documents()) == 4

    if not isinstance(document_store, WeaviateDocumentStore):
        # weaviate is excluded because it would return dummy vectors instead of None
        documents_without_embedding = document_store.get_all_documents(return_embedding=False)
        assert documents_without_embedding[0].embedding is None

    documents_with_embedding = document_store.get_all_documents(return_embedding=True)
    assert isinstance(documents_with_embedding[0].embedding, (list, np.ndarray))


@pytest.mark.parametrize(
    "document_store", ["elasticsearch", "faiss", "memory", "milvus1", "milvus", "weaviate"], indirect=True
)
@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
def test_update_embeddings(document_store, retriever):
    documents = []
    for i in range(6):
        documents.append({"content": f"text_{i}", "id": str(i), "meta_field": f"value_{i}"})
    documents.append({"content": "text_0", "id": "6", "meta_field": "value_0"})

    document_store.write_documents(documents)
    document_store.update_embeddings(retriever, batch_size=3)
    documents = document_store.get_all_documents(return_embedding=True)
    assert len(documents) == 7
    for doc in documents:
        assert type(doc.embedding) is np.ndarray

    documents = document_store.get_all_documents(filters={"meta_field": ["value_0"]}, return_embedding=True)
    assert len(documents) == 2
    for doc in documents:
        assert doc.meta["meta_field"] == "value_0"
    np.testing.assert_array_almost_equal(documents[0].embedding, documents[1].embedding, decimal=4)

    documents = document_store.get_all_documents(filters={"meta_field": ["value_0", "value_5"]}, return_embedding=True)
    documents_with_value_0 = [doc for doc in documents if doc.meta["meta_field"] == "value_0"]
    documents_with_value_5 = [doc for doc in documents if doc.meta["meta_field"] == "value_5"]
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        documents_with_value_0[0].embedding,
        documents_with_value_5[0].embedding,
    )

    doc = {
        "content": "text_7",
        "id": "7",
        "meta_field": "value_7",
        "embedding": retriever.embed_queries(queries=["a random string"])[0],
    }
    document_store.write_documents([doc])

    documents = []
    for i in range(8, 11):
        documents.append({"content": f"text_{i}", "id": str(i), "meta_field": f"value_{i}"})
    document_store.write_documents(documents)

    doc_before_update = document_store.get_all_documents(filters={"meta_field": ["value_7"]})[0]
    embedding_before_update = doc_before_update.embedding

    # test updating only documents without embeddings
    if not isinstance(document_store, WeaviateDocumentStore):
        # All the documents in Weaviate store have an embedding by default. "update_existing_embeddings=False" is not allowed
        document_store.update_embeddings(retriever, batch_size=3, update_existing_embeddings=False)
        doc_after_update = document_store.get_all_documents(filters={"meta_field": ["value_7"]})[0]
        embedding_after_update = doc_after_update.embedding
        np.testing.assert_array_equal(embedding_before_update, embedding_after_update)

    # test updating with filters
    if isinstance(document_store, FAISSDocumentStore):
        with pytest.raises(Exception):
            document_store.update_embeddings(
                retriever, update_existing_embeddings=True, filters={"meta_field": ["value"]}
            )
    else:
        document_store.update_embeddings(retriever, batch_size=3, filters={"meta_field": ["value_0", "value_1"]})
        doc_after_update = document_store.get_all_documents(filters={"meta_field": ["value_7"]})[0]
        embedding_after_update = doc_after_update.embedding
        np.testing.assert_array_equal(embedding_before_update, embedding_after_update)

    # test update all embeddings
    document_store.update_embeddings(retriever, batch_size=3, update_existing_embeddings=True)
    assert document_store.get_embedding_count() == 11
    doc_after_update = document_store.get_all_documents(filters={"meta_field": ["value_7"]})[0]
    embedding_after_update = doc_after_update.embedding
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, embedding_before_update, embedding_after_update
    )

    # test update embeddings for newly added docs
    documents = []
    for i in range(12, 15):
        documents.append({"content": f"text_{i}", "id": str(i), "meta_field": f"value_{i}"})
    document_store.write_documents(documents)

    if not isinstance(document_store, WeaviateDocumentStore):
        # All the documents in Weaviate store have an embedding by default. "update_existing_embeddings=False" is not allowed
        document_store.update_embeddings(retriever, batch_size=3, update_existing_embeddings=False)
        assert document_store.get_embedding_count() == 14


@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("retriever", ["table_text_retriever"], indirect=True)
@pytest.mark.embedding_dim(512)
def test_update_embeddings_table_text_retriever(document_store, retriever):
    documents = []
    for i in range(3):
        documents.append(
            {"content": f"text_{i}", "id": f"pssg_{i}", "meta_field": f"value_text_{i}", "content_type": "text"}
        )
        documents.append(
            {
                "content": pd.DataFrame(columns=[f"col_{i}", f"col_{i+1}"], data=[[f"cell_{i}", f"cell_{i+1}"]]),
                "id": f"table_{i}",
                f"meta_field": f"value_table_{i}",
                "content_type": "table",
            }
        )
    documents.append({"content": "text_0", "id": "pssg_4", "meta_field": "value_text_0", "content_type": "text"})
    documents.append(
        {
            "content": pd.DataFrame(columns=["col_0", "col_1"], data=[["cell_0", "cell_1"]]),
            "id": "table_4",
            "meta_field": "value_table_0",
            "content_type": "table",
        }
    )

    document_store.write_documents(documents)
    document_store.update_embeddings(retriever, batch_size=3)
    documents = document_store.get_all_documents(return_embedding=True)
    assert len(documents) == 8
    for doc in documents:
        assert type(doc.embedding) is np.ndarray

    # Check if Documents with same content (text) get same embedding
    documents = document_store.get_all_documents(filters={"meta_field": ["value_text_0"]}, return_embedding=True)
    assert len(documents) == 2
    for doc in documents:
        assert doc.meta["meta_field"] == "value_text_0"
    np.testing.assert_array_almost_equal(documents[0].embedding, documents[1].embedding, decimal=4)

    # Check if Documents with same content (table) get same embedding
    documents = document_store.get_all_documents(filters={"meta_field": ["value_table_0"]}, return_embedding=True)
    assert len(documents) == 2
    for doc in documents:
        assert doc.meta["meta_field"] == "value_table_0"
    np.testing.assert_array_almost_equal(documents[0].embedding, documents[1].embedding, decimal=4)

    # Check if Documents wih different content (text) get different embedding
    documents = document_store.get_all_documents(
        filters={"meta_field": ["value_text_1", "value_text_2"]}, return_embedding=True
    )
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, documents[0].embedding, documents[1].embedding
    )

    # Check if Documents with different content (table) get different embeddings
    documents = document_store.get_all_documents(
        filters={"meta_field": ["value_table_1", "value_table_2"]}, return_embedding=True
    )
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, documents[0].embedding, documents[1].embedding
    )

    # Check if Documents with different content (table + text) get different embeddings
    documents = document_store.get_all_documents(
        filters={"meta_field": ["value_text_1", "value_table_1"]}, return_embedding=True
    )
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, documents[0].embedding, documents[1].embedding
    )


def test_delete_all_documents(document_store_with_docs):
    assert len(document_store_with_docs.get_all_documents()) == 5

    document_store_with_docs.delete_documents()
    documents = document_store_with_docs.get_all_documents()
    assert len(documents) == 0


def test_delete_documents(document_store_with_docs):
    assert len(document_store_with_docs.get_all_documents()) == 5

    document_store_with_docs.delete_documents()
    documents = document_store_with_docs.get_all_documents()
    assert len(documents) == 0


def test_delete_documents_with_filters(document_store_with_docs):
    document_store_with_docs.delete_documents(filters={"meta_field": ["test1", "test2", "test4", "test5"]})
    documents = document_store_with_docs.get_all_documents()
    assert len(documents) == 1
    assert documents[0].meta["meta_field"] == "test3"


def test_delete_documents_by_id(document_store_with_docs):
    import logging

    logging.info(len(document_store_with_docs.get_all_documents()))
    docs_to_delete = document_store_with_docs.get_all_documents(
        filters={"meta_field": ["test1", "test2", "test4", "test5"]}
    )
    logging.info(len(docs_to_delete))
    docs_not_to_delete = document_store_with_docs.get_all_documents(filters={"meta_field": ["test3"]})
    logging.info(len(docs_not_to_delete))

    document_store_with_docs.delete_documents(ids=[doc.id for doc in docs_to_delete])
    all_docs_left = document_store_with_docs.get_all_documents()
    assert len(all_docs_left) == 1
    assert all_docs_left[0].meta["meta_field"] == "test3"

    all_ids_left = [doc.id for doc in all_docs_left]
    assert all(doc.id in all_ids_left for doc in docs_not_to_delete)


def test_delete_documents_by_id_with_filters(document_store_with_docs):
    docs_to_delete = document_store_with_docs.get_all_documents(filters={"meta_field": ["test1", "test2"]})
    docs_not_to_delete = document_store_with_docs.get_all_documents(filters={"meta_field": ["test3"]})

    document_store_with_docs.delete_documents(ids=[doc.id for doc in docs_to_delete], filters={"meta_field": ["test1"]})

    all_docs_left = document_store_with_docs.get_all_documents()
    assert len(all_docs_left) == 4
    assert all(doc.meta["meta_field"] != "test1" for doc in all_docs_left)

    all_ids_left = [doc.id for doc in all_docs_left]
    assert all(doc.id in all_ids_left for doc in docs_not_to_delete)


# exclude weaviate because it does not support storing labels
@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "milvus1", "pinecone"], indirect=True)
def test_labels(document_store: BaseDocumentStore):
    label = Label(
        query="question1",
        answer=Answer(
            answer="answer",
            type="extractive",
            score=0.0,
            context="something",
            offsets_in_document=[Span(start=12, end=14)],
            offsets_in_context=[Span(start=12, end=14)],
        ),
        is_correct_answer=True,
        is_correct_document=True,
        document=Document(content="something", id="123"),
        origin="gold-label",
    )
    document_store.write_labels([label])
    labels = document_store.get_all_labels()
    assert len(labels) == 1
    assert label == labels[0]

    # different index
    document_store.write_labels([label], index="another_index")
    labels = document_store.get_all_labels(index="another_index")
    assert len(labels) == 1
    document_store.delete_labels(index="another_index")
    labels = document_store.get_all_labels(index="another_index")
    assert len(labels) == 0
    labels = document_store.get_all_labels()
    assert len(labels) == 1

    # write second label + duplicate
    label2 = Label(
        query="question2",
        answer=Answer(
            answer="another answer",
            type="extractive",
            score=0.0,
            context="something",
            offsets_in_document=[Span(start=12, end=14)],
            offsets_in_context=[Span(start=12, end=14)],
        ),
        is_correct_answer=True,
        is_correct_document=True,
        document=Document(content="something", id="324"),
        origin="gold-label",
    )
    document_store.write_labels([label, label2])
    labels = document_store.get_all_labels()

    # check that second label has been added but not the duplicate
    assert len(labels) == 2
    assert label in labels
    assert label2 in labels

    # delete filtered label2 by id
    document_store.delete_labels(ids=[label2.id])
    labels = document_store.get_all_labels()
    assert label == labels[0]
    assert len(labels) == 1

    # re-add label2
    document_store.write_labels([label2])
    labels = document_store.get_all_labels()
    assert len(labels) == 2

    # delete filtered label2 by query text
    document_store.delete_labels(filters={"query": [label2.query]})
    labels = document_store.get_all_labels()
    assert label == labels[0]
    assert len(labels) == 1

    # re-add label2
    document_store.write_labels([label2])
    labels = document_store.get_all_labels()
    assert len(labels) == 2

    # delete intersection of filters and ids, which is empty
    document_store.delete_labels(ids=[label.id], filters={"query": [label2.query]})
    labels = document_store.get_all_labels()
    assert len(labels) == 2
    assert label in labels
    assert label2 in labels

    # delete all labels
    document_store.delete_labels()
    labels = document_store.get_all_labels()
    assert len(labels) == 0


# exclude weaviate because it does not support storing labels
@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "milvus1", "pinecone"], indirect=True)
def test_multilabel(document_store: BaseDocumentStore):
    labels = [
        Label(
            id="standard",
            query="question",
            answer=Answer(answer="answer1", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        # different answer in same doc
        Label(
            id="diff-answer-same-doc",
            query="question",
            answer=Answer(answer="answer2", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        # answer in different doc
        Label(
            id="diff-answer-diff-doc",
            query="question",
            answer=Answer(answer="answer3", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some other", id="333"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        # 'no answer', should be excluded from MultiLabel
        Label(
            id="4-no-answer",
            query="question",
            answer=Answer(answer="", offsets_in_document=[Span(start=0, end=0)]),
            document=Document(content="some", id="777"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        # is_correct_answer=False, should be excluded from MultiLabel if "drop_negatives = True"
        Label(
            id="5-negative",
            query="question",
            answer=Answer(answer="answer5", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=False,
            is_correct_document=True,
            origin="gold-label",
        ),
    ]
    document_store.write_labels(labels)
    # regular labels - not aggregated
    list_labels = document_store.get_all_labels()
    assert set(list_labels) == set(labels)
    assert len(list_labels) == 5

    # Currently we don't enforce writing (missing) docs automatically when adding labels and there's no DB relationship between the two.
    # We should introduce this when we refactored the logic of "index" to be rather a "collection" of labels+documents
    # docs = document_store.get_all_documents()
    # assert len(docs) == 3

    # Multi labels (open domain)
    multi_labels_open = document_store.get_all_labels_aggregated(open_domain=True, drop_negative_labels=True)

    # for open-domain we group all together as long as they have the same question
    assert len(multi_labels_open) == 1
    # all labels are in there except the negative one and the no_answer
    assert len(multi_labels_open[0].labels) == 4
    assert len(multi_labels_open[0].answers) == 3
    assert "5-negative" not in [l.id for l in multi_labels_open[0].labels]
    assert len(multi_labels_open[0].document_ids) == 3

    # Don't drop the negative label
    multi_labels_open = document_store.get_all_labels_aggregated(
        open_domain=True, drop_no_answers=False, drop_negative_labels=False
    )
    assert len(multi_labels_open[0].labels) == 5
    assert len(multi_labels_open[0].answers) == 4
    assert len(multi_labels_open[0].document_ids) == 4

    # Drop no answer + negative
    multi_labels_open = document_store.get_all_labels_aggregated(
        open_domain=True, drop_no_answers=True, drop_negative_labels=True
    )
    assert len(multi_labels_open[0].labels) == 3
    assert len(multi_labels_open[0].answers) == 3
    assert len(multi_labels_open[0].document_ids) == 3

    # for closed domain we group by document so we expect 3 multilabels with 2,1,1 labels each (negative dropped again)
    multi_labels = document_store.get_all_labels_aggregated(open_domain=False, drop_negative_labels=True)
    assert len(multi_labels) == 3
    label_counts = set([len(ml.labels) for ml in multi_labels])
    assert label_counts == set([2, 1, 1])

    assert len(multi_labels[0].answers) == len(multi_labels[0].document_ids)


# exclude weaviate because it does not support storing labels
@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "milvus1", "pinecone"], indirect=True)
def test_multilabel_no_answer(document_store: BaseDocumentStore):
    labels = [
        Label(
            query="question",
            answer=Answer(answer=""),
            is_correct_answer=True,
            is_correct_document=True,
            document=Document(content="some", id="777"),
            origin="gold-label",
        ),
        # no answer in different doc
        Label(
            query="question",
            answer=Answer(answer=""),
            is_correct_answer=True,
            is_correct_document=True,
            document=Document(content="some", id="123"),
            origin="gold-label",
        ),
        # no answer in same doc, should be excluded
        Label(
            query="question",
            answer=Answer(answer=""),
            is_correct_answer=True,
            is_correct_document=True,
            document=Document(content="some", id="777"),
            origin="gold-label",
        ),
        # no answer with is_correct_answer=False, should be excluded
        Label(
            query="question",
            answer=Answer(answer=""),
            is_correct_answer=False,
            is_correct_document=True,
            document=Document(content="some", id="777"),
            origin="gold-label",
        ),
    ]

    document_store.write_labels(labels)

    labels = document_store.get_all_labels()
    assert len(labels) == 4

    multi_labels = document_store.get_all_labels_aggregated(
        open_domain=True, drop_no_answers=False, drop_negative_labels=True
    )
    assert len(multi_labels) == 1
    assert multi_labels[0].no_answer == True
    assert len(multi_labels[0].document_ids) == 0
    assert len(multi_labels[0].answers) == 1

    multi_labels = document_store.get_all_labels_aggregated(
        open_domain=True, drop_no_answers=False, drop_negative_labels=False
    )
    assert len(multi_labels) == 1
    assert multi_labels[0].no_answer == True
    assert len(multi_labels[0].document_ids) == 0
    assert len(multi_labels[0].labels) == 3
    assert len(multi_labels[0].answers) == 1


# exclude weaviate because it does not support storing labels
# exclude faiss and milvus as label metadata is not implemented
@pytest.mark.parametrize("document_store", ["elasticsearch", "memory"], indirect=True)
def test_multilabel_filter_aggregations(document_store: BaseDocumentStore):
    labels = [
        Label(
            id="standard",
            query="question",
            answer=Answer(answer="answer1", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
            filters={"name": ["123"]},
        ),
        # different answer in same doc
        Label(
            id="diff-answer-same-doc",
            query="question",
            answer=Answer(answer="answer2", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
            filters={"name": ["123"]},
        ),
        # answer in different doc
        Label(
            id="diff-answer-diff-doc",
            query="question",
            answer=Answer(answer="answer3", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some other", id="333"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
            filters={"name": ["333"]},
        ),
        # 'no answer', should be excluded from MultiLabel
        Label(
            id="4-no-answer",
            query="question",
            answer=Answer(answer="", offsets_in_document=[Span(start=0, end=0)]),
            document=Document(content="some", id="777"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
            filters={"name": ["777"]},
        ),
        # is_correct_answer=False, should be excluded from MultiLabel if "drop_negatives = True"
        Label(
            id="5-negative",
            query="question",
            answer=Answer(answer="answer5", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=False,
            is_correct_document=True,
            origin="gold-label",
            filters={"name": ["123"]},
        ),
    ]
    document_store.write_labels(labels)
    # regular labels - not aggregated
    list_labels = document_store.get_all_labels()
    assert list_labels == labels
    assert len(list_labels) == 5

    # Multi labels (open domain)
    multi_labels_open = document_store.get_all_labels_aggregated(open_domain=True, drop_negative_labels=True)

    # for open-domain we group all together as long as they have the same question and filters
    assert len(multi_labels_open) == 3
    label_counts = set([len(ml.labels) for ml in multi_labels_open])
    assert label_counts == set([2, 1, 1])
    # all labels are in there except the negative one and the no_answer
    assert "5-negative" not in [l.id for multi_label in multi_labels_open for l in multi_label.labels]

    assert len(multi_labels_open[0].answers) == len(multi_labels_open[0].document_ids)

    # for closed domain we group by document so we expect the same as with filters
    multi_labels = document_store.get_all_labels_aggregated(open_domain=False, drop_negative_labels=True)
    assert len(multi_labels) == 3
    label_counts = set([len(ml.labels) for ml in multi_labels])
    assert label_counts == set([2, 1, 1])

    assert len(multi_labels[0].answers) == len(multi_labels[0].document_ids)


# exclude weaviate because it does not support storing labels
# exclude faiss and milvus as label metadata is not implemented
@pytest.mark.parametrize("document_store", ["elasticsearch", "memory"], indirect=True)
def test_multilabel_meta_aggregations(document_store: BaseDocumentStore):
    labels = [
        Label(
            id="standard",
            query="question",
            answer=Answer(answer="answer1", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
            meta={"file_id": ["123"]},
        ),
        # different answer in same doc
        Label(
            id="diff-answer-same-doc",
            query="question",
            answer=Answer(answer="answer2", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
            meta={"file_id": ["123"]},
        ),
        # answer in different doc
        Label(
            id="diff-answer-diff-doc",
            query="question",
            answer=Answer(answer="answer3", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some other", id="333"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
            meta={"file_id": ["333"]},
        ),
        # 'no answer', should be excluded from MultiLabel
        Label(
            id="4-no-answer",
            query="question",
            answer=Answer(answer="", offsets_in_document=[Span(start=0, end=0)]),
            document=Document(content="some", id="777"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
            meta={"file_id": ["777"]},
        ),
        # is_correct_answer=False, should be excluded from MultiLabel if "drop_negatives = True"
        Label(
            id="5-888",
            query="question",
            answer=Answer(answer="answer5", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
            meta={"file_id": ["888"]},
        ),
    ]
    document_store.write_labels(labels)
    # regular labels - not aggregated
    list_labels = document_store.get_all_labels()
    assert list_labels == labels
    assert len(list_labels) == 5

    # Multi labels (open domain)
    multi_labels_open = document_store.get_all_labels_aggregated(open_domain=True, drop_negative_labels=True)

    # for open-domain we group all together as long as they have the same question and filters
    assert len(multi_labels_open) == 1
    assert len(multi_labels_open[0].labels) == 5

    multi_labels = document_store.get_all_labels_aggregated(
        open_domain=True, drop_negative_labels=True, aggregate_by_meta="file_id"
    )
    assert len(multi_labels) == 4
    label_counts = set([len(ml.labels) for ml in multi_labels])
    assert label_counts == set([2, 1, 1, 1])
    for multi_label in multi_labels:
        for l in multi_label.labels:
            assert l.filters == l.meta
            assert multi_label.filters == l.filters


@pytest.mark.parametrize(
    "document_store", ["elasticsearch", "faiss", "milvus1", "weaviate", "pinecone", "memory"], indirect=True
)
def test_update_meta(document_store: BaseDocumentStore):
    documents = [
        Document(content="Doc1", meta={"meta_key_1": "1", "meta_key_2": "1"}),
        Document(content="Doc2", meta={"meta_key_1": "2", "meta_key_2": "2"}),
        Document(content="Doc3", meta={"meta_key_1": "3", "meta_key_2": "3"}),
    ]
    document_store.write_documents(documents)
    document_2 = document_store.get_all_documents(filters={"meta_key_2": ["2"]})[0]
    document_store.update_document_meta(document_2.id, meta={"meta_key_1": "99", "meta_key_2": "2"})
    updated_document = document_store.get_document_by_id(document_2.id)
    assert len(updated_document.meta.keys()) == 2
    assert updated_document.meta["meta_key_1"] == "99"
    assert updated_document.meta["meta_key_2"] == "2"


@pytest.mark.parametrize("document_store_type", ["elasticsearch", "memory"])
def test_custom_embedding_field(document_store_type, tmp_path):
    document_store = get_document_store(
        document_store_type=document_store_type,
        tmp_path=tmp_path,
        embedding_field="custom_embedding_field",
        index="custom_embedding_field",
    )
    doc_to_write = {"content": "test", "custom_embedding_field": np.random.rand(768).astype(np.float32)}
    document_store.write_documents([doc_to_write])
    documents = document_store.get_all_documents(return_embedding=True)
    assert len(documents) == 1
    assert documents[0].content == "test"
    np.testing.assert_array_equal(doc_to_write["custom_embedding_field"], documents[0].embedding)


@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
def test_get_meta_values_by_key(document_store: BaseDocumentStore):
    documents = [Document(content=f"Doc{i}", meta={"meta_key_1": f"{i}", "meta_key_2": f"{i}{i}"}) for i in range(20)]
    document_store.write_documents(documents)

    # test without filters or query
    result = document_store.get_metadata_values_by_key(key="meta_key_1")
    possible_values = [f"{i}" for i in range(20)]
    assert len(result) == 20
    for bucket in result:
        assert bucket["value"] in possible_values
        assert bucket["count"] == 1

    # test with filters but no query
    result = document_store.get_metadata_values_by_key(key="meta_key_1", filters={"meta_key_2": ["11", "22"]})
    for bucket in result:
        assert bucket["value"] in ["1", "2"]
        assert bucket["count"] == 1

    # test with filters & query
    result = document_store.get_metadata_values_by_key(key="meta_key_1", query="Doc1")
    for bucket in result:
        assert bucket["value"] in ["1"]
        assert bucket["count"] == 1


@pytest.mark.elasticsearch
def test_elasticsearch_custom_fields():
    document_store = ElasticsearchDocumentStore(
        index="haystack_test_custom",
        content_field="custom_text_field",
        embedding_field="custom_embedding_field",
        recreate_index=True,
    )

    doc_to_write = {"custom_text_field": "test", "custom_embedding_field": np.random.rand(768).astype(np.float32)}
    document_store.write_documents([doc_to_write])
    documents = document_store.get_all_documents(return_embedding=True)
    assert len(documents) == 1
    assert documents[0].content == "test"
    np.testing.assert_array_equal(doc_to_write["custom_embedding_field"], documents[0].embedding)


@pytest.mark.elasticsearch
def test_elasticsearch_delete_index():
    client = Elasticsearch()
    index_name = "haystack_test_deletion"

    document_store = ElasticsearchDocumentStore(index=index_name)

    # the index should exist
    index_exists = client.indices.exists(index=index_name)
    assert index_exists

    document_store.delete_index(index_name)

    # the index was deleted and should not exist
    index_exists = client.indices.exists(index=index_name)
    assert not index_exists


@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
def test_elasticsearch_query_with_filters_and_missing_embeddings(document_store: ElasticsearchDocumentStore):
    document_store.write_documents(DOCUMENTS)
    document_without_embedding = Document(
        content="Doc without embedding", meta={"name": "name_7", "year": "2021", "month": "04"}
    )
    document_store.write_documents([document_without_embedding])
    filters = {"year": "2021"}
    document_store.skip_missing_embeddings = False
    with pytest.raises(RequestError):
        document_store.query_by_embedding(np.random.rand(768), filters=filters)

    document_store.skip_missing_embeddings = True
    documents = document_store.query_by_embedding(np.random.rand(768), filters=filters)
    assert len(documents) == 3


@pytest.mark.elasticsearch
def test_get_document_count_only_documents_without_embedding_arg():
    documents = [
        {
            "content": "text1",
            "id": "1",
            "embedding": np.random.rand(768).astype(np.float32),
            "meta_field_for_count": "a",
        },
        {
            "content": "text2",
            "id": "2",
            "embedding": np.random.rand(768).astype(np.float64),
            "meta_field_for_count": "b",
        },
        {"content": "text3", "id": "3", "embedding": np.random.rand(768).astype(np.float32).tolist()},
        {"content": "text4", "id": "4", "meta_field_for_count": "b"},
        {"content": "text5", "id": "5", "meta_field_for_count": "b"},
        {"content": "text6", "id": "6", "meta_field_for_count": "c"},
        {
            "content": "text7",
            "id": "7",
            "embedding": np.random.rand(768).astype(np.float64),
            "meta_field_for_count": "c",
        },
    ]

    _index: str = "haystack_test_count"
    document_store = ElasticsearchDocumentStore(index=_index, recreate_index=True)

    document_store.write_documents(documents)

    assert document_store.get_document_count() == 7
    assert document_store.get_document_count(only_documents_without_embedding=True) == 3
    assert (
        document_store.get_document_count(
            only_documents_without_embedding=True, filters={"meta_field_for_count": ["c"]}
        )
        == 1
    )
    assert (
        document_store.get_document_count(
            only_documents_without_embedding=True, filters={"meta_field_for_count": ["b"]}
        )
        == 2
    )


@pytest.mark.elasticsearch
def test_skip_missing_embeddings(caplog):
    documents = [
        {"content": "text1", "id": "1"},  # a document without embeddings
        {"content": "text2", "id": "2", "embedding": np.random.rand(768).astype(np.float64)},
        {"content": "text3", "id": "3", "embedding": np.random.rand(768).astype(np.float32).tolist()},
        {"content": "text4", "id": "4", "embedding": np.random.rand(768).astype(np.float32)},
    ]
    document_store = ElasticsearchDocumentStore(index="skip_missing_embedding_index", recreate_index=True)
    document_store.write_documents(documents)

    document_store.skip_missing_embeddings = True
    retrieved_docs = document_store.query_by_embedding(np.random.rand(768).astype(np.float32))
    assert len(retrieved_docs) == 3

    document_store.skip_missing_embeddings = False
    with pytest.raises(RequestError):
        document_store.query_by_embedding(np.random.rand(768).astype(np.float32))

    # Test scenario with no embeddings for the entire index
    documents = [
        {"content": "text1", "id": "1"},
        {"content": "text2", "id": "2"},
        {"content": "text3", "id": "3"},
        {"content": "text4", "id": "4"},
    ]

    document_store.delete_documents()
    document_store.write_documents(documents)

    document_store.skip_missing_embeddings = True
    with caplog.at_level(logging.WARNING):
        document_store.query_by_embedding(np.random.rand(768).astype(np.float32))
        assert "No documents with embeddings. Run the document store's update_embeddings() method." in caplog.text


@pytest.mark.elasticsearch
def test_elasticsearch_synonyms():
    synonyms = ["i-pod, i pod, ipod", "sea biscuit, sea biscit, seabiscuit", "foo, foo bar, baz"]
    synonym_type = "synonym_graph"

    client = Elasticsearch()
    client.indices.delete(index="haystack_synonym_arg", ignore=[404])
    document_store = ElasticsearchDocumentStore(
        index="haystack_synonym_arg", synonyms=synonyms, synonym_type=synonym_type
    )
    indexed_settings = client.indices.get_settings(index="haystack_synonym_arg")

    assert (
        synonym_type
        == indexed_settings["haystack_synonym_arg"]["settings"]["index"]["analysis"]["filter"]["synonym"]["type"]
    )
    assert (
        synonyms
        == indexed_settings["haystack_synonym_arg"]["settings"]["index"]["analysis"]["filter"]["synonym"]["synonyms"]
    )


@pytest.mark.parametrize(
    "document_store_with_docs", ["memory", "faiss", "milvus1", "weaviate", "elasticsearch"], indirect=True
)
@pytest.mark.embedding_dim(384)
def test_similarity_score_sentence_transformers(document_store_with_docs):
    retriever = EmbeddingRetriever(
        document_store=document_store_with_docs, embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    document_store_with_docs.update_embeddings(retriever)
    pipeline = DocumentSearchPipeline(retriever)
    prediction = pipeline.run("Paul lives in New York")
    scores = [document.score for document in prediction["documents"]]
    assert [document.content for document in prediction["documents"]] == [
        "My name is Paul and I live in New York",
        "My name is Matteo and I live in Rome",
        "My name is Christelle and I live in Paris",
        "My name is Carla and I live in Berlin",
        "My name is Camila and I live in Madrid",
    ]
    assert scores == pytest.approx(
        [0.9149981737136841, 0.6895168423652649, 0.641706794500351, 0.6206043660640717, 0.5837393924593925], abs=1e-3
    )


@pytest.mark.parametrize(
    "document_store_with_docs", ["memory", "faiss", "milvus1", "weaviate", "elasticsearch"], indirect=True
)
@pytest.mark.embedding_dim(384)
def test_similarity_score(document_store_with_docs):
    retriever = EmbeddingRetriever(
        document_store=document_store_with_docs,
        embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_format="farm",
    )
    document_store_with_docs.update_embeddings(retriever)
    pipeline = DocumentSearchPipeline(retriever)
    prediction = pipeline.run("Paul lives in New York")
    scores = [document.score for document in prediction["documents"]]
    assert scores == pytest.approx(
        [0.9102507941407827, 0.6937791467877008, 0.6491682889305038, 0.6321622491318529, 0.5909129441370939], abs=1e-3
    )


@pytest.mark.parametrize(
    "document_store_with_docs", ["memory", "faiss", "milvus1", "weaviate", "elasticsearch"], indirect=True
)
@pytest.mark.embedding_dim(384)
def test_similarity_score_without_scaling(document_store_with_docs):
    retriever = EmbeddingRetriever(
        document_store=document_store_with_docs,
        embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
        scale_score=False,
        model_format="farm",
    )
    document_store_with_docs.update_embeddings(retriever)
    pipeline = DocumentSearchPipeline(retriever)
    prediction = pipeline.run("Paul lives in New York")
    scores = [document.score for document in prediction["documents"]]
    assert scores == pytest.approx(
        [0.8205015882815654, 0.3875582935754016, 0.29833657786100765, 0.26432449826370585, 0.18182588827418789],
        abs=1e-3,
    )


@pytest.mark.parametrize(
    "document_store_dot_product_with_docs", ["memory", "faiss", "milvus1", "elasticsearch", "weaviate"], indirect=True
)
@pytest.mark.embedding_dim(384)
def test_similarity_score_dot_product(document_store_dot_product_with_docs):
    retriever = EmbeddingRetriever(
        document_store=document_store_dot_product_with_docs,
        embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_format="farm",
    )
    document_store_dot_product_with_docs.update_embeddings(retriever)
    pipeline = DocumentSearchPipeline(retriever)
    prediction = pipeline.run("Paul lives in New York")
    scores = [document.score for document in prediction["documents"]]
    assert scores == pytest.approx(
        [0.5526494403409358, 0.5247784342375555, 0.5189836829440964, 0.5179697273254912, 0.5112024928228626], abs=1e-3
    )


@pytest.mark.parametrize(
    "document_store_dot_product_with_docs", ["memory", "faiss", "milvus1", "elasticsearch", "weaviate"], indirect=True
)
@pytest.mark.embedding_dim(384)
def test_similarity_score_dot_product_without_scaling(document_store_dot_product_with_docs):
    retriever = EmbeddingRetriever(
        document_store=document_store_dot_product_with_docs,
        embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
        scale_score=False,
        model_format="farm",
    )
    document_store_dot_product_with_docs.update_embeddings(retriever)
    pipeline = DocumentSearchPipeline(retriever)
    prediction = pipeline.run("Paul lives in New York")
    scores = [document.score for document in prediction["documents"]]
    assert scores == pytest.approx(
        [21.13810000000001, 9.919499999999971, 7.597099999999955, 7.191000000000031, 4.481750000000034], abs=1e-3
    )


def test_custom_headers(document_store_with_docs: BaseDocumentStore):
    mock_client = None
    if isinstance(document_store_with_docs, ElasticsearchDocumentStore):
        es_document_store: ElasticsearchDocumentStore = document_store_with_docs
        mock_client = Mock(wraps=es_document_store.client)
        es_document_store.client = mock_client
    custom_headers = {"X-My-Custom-Header": "header-value"}
    if not mock_client:
        with pytest.raises(NotImplementedError):
            documents = document_store_with_docs.get_all_documents(headers=custom_headers)
    else:
        documents = document_store_with_docs.get_all_documents(headers=custom_headers)
        mock_client.search.assert_called_once()
        args, kwargs = mock_client.search.call_args
        assert "headers" in kwargs
        assert kwargs["headers"] == custom_headers
        assert len(documents) > 0


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_DeepsetCloudDocumentStore_init_with_dot_product():
    document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=DC_TEST_INDEX)
    assert document_store.return_embedding == False
    assert document_store.similarity == "dot_product"


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_DeepsetCloudDocumentStore_init_with_cosine():
    document_store = DeepsetCloudDocumentStore(
        api_endpoint=DC_API_ENDPOINT,
        api_key=DC_API_KEY,
        index=DC_TEST_INDEX,
        similarity="cosine",
        return_embedding=True,
    )
    assert document_store.return_embedding == True
    assert document_store.similarity == "cosine"


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_DeepsetCloudDocumentStore_invalid_token():
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines",
            match=[matchers.header_matcher({"authorization": "Bearer invalid_token"})],
            body="Internal Server Error",
            status=500,
        )

    with pytest.raises(
        DeepsetCloudError,
        match=f"Could not connect to deepset Cloud:\nGET {DC_API_ENDPOINT}/workspaces/default/pipelines failed: HTTP 500 - Internal Server Error",
    ):
        DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key="invalid_token", index=DC_TEST_INDEX)


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_DeepsetCloudDocumentStore_invalid_api_endpoint():
    if MOCK_DC:
        responses.add(
            method=responses.GET, url=f"{DC_API_ENDPOINT}00/workspaces/default/pipelines", body="Not Found", status=404
        )

    with pytest.raises(
        DeepsetCloudError,
        match=f"Could not connect to deepset Cloud:\nGET {DC_API_ENDPOINT}00/workspaces/default/pipelines failed: "
        f"HTTP 404 - Not Found\nNot Found",
    ):
        DeepsetCloudDocumentStore(api_endpoint=f"{DC_API_ENDPOINT}00", api_key=DC_API_KEY, index=DC_TEST_INDEX)


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_DeepsetCloudDocumentStore_invalid_index(caplog):
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/invalid_index",
            body="Not Found",
            status=404,
        )

    with caplog.at_level(logging.INFO):
        DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index="invalid_index")
        assert (
            "You are using a DeepsetCloudDocumentStore with an index that does not exist on deepset Cloud."
            in caplog.text
        )


@responses.activate
def test_DeepsetCloudDocumentStore_documents(deepset_cloud_document_store):
    if MOCK_DC:
        with open(SAMPLES_PATH / "dc" / "documents-stream.response", "r") as f:
            documents_stream_response = f.read()
            docs = [json.loads(l) for l in documents_stream_response.splitlines()]
            filtered_docs = [doc for doc in docs if doc["meta"]["file_id"] == docs[0]["meta"]["file_id"]]
            documents_stream_filtered_response = "\n".join([json.dumps(d) for d in filtered_docs])

            responses.add(
                method=responses.POST,
                url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents-stream",
                body=documents_stream_response,
                status=200,
            )

            responses.add(
                method=responses.POST,
                url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents-stream",
                match=[
                    matchers.json_params_matcher(
                        {"filters": {"file_id": [docs[0]["meta"]["file_id"]]}, "return_embedding": False}
                    )
                ],
                body=documents_stream_filtered_response,
                status=200,
            )

            for doc in filtered_docs:
                responses.add(
                    method=responses.GET,
                    url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents/{doc['id']}",
                    json=doc,
                    status=200,
                )
    else:
        responses.add_passthru(DC_API_ENDPOINT)

    docs = deepset_cloud_document_store.get_all_documents()
    assert len(docs) > 1
    assert isinstance(docs[0], Document)

    first_doc = next(deepset_cloud_document_store.get_all_documents_generator())
    assert isinstance(first_doc, Document)
    assert first_doc.meta["file_id"] is not None

    filtered_docs = deepset_cloud_document_store.get_all_documents(filters={"file_id": [first_doc.meta["file_id"]]})
    assert len(filtered_docs) > 0
    assert len(filtered_docs) < len(docs)

    ids = [doc.id for doc in filtered_docs]
    single_doc_by_id = deepset_cloud_document_store.get_document_by_id(ids[0])
    assert single_doc_by_id is not None
    assert single_doc_by_id.meta["file_id"] == first_doc.meta["file_id"]

    docs_by_id = deepset_cloud_document_store.get_documents_by_id(ids)
    assert len(docs_by_id) == len(filtered_docs)
    for doc in docs_by_id:
        assert doc.meta["file_id"] == first_doc.meta["file_id"]


@responses.activate
def test_DeepsetCloudDocumentStore_query(deepset_cloud_document_store):
    if MOCK_DC:
        with open(SAMPLES_PATH / "dc" / "query_winterfell.response", "r") as f:
            query_winterfell_response = f.read()
            query_winterfell_docs = json.loads(query_winterfell_response)
            query_winterfell_filtered_docs = [
                doc
                for doc in query_winterfell_docs
                if doc["meta"]["file_id"] == query_winterfell_docs[0]["meta"]["file_id"]
            ]
            query_winterfell_filtered_response = json.dumps(query_winterfell_filtered_docs)

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents-query",
            match=[
                matchers.json_params_matcher(
                    {"query": "winterfell", "top_k": 50, "all_terms_must_match": False, "scale_score": True}
                )
            ],
            status=200,
            body=query_winterfell_response,
        )

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents-query",
            match=[
                matchers.json_params_matcher(
                    {
                        "query": "winterfell",
                        "top_k": 50,
                        "filters": {"file_id": [query_winterfell_docs[0]["meta"]["file_id"]]},
                        "all_terms_must_match": False,
                        "scale_score": True,
                    }
                )
            ],
            status=200,
            body=query_winterfell_filtered_response,
        )
    else:
        responses.add_passthru(DC_API_ENDPOINT)

    docs = deepset_cloud_document_store.query("winterfell", top_k=50)
    assert docs is not None
    assert len(docs) > 0

    first_doc = docs[0]
    filtered_docs = deepset_cloud_document_store.query(
        "winterfell", top_k=50, filters={"file_id": [first_doc.meta["file_id"]]}
    )
    assert len(filtered_docs) > 0
    assert len(filtered_docs) < len(docs)


@pytest.mark.parametrize(
    "body, expected_count",
    [
        (
            {
                "data": [
                    {
                        "evaluation_set_id": str(uuid4()),
                        "name": DC_TEST_INDEX,
                        "created_at": "2022-03-22T13:40:27.535Z",
                        "matched_labels": 2,
                        "total_labels": 10,
                    }
                ],
                "has_more": False,
                "total": 1,
            },
            10,
        ),
        (
            {
                "data": [
                    {
                        "evaluation_set_id": str(uuid4()),
                        "name": DC_TEST_INDEX,
                        "created_at": "2022-03-22T13:40:27.535Z",
                        "matched_labels": 0,
                        "total_labels": 0,
                    }
                ],
                "has_more": False,
                "total": 1,
            },
            0,
        ),
    ],
)
@responses.activate
def test_DeepsetCloudDocumentStore_count_of_labels_for_evaluation_set(
    deepset_cloud_document_store, body: dict, expected_count: int
):
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/evaluation_sets",
            status=200,
            body=json.dumps(body),
        )
    else:
        responses.add_passthru(DC_API_ENDPOINT)

    count = deepset_cloud_document_store.get_label_count(index=DC_TEST_INDEX)
    assert count == expected_count


@responses.activate
def test_DeepsetCloudDocumentStore_count_of_labels_for_evaluation_set_raises_DC_error_when_nothing_found(
    deepset_cloud_document_store,
):
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/evaluation_sets",
            status=200,
            body=json.dumps({"data": [], "has_more": False, "total": 0}),
        )
    else:
        responses.add_passthru(DC_API_ENDPOINT)

    with pytest.raises(DeepsetCloudError, match=f"No evaluation set found with the name {DC_TEST_INDEX}"):
        deepset_cloud_document_store.get_label_count(index=DC_TEST_INDEX)


@responses.activate
def test_DeepsetCloudDocumentStore_lists_evaluation_sets(deepset_cloud_document_store):
    response_evaluation_set = {
        "evaluation_set_id": str(uuid4()),
        "name": DC_TEST_INDEX,
        "created_at": "2022-03-22T13:40:27.535Z",
        "matched_labels": 2,
        "total_labels": 10,
    }
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/evaluation_sets",
            status=200,
            body=json.dumps({"data": [response_evaluation_set], "has_more": False, "total": 1}),
        )
    else:
        responses.add_passthru(DC_API_ENDPOINT)

    evaluation_sets = deepset_cloud_document_store.get_evaluation_sets()
    assert evaluation_sets == [response_evaluation_set]


@responses.activate
def test_DeepsetCloudDocumentStore_fetches_labels_for_evaluation_set(deepset_cloud_document_store):
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/evaluation_sets/{DC_TEST_INDEX}",
            status=200,
            body=json.dumps(
                [
                    {
                        "label_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "query": "What is berlin?",
                        "answer": "biggest city in germany",
                        "answer_start": 0,
                        "answer_end": 0,
                        "meta": {},
                        "context": "Berlin is the biggest city in germany.",
                        "external_file_name": "string",
                        "file_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "state": "Label matching status",
                        "candidates": "Candidates that were found in the label <-> file matching",
                    }
                ]
            ),
        )
    else:
        responses.add_passthru(DC_API_ENDPOINT)

    labels = deepset_cloud_document_store.get_all_labels(index=DC_TEST_INDEX)
    assert labels == [
        Label(
            query="What is berlin?",
            document=Document(content="Berlin is the biggest city in germany."),
            is_correct_answer=True,
            is_correct_document=True,
            origin="user-feedback",
            answer=Answer("biggest city in germany"),
            id="3fa85f64-5717-4562-b3fc-2c963f66afa6",
            pipeline_id=None,
            created_at=None,
            updated_at=None,
            meta={},
            filters={},
        )
    ]


@responses.activate
def test_DeepsetCloudDocumentStore_fetches_labels_for_evaluation_set_raises_deepsetclouderror_when_nothing_found(
    deepset_cloud_document_store,
):
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/evaluation_sets/{DC_TEST_INDEX}",
            status=404,
        )
    else:
        responses.add_passthru(DC_API_ENDPOINT)

    with pytest.raises(DeepsetCloudError, match=f"No evaluation set found with the name {DC_TEST_INDEX}"):
        deepset_cloud_document_store.get_all_labels(index=DC_TEST_INDEX)


@responses.activate
def test_DeepsetCloudDocumentStore_query_by_embedding(deepset_cloud_document_store):
    query_emb = np.random.randn(768)
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents-query",
            match=[
                matchers.json_params_matcher(
                    {"query_emb": query_emb.tolist(), "top_k": 10, "return_embedding": False, "scale_score": True}
                )
            ],
            json=[],
            status=200,
        )
    else:
        responses.add_passthru(DC_API_ENDPOINT)

    emb_docs = deepset_cloud_document_store.query_by_embedding(query_emb)
    assert len(emb_docs) == 0


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_DeepsetCloudDocumentStore_get_all_docs_without_index():
    document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
    assert document_store.get_all_documents() == []


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_DeepsetCloudDocumentStore_get_all_docs_generator_without_index():
    document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
    assert list(document_store.get_all_documents_generator()) == []


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_DeepsetCloudDocumentStore_get_doc_by_id_without_index():
    document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
    assert document_store.get_document_by_id(id="some id") == None


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_DeepsetCloudDocumentStore_get_docs_by_id_without_index():
    document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
    assert document_store.get_documents_by_id(ids=["some id"]) == []


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_DeepsetCloudDocumentStore_get_doc_count_without_index():
    document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
    assert document_store.get_document_count() == 0


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_DeepsetCloudDocumentStore_query_by_emb_without_index():
    query_emb = np.random.randn(768)
    document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
    assert document_store.query_by_embedding(query_emb=query_emb) == []


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_DeepsetCloudDocumentStore_query_without_index():
    document_store = DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=None)
    assert document_store.query(query="some query") == []


@pytest.mark.elasticsearch
def test_elasticsearch_search_field_mapping():

    client = Elasticsearch()
    client.indices.delete(index="haystack_search_field_mapping", ignore=[404])

    index_data = [
        {
            "title": "Green tea components",
            "meta": {
                "content": "The green tea plant contains a range of healthy compounds that make it into the final drink",
                "sub_content": "Drink tip",
            },
            "id": "1",
        },
        {
            "title": "Green tea catechin",
            "meta": {
                "content": "Green tea contains a catechin called epigallocatechin-3-gallate (EGCG).",
                "sub_content": "Ingredients tip",
            },
            "id": "2",
        },
        {
            "title": "Minerals in Green tea",
            "meta": {
                "content": "Green tea also has small amounts of minerals that can benefit your health.",
                "sub_content": "Minerals tip",
            },
            "id": "3",
        },
        {
            "title": "Green tea Benefits",
            "meta": {
                "content": "Green tea does more than just keep you alert, it may also help boost brain function.",
                "sub_content": "Health tip",
            },
            "id": "4",
        },
    ]

    document_store = ElasticsearchDocumentStore(
        index="haystack_search_field_mapping", search_fields=["content", "sub_content"], content_field="title"
    )
    document_store.write_documents(index_data)

    indexed_settings = client.indices.get_mapping(index="haystack_search_field_mapping")

    assert indexed_settings["haystack_search_field_mapping"]["mappings"]["properties"]["content"]["type"] == "text"
    assert indexed_settings["haystack_search_field_mapping"]["mappings"]["properties"]["sub_content"]["type"] == "text"


@pytest.mark.elasticsearch
def test_elasticsearch_existing_alias():

    client = Elasticsearch()
    client.indices.delete(index="haystack_existing_alias_1", ignore=[404])
    client.indices.delete(index="haystack_existing_alias_2", ignore=[404])
    client.indices.delete_alias(index="_all", name="haystack_existing_alias", ignore=[404])

    settings = {"mappings": {"properties": {"content": {"type": "text"}}}}

    client.indices.create(index="haystack_existing_alias_1", body=settings)
    client.indices.create(index="haystack_existing_alias_2", body=settings)

    client.indices.put_alias(
        index="haystack_existing_alias_1,haystack_existing_alias_2", name="haystack_existing_alias"
    )

    # To be valid, all indices related to the alias must have content field of type text
    _ = ElasticsearchDocumentStore(index="haystack_existing_alias", search_fields=["content"])


@pytest.mark.elasticsearch
def test_elasticsearch_existing_alias_missing_fields():

    client = Elasticsearch()
    client.indices.delete(index="haystack_existing_alias_1", ignore=[404])
    client.indices.delete(index="haystack_existing_alias_2", ignore=[404])
    client.indices.delete_alias(index="_all", name="haystack_existing_alias", ignore=[404])

    right_settings = {"mappings": {"properties": {"content": {"type": "text"}}}}

    wrong_settings = {"mappings": {"properties": {"content": {"type": "histogram"}}}}

    client.indices.create(index="haystack_existing_alias_1", body=right_settings)
    client.indices.create(index="haystack_existing_alias_2", body=wrong_settings)

    client.indices.put_alias(
        index="haystack_existing_alias_1,haystack_existing_alias_2", name="haystack_existing_alias"
    )

    with pytest.raises(Exception):
        # wrong field type for "content" in index "haystack_existing_alias_2"
        _ = ElasticsearchDocumentStore(
            index="haystack_existing_alias", search_fields=["content"], content_field="title"
        )


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_elasticsearch_brownfield_support(document_store_with_docs):
    new_document_store = InMemoryDocumentStore()
    new_document_store = elasticsearch_index_to_document_store(
        document_store=new_document_store,
        original_index_name="haystack_test",
        original_content_field="content",
        original_name_field="name",
        included_metadata_fields=["date_field"],
        index="test_brownfield_support",
    )

    original_documents = document_store_with_docs.get_all_documents(index="haystack_test")
    transferred_documents = new_document_store.get_all_documents(index="test_brownfield_support")
    assert len(original_documents) == len(transferred_documents)
    assert all("name" in doc.meta for doc in transferred_documents)
    assert all("date_field" in doc.meta for doc in transferred_documents)
    assert all("meta_field" not in doc.meta for doc in transferred_documents)
    assert all("numeric_field" not in doc.meta for doc in transferred_documents)

    original_content = set([doc.content for doc in original_documents])
    transferred_content = set([doc.content for doc in transferred_documents])
    assert original_content == transferred_content

    # Test transferring docs with PreProcessor
    new_document_store = elasticsearch_index_to_document_store(
        document_store=new_document_store,
        original_index_name="haystack_test",
        original_content_field="content",
        excluded_metadata_fields=["date_field"],
        index="test_brownfield_support_2",
        preprocessor=PreProcessor(split_length=1, split_respect_sentence_boundary=False),
    )
    transferred_documents = new_document_store.get_all_documents(index="test_brownfield_support_2")
    assert all("date_field" not in doc.meta for doc in transferred_documents)
    assert all("name" in doc.meta for doc in transferred_documents)
    assert all("meta_field" in doc.meta for doc in transferred_documents)
    assert all("numeric_field" in doc.meta for doc in transferred_documents)
    # Check if number of transferred_documents is equal to number of unique words.
    assert len(transferred_documents) == len(set(" ".join(original_content).split()))


@pytest.mark.parametrize(
    "document_store",
    ["faiss", "milvus1", "milvus", "weaviate", "opensearch_faiss", "opensearch", "elasticsearch", "memory"],
    indirect=True,
)
def test_cosine_similarity(document_store: BaseDocumentStore):
    # below we will write documents to the store and then query it to see if vectors were normalized or not
    ensure_ids_are_correct_uuids(docs=DOCUMENTS, document_store=document_store)
    document_store.write_documents(documents=DOCUMENTS)

    query = np.random.rand(768).astype(np.float32)
    query_results = document_store.query_by_embedding(
        query_emb=query, top_k=len(DOCUMENTS), return_embedding=True, scale_score=False
    )

    # check if search with cosine similarity returns the correct number of results
    assert len(query_results) == len(DOCUMENTS)

    original_embeddings = {doc["content"]: doc["embedding"] for doc in DOCUMENTS}

    for doc in query_results:
        result_emb = doc.embedding
        original_emb = original_embeddings[doc.content]

        expected_emb = original_emb
        # embeddings of document stores which only support dot product out of the box must be normalized
        if (
            isinstance(document_store, (FAISSDocumentStore, MilvusDocumentStore, WeaviateDocumentStore))
            or type(document_store).name == "Milvus1DocumentStore"
            or isinstance(document_store, OpenSearchDocumentStore)
            and document_store.knn_engine == "faiss"
        ):
            expected_emb = original_emb / np.linalg.norm(original_emb)

        # check if the stored embedding was normalized or not
        np.testing.assert_allclose(
            expected_emb, result_emb, rtol=0.2, atol=5e-07
        )  # high tolerance necessary for Milvus 2

        # check if the score is plausible for cosine similarity
        cosine_score = np.dot(result_emb, query) / (np.linalg.norm(result_emb) * np.linalg.norm(query))
        assert cosine_score == pytest.approx(doc.score, 0.01)


@pytest.mark.parametrize(
    "document_store",
    ["faiss", "milvus1", "milvus", "weaviate", "opensearch_faiss", "opensearch", "elasticsearch", "memory"],
    indirect=True,
)
def test_update_embeddings_cosine_similarity(document_store: BaseDocumentStore):
    # below we will write documents to the store and then query it to see if vectors were normalized
    ensure_ids_are_correct_uuids(docs=DOCUMENTS, document_store=document_store)
    # clear embeddings
    docs = deepcopy(DOCUMENTS)
    for doc in docs:
        doc.pop("embedding")

    document_store.write_documents(documents=docs)
    original_embeddings = {}

    # now check if vectors are normalized when updating embeddings
    class MockRetriever:
        def embed_documents(self, docs):
            embeddings = []
            for doc in docs:
                embedding = np.random.rand(768).astype(np.float32)
                original_embeddings[doc.content] = embedding
                embeddings.append(embedding)
            return np.stack(embeddings)

    retriever = MockRetriever()
    document_store.update_embeddings(retriever=retriever)

    query = np.random.rand(768).astype(np.float32)
    query_results = document_store.query_by_embedding(
        query_emb=query, top_k=len(DOCUMENTS), return_embedding=True, scale_score=False
    )

    # check if search with cosine similarity returns the correct number of results
    assert len(query_results) == len(DOCUMENTS)

    for doc in query_results:
        result_emb = doc.embedding
        original_emb = original_embeddings[doc.content]

        expected_emb = original_emb
        # embeddings of document stores which only support dot product out of the box must be normalized
        if (
            isinstance(document_store, (FAISSDocumentStore, MilvusDocumentStore, WeaviateDocumentStore))
            or type(document_store).name == "Milvus1DocumentStore"
            or isinstance(document_store, OpenSearchDocumentStore)
            and document_store.knn_engine == "faiss"
        ):
            expected_emb = original_emb / np.linalg.norm(original_emb)

        # check if the stored embedding was normalized or not
        np.testing.assert_allclose(
            expected_emb, result_emb, rtol=0.2, atol=5e-07
        )  # high tolerance necessary for Milvus 2

        # check if the score is plausible for cosine similarity
        cosine_score = np.dot(result_emb, query) / (np.linalg.norm(result_emb) * np.linalg.norm(query))
        assert cosine_score == pytest.approx(doc.score, 0.01)


@pytest.mark.parametrize(
    "document_store_small",
    ["faiss", "milvus1", "milvus", "weaviate", "memory", "elasticsearch", "opensearch", "opensearch_faiss"],
    indirect=True,
)
def test_cosine_sanity_check(document_store_small):
    VEC_1 = np.array([0.1, 0.2, 0.3], dtype="float32")
    VEC_2 = np.array([0.4, 0.5, 0.6], dtype="float32")

    # This is the cosine similarity of VEC_1 and VEC_2 calculated using sklearn.metrics.pairwise.cosine_similarity
    # The score is normalized to yield a value between 0 and 1.
    KNOWN_COSINE = 0.9746317
    KNOWN_SCALED_COSINE = (KNOWN_COSINE + 1) / 2

    docs = [{"name": "vec_1", "text": "vec_1", "content": "vec_1", "embedding": VEC_1}]
    ensure_ids_are_correct_uuids(docs=docs, document_store=document_store_small)
    document_store_small.write_documents(documents=docs)

    query_results = document_store_small.query_by_embedding(
        query_emb=VEC_2, top_k=1, return_embedding=True, scale_score=True
    )

    # check if faiss returns the same cosine similarity. Manual testing with faiss yielded 0.9746318
    assert math.isclose(query_results[0].score, KNOWN_SCALED_COSINE, abs_tol=0.0002)

    query_results = document_store_small.query_by_embedding(
        query_emb=VEC_2, top_k=1, return_embedding=True, scale_score=False
    )

    # check if faiss returns the same cosine similarity. Manual testing with faiss yielded 0.9746318
    assert math.isclose(query_results[0].score, KNOWN_COSINE, abs_tol=0.0002)


def test_normalize_embeddings_diff_shapes():
    VEC_1 = np.array([0.1, 0.2, 0.3], dtype="float32")
    BaseDocumentStore.normalize_embedding(VEC_1)
    assert np.linalg.norm(VEC_1) - 1 < 0.01

    VEC_1 = np.array([0.1, 0.2, 0.3], dtype="float32").reshape(1, -1)
    BaseDocumentStore.normalize_embedding(VEC_1)
    assert np.linalg.norm(VEC_1) - 1 < 0.01
