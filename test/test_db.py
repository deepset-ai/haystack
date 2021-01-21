import numpy as np
import pytest
from elasticsearch import Elasticsearch

from conftest import get_document_store
from haystack import Document, Label
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore


@pytest.mark.elasticsearch
def test_get_all_documents_without_filters(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents()
    assert all(isinstance(d, Document) for d in documents)
    assert len(documents) == 3
    assert {d.meta["name"] for d in documents} == {"filename1", "filename2", "filename3"}
    assert {d.meta["meta_field"] for d in documents} == {"test1", "test2", "test3"}


@pytest.mark.elasticsearch
def test_get_all_document_filter_duplicate_value(document_store):
    documents = [
        Document(
            text="Doc1",
            meta={"f1": "0"}
        ),
        Document(
            text="Doc1",
            meta={"f1": "1", "meta_id": "0"}
        ),
        Document(
            text="Doc2",
            meta={"f3": "0"}
        )
    ]
    document_store.write_documents(documents)
    documents = document_store.get_all_documents(filters={"f1": ["1"]})
    assert documents[0].text == "Doc1"
    assert len(documents) == 1
    assert {d.meta["meta_id"] for d in documents} == {"0"}


@pytest.mark.elasticsearch
def test_get_all_documents_with_correct_filters(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents(filters={"meta_field": ["test2"]})
    assert len(documents) == 1
    assert documents[0].meta["name"] == "filename2"

    documents = document_store_with_docs.get_all_documents(filters={"meta_field": ["test1", "test3"]})
    assert len(documents) == 2
    assert {d.meta["name"] for d in documents} == {"filename1", "filename3"}
    assert {d.meta["meta_field"] for d in documents} == {"test1", "test3"}


@pytest.mark.elasticsearch
def test_get_all_documents_with_incorrect_filter_name(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents(filters={"incorrect_meta_field": ["test2"]})
    assert len(documents) == 0


@pytest.mark.elasticsearch
def test_get_all_documents_with_incorrect_filter_value(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents(filters={"meta_field": ["incorrect_value"]})
    assert len(documents) == 0


@pytest.mark.elasticsearch
def test_get_documents_by_id(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents()
    doc = document_store_with_docs.get_document_by_id(documents[0].id)
    assert doc.id == documents[0].id
    assert doc.text == documents[0].text


@pytest.mark.elasticsearch
def test_get_document_count(document_store):
    documents = [
        {"text": "text1", "id": "1", "meta_field_for_count": "a"},
        {"text": "text2", "id": "2", "meta_field_for_count": "b"},
        {"text": "text3", "id": "3", "meta_field_for_count": "b"},
        {"text": "text4", "id": "4", "meta_field_for_count": "b"},
    ]
    document_store.write_documents(documents)
    assert document_store.get_document_count() == 4
    assert document_store.get_document_count(filters={"meta_field_for_count": ["a"]}) == 1
    assert document_store.get_document_count(filters={"meta_field_for_count": ["b"]}) == 3


@pytest.mark.elasticsearch
def test_get_all_documents_generator(document_store):
    documents = [
        {"text": "text1", "id": "1", "meta_field_for_count": "a"},
        {"text": "text2", "id": "2", "meta_field_for_count": "b"},
        {"text": "text3", "id": "3", "meta_field_for_count": "b"},
        {"text": "text4", "id": "4", "meta_field_for_count": "b"},
        {"text": "text5", "id": "5", "meta_field_for_count": "b"},
    ]

    document_store.write_documents(documents)
    assert len(list(document_store.get_all_documents_generator(batch_size=2))) == 5


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch", "sql", "faiss"], indirect=True)
@pytest.mark.parametrize("update_existing_documents", [True, False])
def test_update_existing_documents(document_store, update_existing_documents):
    original_docs = [
        {"text": "text1_orig", "id": "1", "meta_field_for_count": "a"},
    ]

    updated_docs = [
        {"text": "text1_new", "id": "1", "meta_field_for_count": "a"},
    ]

    document_store.update_existing_documents = update_existing_documents
    document_store.write_documents(original_docs)
    assert document_store.get_document_count() == 1

    if update_existing_documents:
        document_store.write_documents(updated_docs)
    else:
        with pytest.raises(Exception):
            document_store.write_documents(updated_docs)

    stored_docs = document_store.get_all_documents()
    assert len(stored_docs) == 1
    if update_existing_documents:
        assert stored_docs[0].text == updated_docs[0]["text"]
    else:
        assert stored_docs[0].text == original_docs[0]["text"]


@pytest.mark.elasticsearch
def test_write_document_meta(document_store):
    documents = [
        {"text": "dict_without_meta", "id": "1"},
        {"text": "dict_with_meta", "meta_field": "test2", "name": "filename2", "id": "2"},
        Document(text="document_object_without_meta", id="3"),
        Document(text="document_object_with_meta", meta={"meta_field": "test4", "name": "filename3"}, id="4"),
    ]
    document_store.write_documents(documents)
    documents_in_store = document_store.get_all_documents()
    assert len(documents_in_store) == 4

    assert not document_store.get_document_by_id("1").meta
    assert document_store.get_document_by_id("2").meta["meta_field"] == "test2"
    assert not document_store.get_document_by_id("3").meta
    assert document_store.get_document_by_id("4").meta["meta_field"] == "test4"


@pytest.mark.elasticsearch
def test_write_document_index(document_store):
    documents = [
        {"text": "text1", "id": "1"},
        {"text": "text2", "id": "2"},
    ]
    document_store.write_documents([documents[0]], index="haystack_test_1")
    assert len(document_store.get_all_documents(index="haystack_test_1")) == 1

    document_store.write_documents([documents[1]], index="haystack_test_2")
    assert len(document_store.get_all_documents(index="haystack_test_2")) == 1

    assert len(document_store.get_all_documents(index="haystack_test_1")) == 1
    assert len(document_store.get_all_documents()) == 0


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory"], indirect=True)
def test_document_with_embeddings(document_store):
    documents = [
        {"text": "text1", "id": "1", "embedding": np.random.rand(768).astype(np.float32)},
        {"text": "text2", "id": "2", "embedding": np.random.rand(768).astype(np.float64)},
        {"text": "text3", "id": "3", "embedding": np.random.rand(768).astype(np.float32).tolist()},
        {"text": "text4", "id": "4", "embedding": np.random.rand(768).astype(np.float32)},
    ]
    document_store.write_documents(documents, index="haystack_test_1")
    assert len(document_store.get_all_documents(index="haystack_test_1")) == 4

    documents_without_embedding = document_store.get_all_documents(index="haystack_test_1", return_embedding=False)
    assert documents_without_embedding[0].embedding is None

    documents_with_embedding = document_store.get_all_documents(index="haystack_test_1", return_embedding=True)
    assert isinstance(documents_with_embedding[0].embedding, (list, np.ndarray))


@pytest.mark.parametrize("retriever", ["dpr", "embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory"], indirect=True)
def test_update_embeddings(document_store, retriever):
    documents = []
    for i in range(23):
        documents.append({"text": f"text_{i}", "id": str(i), "meta_field": f"value_{i}"})
    documents.append({"text": "text_0", "id": "23", "meta_field": "value_0"})

    document_store.write_documents(documents, index="haystack_test_1")
    document_store.update_embeddings(retriever, index="haystack_test_1")
    documents = document_store.get_all_documents(index="haystack_test_1", return_embedding=True)
    assert len(documents) == 24
    for doc in documents:
        assert type(doc.embedding) is np.ndarray

    documents = document_store.get_all_documents(
        index="haystack_test_1",
        filters={"meta_field": ["value_0", "value_23"]},
        return_embedding=True,
    )
    np.testing.assert_array_equal(documents[0].embedding, documents[1].embedding)

    documents = document_store.get_all_documents(
        index="haystack_test_1",
        filters={"meta_field": ["value_0", "value_10"]},
        return_embedding=True,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        documents[0].embedding,
        documents[1].embedding
    )


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_delete_documents(document_store_with_docs):
    assert len(document_store_with_docs.get_all_documents()) == 3

    document_store_with_docs.delete_all_documents(index="haystack_test", filters={"meta_field": ["test1", "test2"]})
    documents = document_store_with_docs.get_all_documents()
    assert len(documents) == 1
    assert documents[0].meta["meta_field"] == "test3"

    document_store_with_docs.delete_all_documents(index="haystack_test")
    documents = document_store_with_docs.get_all_documents()
    assert len(documents) == 0


@pytest.mark.elasticsearch
def test_labels(document_store):
    label = Label(
        question="question",
        answer="answer",
        is_correct_answer=True,
        is_correct_document=True,
        document_id="123",
        offset_start_in_doc=12,
        no_answer=False,
        origin="gold_label",
    )
    document_store.write_labels([label], index="haystack_test_label")
    labels = document_store.get_all_labels(index="haystack_test_label")
    assert len(labels) == 1

    labels = document_store.get_all_labels()
    assert len(labels) == 0


@pytest.mark.elasticsearch
def test_multilabel(document_store):
    labels =[
        Label(
            question="question",
            answer="answer1",
            is_correct_answer=True,
            is_correct_document=True,
            document_id="123",
            offset_start_in_doc=12,
            no_answer=False,
            origin="gold_label",
        ),
        # different answer in same doc
        Label(
            question="question",
            answer="answer2",
            is_correct_answer=True,
            is_correct_document=True,
            document_id="123",
            offset_start_in_doc=42,
            no_answer=False,
            origin="gold_label",
        ),
        # answer in different doc
        Label(
            question="question",
            answer="answer3",
            is_correct_answer=True,
            is_correct_document=True,
            document_id="321",
            offset_start_in_doc=7,
            no_answer=False,
            origin="gold_label",
        ),
        # 'no answer', should be excluded from MultiLabel
        Label(
            question="question",
            answer="",
            is_correct_answer=True,
            is_correct_document=True,
            document_id="777",
            offset_start_in_doc=0,
            no_answer=True,
            origin="gold_label",
        ),
        # is_correct_answer=False, should be excluded from MultiLabel
        Label(
            question="question",
            answer="answer5",
            is_correct_answer=False,
            is_correct_document=True,
            document_id="123",
            offset_start_in_doc=99,
            no_answer=True,
            origin="gold_label",
        ),
    ]
    document_store.write_labels(labels, index="haystack_test_multilabel")
    multi_labels = document_store.get_all_labels_aggregated(index="haystack_test_multilabel")
    labels = document_store.get_all_labels(index="haystack_test_multilabel")

    assert len(multi_labels) == 1
    assert len(labels) == 5

    assert len(multi_labels[0].multiple_answers) == 3
    assert len(multi_labels[0].multiple_answers) \
           == len(multi_labels[0].multiple_document_ids) \
           == len(multi_labels[0].multiple_offset_start_in_docs)

    multi_labels = document_store.get_all_labels_aggregated()
    assert len(multi_labels) == 0

    # clean up
    document_store.delete_all_documents(index="haystack_test_multilabel")


@pytest.mark.elasticsearch
def test_multilabel_no_answer(document_store):
    labels = [
        Label(
            question="question",
            answer="",
            is_correct_answer=True,
            is_correct_document=True,
            document_id="777",
            offset_start_in_doc=0,
            no_answer=True,
            origin="gold_label",
        ),
        # no answer in different doc
        Label(
            question="question",
            answer="",
            is_correct_answer=True,
            is_correct_document=True,
            document_id="123",
            offset_start_in_doc=0,
            no_answer=True,
            origin="gold_label",
        ),
        # no answer in same doc, should be excluded
        Label(
            question="question",
            answer="",
            is_correct_answer=True,
            is_correct_document=True,
            document_id="777",
            offset_start_in_doc=0,
            no_answer=True,
            origin="gold_label",
        ),
        # no answer with is_correct_answer=False, should be excluded
        Label(
            question="question",
            answer="",
            is_correct_answer=False,
            is_correct_document=True,
            document_id="321",
            offset_start_in_doc=0,
            no_answer=True,
            origin="gold_label",
        ),
    ]

    document_store.write_labels(labels, index="haystack_test_multilabel_no_answer")
    multi_labels = document_store.get_all_labels_aggregated(index="haystack_test_multilabel_no_answer")
    labels = document_store.get_all_labels(index="haystack_test_multilabel_no_answer")

    assert len(multi_labels) == 1
    assert len(labels) == 4

    assert len(multi_labels[0].multiple_document_ids) == 2
    assert len(multi_labels[0].multiple_answers) \
           == len(multi_labels[0].multiple_document_ids) \
           == len(multi_labels[0].multiple_offset_start_in_docs)

    # clean up
    document_store.delete_all_documents(index="haystack_test_multilabel_no_answer")


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch", "sql"], indirect=True)
def test_elasticsearch_update_meta(document_store):
    documents = [
        Document(
            text="Doc1",
            meta={"meta_key_1": "1", "meta_key_2": "1"}
        ),
        Document(
            text="Doc2",
            meta={"meta_key_1": "2", "meta_key_2": "2"}
        ),
        Document(
            text="Doc3",
            meta={"meta_key_1": "3", "meta_key_2": "3"}
        )
    ]
    document_store.write_documents(documents)
    document_2 = document_store.get_all_documents(filters={"meta_key_2": ["2"]})[0]
    document_store.update_document_meta(document_2.id, meta={"meta_key_1": "99", "meta_key_2": "2"})
    updated_document = document_store.get_document_by_id(document_2.id)
    assert len(updated_document.meta.keys()) == 2
    assert updated_document.meta["meta_key_1"] == "99"
    assert updated_document.meta["meta_key_2"] == "2"


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store_type", ["elasticsearch", "memory"])
def test_custom_embedding_field(document_store_type):
    document_store = get_document_store(
        document_store_type=document_store_type, embedding_field="custom_embedding_field"
    )
    doc_to_write = {"text": "test", "custom_embedding_field": np.random.rand(768).astype(np.float32)}
    document_store.write_documents([doc_to_write])
    documents = document_store.get_all_documents(return_embedding=True)
    assert len(documents) == 1
    assert documents[0].text == "test"
    np.testing.assert_array_equal(doc_to_write["custom_embedding_field"], documents[0].embedding)


@pytest.mark.elasticsearch
def test_elasticsearch_custom_fields(elasticsearch_fixture):
    client = Elasticsearch()
    client.indices.delete(index='haystack_test_custom', ignore=[404])
    document_store = ElasticsearchDocumentStore(index="haystack_test_custom", text_field="custom_text_field",
                                                embedding_field="custom_embedding_field")

    doc_to_write = {"custom_text_field": "test", "custom_embedding_field": np.random.rand(768).astype(np.float32)}
    document_store.write_documents([doc_to_write])
    documents = document_store.get_all_documents(return_embedding=True)
    assert len(documents) == 1
    assert documents[0].text == "test"
    np.testing.assert_array_equal(doc_to_write["custom_embedding_field"], documents[0].embedding)
