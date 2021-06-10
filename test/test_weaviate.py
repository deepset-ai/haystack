import numpy as np
import pytest
from haystack import Document
from conftest import get_document_store
import uuid

embedding_dim = 768

def get_uuid():
    return str(uuid.uuid4())

DOCUMENTS = [
    {"text": "text1", "id":get_uuid(), "key": "a", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    {"text": "text2", "id":get_uuid(), "key": "b", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    {"text": "text3", "id":get_uuid(), "key": "b", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    {"text": "text4", "id":get_uuid(), "key": "b", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    {"text": "text5", "id":get_uuid(), "key": "b", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
]

DOCUMENTS_XS = [
        # current "dict" format for a document
        {"text": "My name is Carla and I live in Berlin", "id":get_uuid(), "meta": {"metafield": "test1", "name": "filename1"}, "embedding": np.random.rand(embedding_dim).astype(np.float32)},
        # meta_field at the top level for backward compatibility
        {"text": "My name is Paul and I live in New York", "id":get_uuid(), "metafield": "test2", "name": "filename2", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
        # Document object for a doc
        Document(text="My name is Christelle and I live in Paris", id=get_uuid(), meta={"metafield": "test3", "name": "filename3"}, embedding=np.random.rand(embedding_dim).astype(np.float32))
    ]

@pytest.fixture(params=["weaviate"])
def document_store_with_docs(request):
    document_store = get_document_store(request.param)
    document_store.write_documents(DOCUMENTS_XS)
    yield document_store
    document_store.delete_all_documents()

@pytest.fixture(params=["weaviate"])
def document_store(request):
    document_store = get_document_store(request.param)
    yield document_store
    document_store.delete_all_documents()

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store_with_docs", ["weaviate"], indirect=True)
def test_get_all_documents_without_filters(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents()
    assert all(isinstance(d, Document) for d in documents)
    assert len(documents) == 3
    assert {d.meta["name"] for d in documents} == {"filename1", "filename2", "filename3"}
    assert {d.meta["metafield"] for d in documents} == {"test1", "test2", "test3"}

@pytest.mark.weaviate
def test_get_all_documents_with_correct_filters(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents(filters={"metafield": ["test2"]})
    assert len(documents) == 1
    assert documents[0].meta["name"] == "filename2"

    documents = document_store_with_docs.get_all_documents(filters={"metafield": ["test1", "test3"]})
    assert len(documents) == 2
    assert {d.meta["name"] for d in documents} == {"filename1", "filename3"}
    assert {d.meta["metafield"] for d in documents} == {"test1", "test3"}

@pytest.mark.weaviate
def test_get_all_documents_with_incorrect_filter_name(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents(filters={"incorrectmetafield": ["test2"]})
    assert len(documents) == 0

@pytest.mark.weaviate
def test_get_all_documents_with_incorrect_filter_value(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents(filters={"metafield": ["incorrect_value"]})
    assert len(documents) == 0

@pytest.mark.weaviate
def test_get_documents_by_id(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents()
    doc = document_store_with_docs.get_document_by_id(documents[0].id)
    assert doc.id == documents[0].id
    assert doc.text == documents[0].text

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
def test_get_document_count(document_store):
    document_store.write_documents(DOCUMENTS)
    assert document_store.get_document_count() == 5
    assert document_store.get_document_count(filters={"key": ["a"]}) == 1
    assert document_store.get_document_count(filters={"key": ["b"]}) == 4

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
@pytest.mark.parametrize("batch_size", [2])
def test_weaviate_write_docs(document_store, batch_size):
    # Write in small batches
    for i in range(0, len(DOCUMENTS), batch_size):
        document_store.write_documents(DOCUMENTS[i: i + batch_size])

    documents_indexed = document_store.get_all_documents()
    assert len(documents_indexed) == len(DOCUMENTS)

    documents_indexed = document_store.get_all_documents(batch_size=batch_size)
    assert len(documents_indexed) == len(DOCUMENTS)

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
def test_get_all_document_filter_duplicate_value(document_store):
    documents = [
        Document(
            text="Doc1",
            meta={"fone": "f0"},
            id = get_uuid(),
            embedding= np.random.rand(embedding_dim).astype(np.float32)
        ),
        Document(
            text="Doc1",
            meta={"fone": "f1", "metaid": "0"},
            id = get_uuid(),
            embedding = np.random.rand(embedding_dim).astype(np.float32)
        ),
        Document(
            text="Doc2",
            meta={"fthree": "f0"},
            id = get_uuid(),
            embedding=np.random.rand(embedding_dim).astype(np.float32)
        )
    ]
    document_store.write_documents(documents)
    documents = document_store.get_all_documents(filters={"fone": ["f1"]})
    assert documents[0].text == "Doc1"
    assert len(documents) == 1
    assert {d.meta["metaid"] for d in documents} == {"0"}

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
def test_get_all_documents_generator(document_store):
    document_store.write_documents(DOCUMENTS)
    assert len(list(document_store.get_all_documents_generator(batch_size=2))) == 5

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
def test_write_with_duplicate_doc_ids(document_store):
    id = get_uuid()
    documents = [
        Document(
            text="Doc1",
            id=id,
            embedding=np.random.rand(embedding_dim).astype(np.float32)
        ),
        Document(
            text="Doc2",
            id=id,
            embedding=np.random.rand(embedding_dim).astype(np.float32)
        )
    ]
    document_store.write_documents(documents, duplicate_documents="skip")
    with pytest.raises(Exception):
        document_store.write_documents(documents, duplicate_documents="fail")

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
@pytest.mark.parametrize("update_existing_documents", [True, False])
def test_update_existing_documents(document_store, update_existing_documents):
    id = uuid.uuid4()
    original_docs = [
        {"text": "text1_orig", "id": id, "metafieldforcount": "a", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    ]

    updated_docs = [
        {"text": "text1_new", "id": id, "metafieldforcount": "a", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    ]

    document_store.update_existing_documents = update_existing_documents
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
        assert stored_docs[0].text == updated_docs[0]["text"]
    else:
        assert stored_docs[0].text == original_docs[0]["text"]

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
def test_write_document_meta(document_store):
    uid1 = get_uuid()
    uid2 = get_uuid()
    uid3 = get_uuid()
    uid4 = get_uuid()
    documents = [
        {"text": "dict_without_meta", "id": uid1, "embedding": np.random.rand(embedding_dim).astype(np.float32)},
        {"text": "dict_with_meta", "metafield": "test2", "name": "filename2", "id": uid2, "embedding": np.random.rand(embedding_dim).astype(np.float32)},
        Document(text="document_object_without_meta", id=uid3, embedding=np.random.rand(embedding_dim).astype(np.float32)),
        Document(text="document_object_with_meta", meta={"metafield": "test4", "name": "filename3"}, id=uid4, embedding=np.random.rand(embedding_dim).astype(np.float32)),
    ]
    document_store.write_documents(documents)
    documents_in_store = document_store.get_all_documents()
    assert len(documents_in_store) == 4

    assert not document_store.get_document_by_id(uid1).meta
    assert document_store.get_document_by_id(uid2).meta["metafield"] == "test2"
    assert not document_store.get_document_by_id(uid3).meta
    assert document_store.get_document_by_id(uid4).meta["metafield"] == "test4"

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
def test_write_document_index(document_store):
    documents = [
        {"text": "text1", "id": uuid.uuid4(), "embedding": np.random.rand(embedding_dim).astype(np.float32)},
        {"text": "text2", "id": uuid.uuid4(), "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    ]

    document_store.write_documents([documents[0]], index="Haystackone")
    assert len(document_store.get_all_documents(index="Haystackone")) == 1

    document_store.write_documents([documents[1]], index="Haystacktwo")
    assert len(document_store.get_all_documents(index="Haystacktwo")) == 1

    assert len(document_store.get_all_documents(index="Haystackone")) == 1
    assert len(document_store.get_all_documents()) == 0

@pytest.mark.weaviate
@pytest.mark.parametrize("retriever", ["dpr", "embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
def test_update_embeddings(document_store, retriever):
    documents = []
    for i in range(6):
        documents.append({"text": f"text_{i}", "id": str(uuid.uuid4()), "metafield": f"value_{i}", "embedding": np.random.rand(embedding_dim).astype(np.float32)})
    documents.append({"text": "text_0", "id": str(uuid.uuid4()), "metafield": "value_0", "embedding": np.random.rand(embedding_dim).astype(np.float32)})

    document_store.write_documents(documents, index="HaystackTestOne")
    document_store.update_embeddings(retriever, index="HaystackTestOne", batch_size=3)
    documents = document_store.get_all_documents(index="HaystackTestOne", return_embedding=True)
    assert len(documents) == 7
    for doc in documents:
        assert type(doc.embedding) is np.ndarray

    documents = document_store.get_all_documents(
        index="HaystackTestOne",
        filters={"metafield": ["value_0"]},
        return_embedding=True,
    )
    assert len(documents) == 2
    for doc in documents:
        assert doc.meta["metafield"] == "value_0"
    np.testing.assert_array_almost_equal(documents[0].embedding, documents[1].embedding, decimal=4)

    documents = document_store.get_all_documents(
        index="HaystackTestOne",
        filters={"metafield": ["value_1", "value_5"]},
        return_embedding=True,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        documents[0].embedding,
        documents[1].embedding
    )

    doc = {"text": "text_7", "id": str(uuid.uuid4()), "metafield": "value_7",
           "embedding": retriever.embed_queries(texts=["a random string"])[0]}
    document_store.write_documents([doc], index="HaystackTestOne")

    doc_before_update = document_store.get_all_documents(index="HaystackTestOne", filters={"metafield": ["value_7"]})[0]
    embedding_before_update = doc_before_update.embedding

    document_store.update_embeddings(
        retriever, index="HaystackTestOne", batch_size=3, filters={"metafield": ["value_0", "value_1"]}
    )
    doc_after_update = document_store.get_all_documents(index="HaystackTestOne", filters={"metafield": ["value_7"]})[0]
    embedding_after_update = doc_after_update.embedding
    np.testing.assert_array_equal(embedding_before_update, embedding_after_update)

    # test update all embeddings
    document_store.update_embeddings(retriever, index="HaystackTestOne", batch_size=3, update_existing_embeddings=True)
    assert document_store.get_document_count(index="HaystackTestOne") == 8
    doc_after_update = document_store.get_all_documents(index="HaystackTestOne", filters={"metafield": ["value_7"]})[0]
    embedding_after_update = doc_after_update.embedding
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, embedding_before_update, embedding_after_update)

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store_with_docs", ["weaviate"], indirect=True)
def test_query_by_embedding(document_store_with_docs):
    docs = document_store_with_docs.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32))
    assert len(docs) == 3

    docs = document_store_with_docs.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32),
                                                       top_k=1)
    assert len(docs) == 1

    docs = document_store_with_docs.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32),
                                                       filters = {"name": ['filename2']})
    assert len(docs) == 1

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store_with_docs", ["weaviate"], indirect=True)
def test_query(document_store_with_docs):
    query_text = 'My name is Carla and I live in Berlin'
    with pytest.raises(Exception):
        docs = document_store_with_docs.query(query_text)

    docs = document_store_with_docs.query(filters = {"name": ['filename2']})
    assert len(docs) == 1

    docs = document_store_with_docs.query(filters={"text":[query_text.lower()]})
    assert len(docs) == 1

    docs = document_store_with_docs.query(filters={"text":['live']})
    assert len(docs) == 3

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store_with_docs", ["weaviate"], indirect=True)
def test_delete_all_documents(document_store_with_docs):
    assert len(document_store_with_docs.get_all_documents()) == 3

    document_store_with_docs.delete_all_documents()
    documents = document_store_with_docs.get_all_documents()
    assert len(documents) == 0

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store_with_docs", ["weaviate"], indirect=True)
def test_delete_documents_with_filters(document_store_with_docs):
    document_store_with_docs.delete_all_documents(filters={"metafield": ["test1", "test2"]})
    documents = document_store_with_docs.get_all_documents()
    assert len(documents) == 1
    assert documents[0].meta["metafield"] == "test3"

