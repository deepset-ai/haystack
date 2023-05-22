import pytest
import numpy as np

from haystack.schema import Document
from haystack.pipelines import DocumentSearchPipeline

from haystack.pipelines import Pipeline


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


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
@pytest.mark.parametrize("batch_size", [4, 6])
def test_update_docs(document_store, retriever, batch_size):
    # initial write
    document_store.write_documents(DOCUMENTS)

    document_store.update_embeddings(retriever=retriever, batch_size=batch_size)
    documents_indexed = document_store.get_all_documents()
    assert len(documents_indexed) == len(DOCUMENTS)

    # test if correct vectors are associated with docs
    for doc in documents_indexed:
        original_doc = [d for d in DOCUMENTS if d["content"] == doc.content][0]
        updated_embedding = retriever.embed_documents([Document.from_dict(original_doc)])
        stored_doc = document_store.get_all_documents(filters={"name": [doc.meta["name"]]})[0]
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        # original input vec is normalized as faiss only stores normalized vectors
        a = updated_embedding / np.linalg.norm(updated_embedding)
        assert np.allclose(a[0], stored_doc.embedding, rtol=0.2)  # high tolerance was necessary for Milvus 2


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_update_existing_docs(document_store, retriever):
    document_store.duplicate_documents = "overwrite"
    old_document = Document(content="text_1")
    # initial write
    document_store.write_documents([old_document])
    document_store.update_embeddings(retriever=retriever)
    old_documents_indexed = document_store.get_all_documents(return_embedding=True)
    assert len(old_documents_indexed) == 1

    # Update document data
    new_document = Document(content="text_2")
    new_document.id = old_document.id
    document_store.write_documents([new_document])
    document_store.update_embeddings(retriever=retriever)
    new_documents_indexed = document_store.get_all_documents(return_embedding=True)
    assert len(new_documents_indexed) == 1

    assert old_documents_indexed[0].id == new_documents_indexed[0].id
    assert old_documents_indexed[0].content == "text_1"
    assert new_documents_indexed[0].content == "text_2"
    print(type(old_documents_indexed[0].embedding))
    print(type(new_documents_indexed[0].embedding))
    assert not np.allclose(old_documents_indexed[0].embedding, new_documents_indexed[0].embedding, rtol=0.01)


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_update_with_empty_store(document_store, retriever):
    # Call update with empty doc store
    document_store.update_embeddings(retriever=retriever)

    # initial write
    document_store.write_documents(DOCUMENTS)

    documents_indexed = document_store.get_all_documents()

    assert len(documents_indexed) == len(DOCUMENTS)


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_finding(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    pipe = DocumentSearchPipeline(retriever=retriever)

    prediction = pipe.run(query="How to test this?", params={"Retriever": {"top_k": 1}})

    assert len(prediction.get("documents", [])) == 1


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_delete_docs_with_filters_multivalue(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    document_store.delete_documents(filters={"name": ["name_1", "name_2", "name_3", "name_4"]})

    documents = document_store.get_all_documents()
    assert len(documents) == 2
    assert document_store.get_embedding_count() == 2
    assert {doc.meta["name"] for doc in documents} == {"name_5", "name_6"}


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_delete_docs_with_filters(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    document_store.delete_documents(filters={"year": ["2020"]})

    documents = document_store.get_all_documents()
    assert len(documents) == 3
    assert document_store.get_embedding_count() == 3
    assert all("2021" == doc.meta["year"] for doc in documents)


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_delete_docs_with_many_filters(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    document_store.delete_documents(filters={"month": ["01"], "year": ["2020"]})

    documents = document_store.get_all_documents()
    assert len(documents) == 5
    assert document_store.get_embedding_count() == 5
    assert "name_1" not in {doc.meta["name"] for doc in documents}


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_delete_docs_by_id(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6
    doc_ids = [doc.id for doc in document_store.get_all_documents()]
    ids_to_delete = doc_ids[0:3]

    document_store.delete_documents(ids=ids_to_delete)

    documents = document_store.get_all_documents()
    assert len(documents) == len(doc_ids) - len(ids_to_delete)
    assert document_store.get_embedding_count() == len(doc_ids) - len(ids_to_delete)

    remaining_ids = [doc.id for doc in documents]
    assert all(doc_id not in remaining_ids for doc_id in ids_to_delete)


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_delete_docs_by_id_with_filters(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    ids_to_delete = [doc.id for doc in document_store.get_all_documents(filters={"name": ["name_1", "name_2"]})]
    ids_not_to_delete = [
        doc.id for doc in document_store.get_all_documents(filters={"name": ["name_3", "name_4", "name_5", "name_6"]})
    ]

    document_store.delete_documents(ids=ids_to_delete, filters={"name": ["name_1", "name_2", "name_3", "name_4"]})

    documents = document_store.get_all_documents()
    assert len(documents) == len(DOCUMENTS) - len(ids_to_delete)
    assert document_store.get_embedding_count() == len(DOCUMENTS) - len(ids_to_delete)

    assert all(doc.meta["name"] != "name_1" for doc in documents)
    assert all(doc.meta["name"] != "name_2" for doc in documents)

    all_ids_left = [doc.id for doc in documents]
    assert all(doc_id in all_ids_left for doc_id in ids_not_to_delete)


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_get_docs_with_filters_one_value(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    documents = document_store.get_all_documents(filters={"year": ["2020"]})

    assert len(documents) == 3
    assert all("2020" == doc.meta["year"] for doc in documents)


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_get_docs_with_filters_many_values(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    documents = document_store.get_all_documents(filters={"name": ["name_5", "name_6"]})

    assert len(documents) == 2
    assert {doc.meta["name"] for doc in documents} == {"name_5", "name_6"}


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_get_docs_with_many_filters(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    document_store.update_embeddings(retriever=retriever, batch_size=4)
    assert document_store.get_embedding_count() == 6

    documents = document_store.get_all_documents(filters={"month": ["01"], "year": ["2020"]})

    assert len(documents) == 1
    assert "name_1" == documents[0].meta["name"]
    assert "01" == documents[0].meta["month"]
    assert "2020" == documents[0].meta["year"]


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_pipeline(document_store, retriever):
    documents = [
        {"name": "name_1", "content": "text_1", "embedding": np.random.rand(768).astype(np.float32)},
        {"name": "name_2", "content": "text_2", "embedding": np.random.rand(768).astype(np.float32)},
        {"name": "name_3", "content": "text_3", "embedding": np.random.rand(768).astype(np.float64)},
        {"name": "name_4", "content": "text_4", "embedding": np.random.rand(768).astype(np.float32)},
    ]
    document_store.write_documents(documents)
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="FAISS", inputs=["Query"])
    output = pipeline.run(query="How to test this?", params={"FAISS": {"top_k": 3}})
    assert len(output["documents"]) == 3
