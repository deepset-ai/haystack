import numpy as np
import pytest
from haystack import Document
from haystack.pipeline import DocumentSearchPipeline
from haystack.pipeline import Pipeline

DOCUMENTS = [
    {"name": "name_1", "text": "text_1", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_2", "text": "text_2", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_3", "text": "text_3", "embedding": np.random.rand(768).astype(np.float64)},
    {"name": "name_4", "text": "text_4", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_5", "text": "text_5", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_6", "text": "text_6", "embedding": np.random.rand(768).astype(np.float64)},
]

@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["milvus2"], indirect=True)
@pytest.mark.parametrize("batch_size", [4, 6])
def test_update_docs(document_store, retriever, batch_size):
    # initial write
    document_store.write_documents(DOCUMENTS)

    document_store.update_embeddings(retriever=retriever, batch_size=batch_size)
    documents_indexed = document_store.get_all_documents()
    assert len(documents_indexed) == len(DOCUMENTS)

    # test if correct vectors are associated with docs
    for doc in documents_indexed:
        original_doc = [d for d in DOCUMENTS if d["text"] == doc.text][0]
        updated_embedding = retriever.embed_passages([Document.from_dict(original_doc)])
        stored_doc = document_store.get_all_documents(filters={"name": [doc.meta["name"]]})[0]
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        assert np.allclose(updated_embedding, stored_doc.embedding, rtol=0.01)


@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["milvus2"], indirect=True)
def test_update_existing_docs(document_store, retriever):
    document_store.duplicate_documents = "overwrite"
    old_document = Document(text="text_1")
    # initial write
    document_store.write_documents([old_document])
    document_store.update_embeddings(retriever=retriever)
    old_documents_indexed = document_store.get_all_documents()
    assert len(old_documents_indexed) == 1

    # Update document data
    new_document = Document(text="text_2")
    new_document.id = old_document.id
    document_store.write_documents([new_document])
    document_store.update_embeddings(retriever=retriever)
    new_documents_indexed = document_store.get_all_documents()
    assert len(new_documents_indexed) == 1

    assert old_documents_indexed[0].id == new_documents_indexed[0].id
    assert old_documents_indexed[0].text == "text_1"
    assert new_documents_indexed[0].text == "text_2"
    assert not np.allclose(old_documents_indexed[0].embedding, new_documents_indexed[0].embedding, rtol=0.01)


@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["milvus2"], indirect=True)
def test_update_with_empty_store(document_store, retriever):
    # Call update with empty doc store
    document_store.update_embeddings(retriever=retriever)

    # initial write
    document_store.write_documents(DOCUMENTS)

    documents_indexed = document_store.get_all_documents()

    assert len(documents_indexed) == len(DOCUMENTS)


@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["milvus2"], indirect=True)
def test_finding(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    pipe = DocumentSearchPipeline(retriever=retriever)

    prediction = pipe.run(query="How to test this?", params={"top_k": 1})

    assert len(prediction.get('documents', [])) == 1


@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["milvus2"], indirect=True)
def test_pipeline(document_store, retriever):
    documents = [
        {"name": "name_1", "text": "text_1", "embedding": np.random.rand(768).astype(np.float32)},
        {"name": "name_2", "text": "text_2", "embedding": np.random.rand(768).astype(np.float32)},
        {"name": "name_3", "text": "text_3", "embedding": np.random.rand(768).astype(np.float64)},
        {"name": "name_4", "text": "text_4", "embedding": np.random.rand(768).astype(np.float32)},
    ]
    document_store.write_documents(documents)
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="FAISS", inputs=["Query"])
    output = pipeline.run(query="How to test this?", params={"top_k": 3})
    assert len(output["documents"]) == 3
