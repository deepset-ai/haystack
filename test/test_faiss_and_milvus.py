import faiss
import numpy as np
import pytest
from haystack import Document
from haystack.pipeline import DocumentSearchPipeline
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.pipeline import Pipeline
from haystack.retriever.dense import EmbeddingRetriever

DOCUMENTS = [
    {"name": "name_1", "text": "text_1", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_2", "text": "text_2", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_3", "text": "text_3", "embedding": np.random.rand(768).astype(np.float64)},
    {"name": "name_4", "text": "text_4", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_5", "text": "text_5", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_6", "text": "text_6", "embedding": np.random.rand(768).astype(np.float64)},
]


def test_faiss_index_save_and_load(tmp_path):
    document_store = FAISSDocumentStore(
        sql_url=f"sqlite:////{tmp_path/'haystack_test.db'}",
        index="haystack_test",
    )
    document_store.write_documents(DOCUMENTS)

    # test saving the index
    document_store.save(tmp_path / "haystack_test_faiss")

    # clear existing faiss_index
    document_store.faiss_indexes[document_store.index].reset()

    # test faiss index is cleared
    assert document_store.faiss_indexes[document_store.index].ntotal == 0

    # test loading the index
    new_document_store = FAISSDocumentStore.load(
        sql_url=f"sqlite:////{tmp_path/'haystack_test.db'}",
        faiss_file_path=tmp_path / "haystack_test_faiss",
        index=document_store.index
    )

    # check faiss index is restored
    assert new_document_store.faiss_indexes[document_store.index].ntotal == len(DOCUMENTS)
    # check if documents are restored
    assert len(new_document_store.get_all_documents()) == len(DOCUMENTS)


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
@pytest.mark.parametrize("index_buffer_size", [10_000, 2])
@pytest.mark.parametrize("batch_size", [2])
def test_faiss_write_docs(document_store, index_buffer_size, batch_size):
    document_store.index_buffer_size = index_buffer_size

    # Write in small batches
    for i in range(0, len(DOCUMENTS), batch_size):
        document_store.write_documents(DOCUMENTS[i: i + batch_size])

    documents_indexed = document_store.get_all_documents()
    assert len(documents_indexed) == len(DOCUMENTS)

    # test if correct vectors are associated with docs
    for i, doc in enumerate(documents_indexed):
        # we currently don't get the embeddings back when we call document_store.get_all_documents()
        original_doc = [d for d in DOCUMENTS if d["text"] == doc.text][0]
        stored_emb = document_store.faiss_indexes[document_store.index].reconstruct(int(doc.meta["vector_id"]))
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        assert np.allclose(original_doc["embedding"], stored_emb, rtol=0.01)


@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
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
@pytest.mark.parametrize("document_store", ["milvus", "faiss"], indirect=True)
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
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_update_with_empty_store(document_store, retriever):
    # Call update with empty doc store
    document_store.update_embeddings(retriever=retriever)

    # initial write
    document_store.write_documents(DOCUMENTS)

    documents_indexed = document_store.get_all_documents()

    assert len(documents_indexed) == len(DOCUMENTS)


@pytest.mark.parametrize("index_factory", ["Flat", "HNSW", "IVF1,Flat"])
def test_faiss_retrieving(index_factory, tmp_path):
    document_store = FAISSDocumentStore(
        sql_url=f"sqlite:////{tmp_path/'test_faiss_retrieving.db'}", faiss_index_factory_str=index_factory
    )

    document_store.delete_all_documents(index="document")
    if "ivf" in index_factory.lower():
        document_store.train_index(DOCUMENTS)
    document_store.write_documents(DOCUMENTS)

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="deepset/sentence_bert",
        use_gpu=False
    )
    result = retriever.retrieve(query="How to test this?")

    assert len(result) == len(DOCUMENTS)
    assert type(result[0]) == Document

    # Cleanup
    document_store.faiss_indexes[document_store.index].reset()


@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
def test_finding(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    pipe = DocumentSearchPipeline(retriever=retriever)

    prediction = pipe.run(query="How to test this?", params={"top_k": 1})

    assert len(prediction.get('documents', [])) == 1


@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss", "milvus"], indirect=True)
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


def test_faiss_passing_index_from_outside(tmp_path):
    d = 768
    nlist = 2
    quantizer = faiss.IndexFlatIP(d)
    index = "haystack_test_1"
    faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss_index.set_direct_map_type(faiss.DirectMap.Hashtable)
    faiss_index.nprobe = 2
    document_store = FAISSDocumentStore(
        sql_url=f"sqlite:////{tmp_path/'haystack_test_faiss.db'}", faiss_index=faiss_index, index=index
    )

    document_store.delete_documents()
    # as it is a IVF index we need to train it before adding docs
    document_store.train_index(DOCUMENTS)

    document_store.write_documents(documents=DOCUMENTS)
    documents_indexed = document_store.get_all_documents()

    # test if vectors ids are associated with docs
    for doc in documents_indexed:
        assert 0 <= int(doc.meta["vector_id"]) <= 7


