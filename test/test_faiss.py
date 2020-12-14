import faiss
import numpy as np
import pytest
from haystack import Document
from haystack import Finder
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


def check_data_correctness(documents_indexed, documents_inserted):
    # test if correct vector_ids are assigned
    for i, doc in enumerate(documents_indexed):
        assert doc.meta["vector_id"] == str(i)

    # test if number of documents is correct
    assert len(documents_indexed) == len(documents_inserted)

    # test if two docs have same vector_is assigned
    vector_ids = set()
    for i, doc in enumerate(documents_indexed):
        vector_ids.add(doc.meta["vector_id"])
    assert len(vector_ids) == len(documents_inserted)


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_index_save_and_load(document_store):
    document_store.write_documents(DOCUMENTS)

    # test saving the index
    document_store.save("haystack_test_faiss")

    # clear existing faiss_index
    document_store.faiss_index.reset()

    # test faiss index is cleared
    assert document_store.faiss_index.ntotal == 0

    # test loading the index
    new_document_store = document_store.load(sql_url="sqlite:///haystack_test.db",
                                             faiss_file_path="haystack_test_faiss")

    # check faiss index is restored
    assert new_document_store.faiss_index.ntotal == len(DOCUMENTS)


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
@pytest.mark.parametrize("index_buffer_size", [10_000, 2])
@pytest.mark.parametrize("batch_size", [2])
def test_faiss_write_docs(document_store, index_buffer_size, batch_size):
    document_store.index_buffer_size = index_buffer_size

    # Write in small batches
    for i in range(0, len(DOCUMENTS), batch_size):
        document_store.write_documents(DOCUMENTS[i: i + batch_size])

    documents_indexed = document_store.get_all_documents()

    # test if correct vectors are associated with docs
    for i, doc in enumerate(documents_indexed):
        # we currently don't get the embeddings back when we call document_store.get_all_documents()
        original_doc = [d for d in DOCUMENTS if d["text"] == doc.text][0]
        stored_emb = document_store.faiss_index.reconstruct(int(doc.meta["vector_id"]))
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        assert np.allclose(original_doc["embedding"], stored_emb, rtol=0.01)

    # test document correctness
    check_data_correctness(documents_indexed, DOCUMENTS)


@pytest.mark.slow
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
@pytest.mark.parametrize("index_buffer_size", [10_000, 2])
def test_faiss_update_docs(document_store, index_buffer_size, retriever):
    # adjust buffer size
    document_store.index_buffer_size = index_buffer_size

    # initial write
    document_store.write_documents(DOCUMENTS)

    document_store.update_embeddings(retriever=retriever)
    documents_indexed = document_store.get_all_documents()

    # test if correct vectors are associated with docs
    for i, doc in enumerate(documents_indexed):
        original_doc = [d for d in DOCUMENTS if d["text"] == doc.text][0]
        updated_embedding = retriever.embed_passages([Document.from_dict(original_doc)])
        stored_emb = document_store.faiss_index.reconstruct(int(doc.meta["vector_id"]))
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        assert np.allclose(updated_embedding, stored_emb, rtol=0.01)

    # test document correctness
    check_data_correctness(documents_indexed, DOCUMENTS)


@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_update_with_empty_store(document_store, retriever):
    # Call update with empty doc store
    document_store.update_embeddings(retriever=retriever)

    # initial write
    document_store.write_documents(DOCUMENTS)

    documents_indexed = document_store.get_all_documents()

    # test document correctness
    check_data_correctness(documents_indexed, DOCUMENTS)


@pytest.mark.parametrize("index_factory", ["Flat", "HNSW", "IVF1,Flat"])
def test_faiss_retrieving(index_factory):
    document_store = FAISSDocumentStore(sql_url="sqlite:///haystack_test_faiss.db", faiss_index_factory_str=index_factory)
    document_store.delete_all_documents(index="document")
    if "ivf" in index_factory.lower():
        document_store.train_index(DOCUMENTS)
    document_store.write_documents(DOCUMENTS)
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=False)
    result = retriever.retrieve(query="How to test this?")
    assert len(result) == len(DOCUMENTS)
    assert type(result[0]) == Document


@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_finding(document_store, retriever):
    document_store.write_documents(DOCUMENTS)
    finder = Finder(reader=None, retriever=retriever)

    prediction = finder.get_answers_via_similar_questions(question="How to test this?", top_k_retriever=1)

    assert len(prediction.get('answers', [])) == 1


@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_pipeline(document_store, retriever):
    documents = [
        {"name": "name_1", "text": "text_1", "embedding": np.random.rand(768).astype(np.float32)},
        {"name": "name_2", "text": "text_2", "embedding": np.random.rand(768).astype(np.float32)},
        {"name": "name_3", "text": "text_3", "embedding": np.random.rand(768).astype(np.float64)},
        {"name": "name_4", "text": "text_4", "embedding": np.random.rand(768).astype(np.float32)},
    ]
    document_store.write_documents(documents)
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="FAISS", inputs=["Query"])
    output = pipeline.run(query="How to test this?", top_k_retriever=3)
    assert len(output["documents"]) == 3


def test_faiss_passing_index_from_outside():
    d = 768
    nlist = 2
    quantizer = faiss.IndexFlatIP(d)
    faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss_index.set_direct_map_type(faiss.DirectMap.Hashtable)
    faiss_index.nprobe = 2
    document_store = FAISSDocumentStore(sql_url="sqlite:///haystack_test_faiss.db", faiss_index=faiss_index)

    document_store.delete_all_documents(index="document")
    # as it is a IVF index we need to train it before adding docs
    document_store.train_index(DOCUMENTS)

    document_store.write_documents(documents=DOCUMENTS, index="document")
    documents_indexed = document_store.get_all_documents(index="document")

    # test document correctness
    check_data_correctness(documents_indexed, DOCUMENTS)
