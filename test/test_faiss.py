import numpy as np
import pytest
from haystack import Document
import faiss

from haystack.document_store.faiss import FAISSDocumentStore
from haystack import Finder

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


def test_faiss_index_save_and_load(faiss_document_store):
    faiss_document_store.write_documents(DOCUMENTS)

    # test saving the index
    faiss_document_store.save("haystack_test_faiss")

    # clear existing faiss_index
    faiss_document_store.faiss_index.reset()

    # test faiss index is cleared
    assert faiss_document_store.faiss_index.ntotal == 0

    # test loading the index
    new_document_store = faiss_document_store.load(sql_url="sqlite:///haystack_test.db",
                                             faiss_file_path="haystack_test_faiss")

    # check faiss index is restored
    assert new_document_store.faiss_index.ntotal == len(DOCUMENTS)


@pytest.mark.parametrize("index_buffer_size", [10_000, 2])
@pytest.mark.parametrize("batch_size", [2])
def test_faiss_write_docs(faiss_document_store, index_buffer_size, batch_size):
    faiss_document_store.index_buffer_size = index_buffer_size

    # Write in small batches
    for i in range(0, len(DOCUMENTS), batch_size):
        faiss_document_store.write_documents(DOCUMENTS[i: i + batch_size])

    documents_indexed = faiss_document_store.get_all_documents()

    # test if correct vectors are associated with docs
    for i, doc in enumerate(documents_indexed):
        # we currently don't get the embeddings back when we call document_store.get_all_documents()
        original_doc = [d for d in DOCUMENTS if d["text"] == doc.text][0]
        stored_emb = faiss_document_store.faiss_index.reconstruct(int(doc.meta["vector_id"]))
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        assert np.allclose(original_doc["embedding"], stored_emb, rtol=0.01)

    # test document correctness
    check_data_correctness(documents_indexed, DOCUMENTS)


@pytest.mark.slow
@pytest.mark.parametrize("index_buffer_size", [10_000, 2])
def test_faiss_update_docs(faiss_document_store, index_buffer_size, dpr_retriever):
    # adjust buffer size
    faiss_document_store.index_buffer_size = index_buffer_size

    # initial write
    faiss_document_store.write_documents(DOCUMENTS)

    faiss_document_store.update_embeddings(retriever=dpr_retriever)
    documents_indexed = faiss_document_store.get_all_documents()

    # test if correct vectors are associated with docs
    for i, doc in enumerate(documents_indexed):
        original_doc = [d for d in DOCUMENTS if d["text"] == doc.text][0]
        updated_embedding = dpr_retriever.embed_passages([Document.from_dict(original_doc)])
        stored_emb = faiss_document_store.faiss_index.reconstruct(int(doc.meta["vector_id"]))
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        assert np.allclose(updated_embedding, stored_emb, rtol=0.01)

    # test document correctness
    check_data_correctness(documents_indexed, DOCUMENTS)


def test_faiss_update_with_empty_store(faiss_document_store, dpr_retriever):
    # Call update with empty doc store
    faiss_document_store.update_embeddings(retriever=dpr_retriever)

    # initial write
    faiss_document_store.write_documents(DOCUMENTS)

    documents_indexed = faiss_document_store.get_all_documents()

    # test document correctness
    check_data_correctness(documents_indexed, DOCUMENTS)


def test_faiss_retrieving(faiss_document_store, embedding_retriever):
    faiss_document_store.write_documents(DOCUMENTS)
    result = embedding_retriever.retrieve(query="How to test this?")
    assert len(result) == len(DOCUMENTS)
    assert type(result[0]) == Document


def test_faiss_finding(faiss_document_store, embedding_retriever):
    faiss_document_store.write_documents(DOCUMENTS)
    finder = Finder(reader=None, retriever=embedding_retriever)

    prediction = finder.get_answers_via_similar_questions(question="How to test this?", top_k_retriever=1)

    assert len(prediction.get('answers', [])) == 1


def test_faiss_passing_index_from_outside():
    d = 768
    nlist = 2
    quantizer = faiss.IndexFlatIP(d)
    faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss_index.nprobe = 2
    document_store = FAISSDocumentStore(sql_url="sqlite:///haystack_test_faiss.db", faiss_index=faiss_index)

    document_store.delete_all_documents(index="document")
    # as it is a IVF index we need to train it before adding docs
    document_store.train_index(DOCUMENTS)

    document_store.write_documents(documents=DOCUMENTS, index="document")
    documents_indexed = document_store.get_all_documents(index="document")

    # test document correctness
    check_data_correctness(documents_indexed, DOCUMENTS)
