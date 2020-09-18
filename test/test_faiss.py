import numpy as np
import pytest
from haystack import Document

from haystack.retriever.dense import DensePassageRetriever
from haystack.retriever.dense import EmbeddingRetriever
from haystack import Finder

DOCUMENTS = [
    {"name": "name_1", "text": "text_1", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_2", "text": "text_2", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_3", "text": "text_3", "embedding": np.random.rand(768).astype(np.float64)},
]
@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
@pytest.mark.parametrize("index_buffer_size", [10_000, 2])
def test_faiss_write_docs(document_store, index_buffer_size):
    document_store.index_buffer_size = index_buffer_size

    document_store.write_documents(DOCUMENTS)
    documents_indexed = document_store.get_all_documents()

    # test if correct vector_ids are assigned
    for i, doc in enumerate(documents_indexed):
        assert doc.meta["vector_id"] == str(i)

    # test if correct vectors are associated with docs
    for i, doc in enumerate(documents_indexed):
        # we currently don't get the embeddings back when we call document_store.get_all_documents()
        original_doc = [d for d in DOCUMENTS if d["text"] == doc.text][0]
        stored_emb = document_store.faiss_index.reconstruct(int(doc.meta["vector_id"]))
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        assert np.allclose(original_doc["embedding"], stored_emb[:-1], rtol=0.01)

    # test insertion of documents in an existing index fails
    with pytest.raises(Exception):
        document_store.write_documents(DOCUMENTS)

    # test saving the index
    document_store.save("haystack_test_faiss")

    # test loading the index
    document_store.load(sql_url="sqlite:///haystack_test.db", faiss_file_path="haystack_test_faiss")

@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
@pytest.mark.parametrize("index_buffer_size", [10_000, 2])
def test_faiss_update_docs(document_store, index_buffer_size):

    # adjust buffer size
    document_store.index_buffer_size = index_buffer_size

    # initial write
    document_store.write_documents(DOCUMENTS)

    # do the update
    retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      use_gpu=False, embed_title=True,
                                      remove_sep_tok_from_untitled_passages=True)

    document_store.update_embeddings(retriever=retriever)
    documents_indexed = document_store.get_all_documents()

    # test if number of documents is correct
    assert len(documents_indexed) == len(DOCUMENTS)

    # test if two docs have same vector_is assigned
    vector_ids = set()
    for i, doc in enumerate(documents_indexed):
        vector_ids.add(doc.meta["vector_id"])
    assert len(vector_ids) == len(DOCUMENTS)

    # test if correct vectors are associated with docs
    for i, doc in enumerate(documents_indexed):
        original_doc = [d for d in DOCUMENTS if d["text"] == doc.text][0]
        updated_embedding = retriever.embed_passages([Document.from_dict(original_doc)])
        stored_emb = document_store.faiss_index.reconstruct(int(doc.meta["vector_id"]))
        # compare original input vec with stored one (ignore extra dim added by hnsw)
        assert np.allclose(updated_embedding, stored_emb[:-1], rtol=0.01)

@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_retrieving(document_store):

    document_store.write_documents(DOCUMENTS)

    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=False)
    result = retriever.retrieve(query="How to test this?")
    assert len(result) == 3
    assert type(result[0]) == Document

@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_finding(document_store):

    document_store.write_documents(DOCUMENTS)

    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=False)
    finder = Finder(reader=None, retriever=retriever)

    prediction = finder.get_answers_via_similar_questions(question="How to test this?", top_k_retriever=1)

    assert len(prediction.get('answers', [])) == 1

