import numpy as np
import pytest

from haystack.retriever.dense import EmbeddingRetriever
from haystack import Finder


DOCUMENTS = [
    {"name": "name_1", "text": "text_1", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_2", "text": "text_2", "embedding": np.random.rand(768).astype(np.float32)},
    {"name": "name_3", "text": "text_3", "embedding": np.random.rand(768).astype(np.float32)},
]


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_indexing(document_store):

    document_store.write_documents(DOCUMENTS)
    documents_indexed = document_store.get_all_documents()

    # test if correct vector_ids are assigned
    for i, doc in enumerate(documents_indexed):
        assert doc.meta["vector_id"] == str(i)

    # test insertion of documents in an existing index fails
    with pytest.raises(Exception):
        document_store.write_documents(DOCUMENTS)

    # test saving the index
    document_store.save("haystack_test_faiss")

    # test loading the index
    document_store.load(sql_url="sqlite:///haystack_test.db", faiss_file_path="haystack_test_faiss")


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_retrieving(document_store):

    document_store.write_documents(DOCUMENTS)

    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=False)
    embedding = retriever.retrieve(query="How to test this?")


@pytest.mark.parametrize("document_store", ["faiss"], indirect=True)
def test_faiss_finding(document_store):

    document_store.write_documents(DOCUMENTS)

    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=False)
    finder = Finder(reader=None, retriever=retriever)

    prediction = finder.get_answers_via_similar_questions(question="How to test this?", top_k_retriever=1)

    assert len(prediction.get('answers', [])) == 1
