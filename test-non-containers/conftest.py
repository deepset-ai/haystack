import os

import pytest

from haystack.document_store.memory import InMemoryDocumentStore
from haystack.generator.transformers import RAGenerator, RAGeneratorType
from haystack.retriever.dense import DensePassageRetriever, EmbeddingRetriever
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.sparse import TfidfRetriever


@pytest.fixture()
def rag_generator(dpr_retriever):
    return RAGenerator(
        model_name_or_path="facebook/rag-token-nq",
        retriever=dpr_retriever,
        generator_type=RAGeneratorType.TOKEN
    )


@pytest.fixture()
def faiss_document_store():
    if os.path.exists("haystack_test_faiss.db"):
        os.remove("haystack_test_faiss.db")
    document_store = FAISSDocumentStore(sql_url="sqlite:///haystack_test_faiss.db")
    yield document_store
    document_store.faiss_index.reset()


@pytest.fixture()
def inmemory_document_store():
    return InMemoryDocumentStore()


@pytest.fixture()
def dpr_retriever(faiss_document_store):
    return DensePassageRetriever(
        document_store=faiss_document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
        embed_title=True,
        remove_sep_tok_from_untitled_passages=True
    )


@pytest.fixture()
def embedding_retriever(faiss_document_store):
    return EmbeddingRetriever(
        document_store=faiss_document_store,
        embedding_model="deepset/sentence_bert",
        use_gpu=False
    )


@pytest.fixture()
def tfidf_retriever(inmemory_document_store):
    return TfidfRetriever(document_store=inmemory_document_store)
