import pytest

from haystack.schema import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import GenerativeQAPipeline
from haystack.nodes import BM25Retriever, RAGenerator


@pytest.fixture
def docs():
    return [
        Document(content="The capital of Germany is the city state of Berlin."),
        Document(content="Berlin is the capital and largest city of Germany by both area and population."),
    ]


def test_rag_generator_pipeline(docs):
    document_store = InMemoryDocumentStore(use_bm25=True)
    retriever = BM25Retriever(document_store=document_store)
    rag_generator = RAGenerator(model_name_or_path="facebook/rag-token-nq", generator_type="token", max_length=20)
    document_store.write_documents(docs)

    query = "What is capital of the Germany?"
    pipeline = GenerativeQAPipeline(retriever=retriever, generator=rag_generator)
    output = pipeline.run(query=query, params={"Generator": {"top_k": 2}, "Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 2
    assert "berlin" in answers[0].answer.lower()
    for doc_idx, document in enumerate(output["documents"]):
        assert document.id == answers[0].document_ids[doc_idx]
        assert document.meta == answers[0].meta["doc_metas"][doc_idx]
