import time

import pytest

from haystack import Document

DOCUMENTS = [
    Document(
        text="""The capital of Germany is the city state of Berlin."""
    ),
    Document(
        text="""Berlin is the capital and largest city of Germany by both area and population.""",
    )
]


@pytest.mark.slow
@pytest.mark.generator
def test_rag_token_generator(rag_generator, faiss_document_store, dpr_retriever):
    faiss_document_store.return_embedding = True
    faiss_document_store.write_documents(DOCUMENTS)
    faiss_document_store.update_embeddings(retriever=dpr_retriever)
    time.sleep(1)

    question = "What is capital of the Germany?"

    retrieved_docs = dpr_retriever.retrieve(query=question, top_k=5)
    generated_docs = rag_generator.predict(question=question, documents=retrieved_docs, top_k=1)
    answers = generated_docs["answers"]
    assert len(answers) == 1
    assert "berlin" in answers[0]["answer"]
