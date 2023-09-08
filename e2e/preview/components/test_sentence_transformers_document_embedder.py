import numpy as np

from haystack.preview.dataclasses import Document
from haystack.preview.components.embedders.sentence_transformers_document_embedder import (
    SentenceTransformersDocumentEmbedder,
)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_sentence_transformers_document_embedder():
    docs = [
        Document(content="Giraffe is a long-necked animal", metadata={"topic": "animals"}),
        Document(content="I love animals", metadata={"topic": "animals"}),
        Document(content="Haystack is a great framework", metadata={"topic": "LLMs"}),
    ]

    # Use a very small model for testing
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model_name_or_path="paraphrase-albert-small-v2", metadata_fields_to_embed=["topic"]
    )
    doc_embedder.warm_up()

    output = doc_embedder.run(documents=docs)
    out_docs = output["documents"]

    assert len(out_docs) == 3
    assert all(isinstance(doc.embedding, list) for doc in out_docs)
    assert all(len(doc.embedding) == 768 for doc in out_docs)
    assert all(isinstance(element, float) for doc in out_docs for element in doc.embedding)

    assert cosine_sim(out_docs[0].embedding, out_docs[1].embedding) > cosine_sim(
        out_docs[0].embedding, out_docs[2].embedding
    )
