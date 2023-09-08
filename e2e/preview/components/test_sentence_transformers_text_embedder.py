import numpy as np

from haystack.preview.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_sentence_transformers_text_embedder():
    texts = ["Giraffe is a long-necked animal", "I love animals", "Haystack is a great framework"]

    # Use a very small model for testing
    txt_embedder = SentenceTransformersTextEmbedder(model_name_or_path="paraphrase-albert-small-v2")
    txt_embedder.warm_up()

    output = txt_embedder.run(texts=texts)
    embeddings = output["embeddings"]

    assert len(embeddings) == 3
    assert all(isinstance(embedding, list) for embedding in embeddings)
    assert all(len(embedding) == 768 for embedding in embeddings)
    assert all(isinstance(element, float) for embedding in embeddings for element in embedding)

    assert cosine_sim(embeddings[0], embeddings[1]) > cosine_sim(embeddings[0], embeddings[2])
