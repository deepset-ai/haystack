import pytest

from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import DocumentSearchPipeline

from ..conftest import document_store


@pytest.mark.parametrize("name", ["memory", "faiss", "weaviate", "elasticsearch"])
def test_similarity_score_sentence_transformers(name, docs, tmp_path):
    with document_store(name, docs, tmp_path, embedding_dim=384) as ds:
        retriever = EmbeddingRetriever(
            document_store=ds, embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2"
        )
        ds.update_embeddings(retriever)
        pipeline = DocumentSearchPipeline(retriever)
        prediction = pipeline.run("Paul lives in New York")
        scores = [document.score for document in prediction["documents"]]
        assert [document.content for document in prediction["documents"]] == [
            "My name is Paul and I live in New York",
            "My name is Matteo and I live in Rome",
            "My name is Christelle and I live in Paris",
            "My name is Carla and I live in Berlin",
            "My name is Camila and I live in Madrid",
        ]
        assert scores == pytest.approx(
            [0.9149981737136841, 0.6895168423652649, 0.641706794500351, 0.6206043660640717, 0.5837393924593925],
            abs=1e-3,
        )


@pytest.mark.parametrize("name", ["memory", "faiss", "weaviate", "elasticsearch"])
def test_similarity_score(name, docs, tmp_path):
    with document_store(name, docs, tmp_path, embedding_dim=384) as ds:
        retriever = EmbeddingRetriever(
            document_store=ds, embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2", model_format="farm"
        )
        ds.update_embeddings(retriever)
        pipeline = DocumentSearchPipeline(retriever)
        prediction = pipeline.run("Paul lives in New York")
        scores = [document.score for document in prediction["documents"]]
        assert scores == pytest.approx(
            [0.9102507941407827, 0.6937791467877008, 0.6491682889305038, 0.6321622491318529, 0.5909129441370939],
            abs=1e-3,
        )


@pytest.mark.parametrize("name", ["memory", "faiss", "weaviate", "elasticsearch"])
def test_similarity_score_without_scaling(name, docs, tmp_path):
    with document_store(name, docs, tmp_path, embedding_dim=384) as ds:
        retriever = EmbeddingRetriever(
            document_store=ds,
            embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
            scale_score=False,
            model_format="farm",
        )
        ds.update_embeddings(retriever)
        pipeline = DocumentSearchPipeline(retriever)
        prediction = pipeline.run("Paul lives in New York")
        scores = [document.score for document in prediction["documents"]]
        assert scores == pytest.approx(
            [0.8205015882815654, 0.3875582935754016, 0.29833657786100765, 0.26432449826370585, 0.18182588827418789],
            abs=1e-3,
        )


@pytest.mark.parametrize("name", ["memory", "faiss", "weaviate", "elasticsearch"])
def test_similarity_score_dot_product(name, docs, tmp_path):
    with document_store(name, docs, tmp_path, similarity="dot_product", embedding_dim=384) as ds:
        retriever = EmbeddingRetriever(
            document_store=ds, embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2", model_format="farm"
        )
        ds.update_embeddings(retriever)
        pipeline = DocumentSearchPipeline(retriever)
        prediction = pipeline.run("Paul lives in New York")
        scores = [document.score for document in prediction["documents"]]
        assert scores == pytest.approx(
            [0.5526494403409358, 0.5247784342375555, 0.5189836829440964, 0.5179697273254912, 0.5112024928228626],
            abs=1e-3,
        )


@pytest.mark.parametrize("name", ["memory", "faiss", "weaviate", "elasticsearch"])
def test_similarity_score_dot_product_without_scaling(name, docs, tmp_path):
    with document_store(name, docs, tmp_path, embedding_dim=384, similarity="dot_product") as ds:
        retriever = EmbeddingRetriever(
            document_store=ds,
            embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
            scale_score=False,
            model_format="farm",
        )
        ds.update_embeddings(retriever)
        pipeline = DocumentSearchPipeline(retriever)
        prediction = pipeline.run("Paul lives in New York")
        scores = [document.score for document in prediction["documents"]]
        assert scores == pytest.approx(
            [21.13810000000001, 9.919499999999971, 7.597099999999955, 7.191000000000031, 4.481750000000034], abs=1e-3
        )
