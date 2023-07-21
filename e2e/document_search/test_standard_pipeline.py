import pytest

from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import DocumentSearchPipeline

from ..conftest import document_store


@pytest.mark.parametrize("document_store_name", ["memory", "faiss", "weaviate", "elasticsearch"])
def test_document_search_standard_pipeline(document_store_name, docs, tmp_path):
    """
    Testing the DocumentSearchPipeline with most common parameters according to our template:
    https://github.com/deepset-ai/templates/blob/main/pipelines/DenseDocSearch.yaml
    The common multi-qa-mpnet-base-dot-v1 model is replaced with the very similar paraphrase-MiniLM-L3-v2,
    which reduces runtime and model size by ~6x
    """
    with document_store(document_store_name, docs, tmp_path, embedding_dim=384) as ds:
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
