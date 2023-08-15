import pytest

from haystack import Document
from haystack.nodes import EmbeddingRetriever, BM25Retriever, TransformersSummarizer
from haystack.pipelines import DocumentSearchPipeline, SearchSummarizationPipeline
from haystack.document_stores import InMemoryDocumentStore


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


def test_summarization_pipeline():
    docs = [
        Document(
            content="""
    PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions.
    The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected
    by the shutoffs which were expected to last through at least midday tomorrow.
    """
        ),
        Document(
            content="""
    The Eiffel Tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest
    structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction,
    the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a
    title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first
    structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower
    in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel
    Tower is the second tallest free-standing structure in France after the Millau Viaduct.
    """
        ),
    ]
    summarizer = TransformersSummarizer(model_name_or_path="sshleifer/distilbart-xsum-12-6", use_gpu=False)

    ds = InMemoryDocumentStore(use_bm25=True)
    retriever = BM25Retriever(document_store=ds)
    ds.write_documents(docs)

    query = "Eiffel Tower"
    pipeline = SearchSummarizationPipeline(retriever=retriever, summarizer=summarizer, return_in_answer_format=True)
    output = pipeline.run(query=query, params={"Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 1
    assert "The Eiffel Tower is one of the world's tallest structures" == answers[0]["answer"].strip()


def test_summarization_pipeline_one_summary():
    split_docs = [
        Document(
            content="""
    The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris.
    Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the
    Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler
    Building in New York City was finished in 1930.
    """
        ),
        Document(
            content="""
    It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the
    top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters,
    the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
    """
        ),
    ]
    ds = InMemoryDocumentStore(use_bm25=True)
    retriever = BM25Retriever(document_store=ds)
    ds.write_documents(split_docs)
    summarizer = TransformersSummarizer(model_name_or_path="sshleifer/distilbart-xsum-12-6", use_gpu=False)

    query = "Eiffel Tower"
    pipeline = SearchSummarizationPipeline(
        retriever=retriever, summarizer=summarizer, generate_single_summary=True, return_in_answer_format=True
    )
    output = pipeline.run(query=query, params={"Retriever": {"top_k": 2}})
    answers = output["answers"]
    assert len(answers) == 1
    assert answers[0]["answer"].strip() == "The Eiffel Tower was built in 1924 in Paris, France."
