import pytest

from haystack import Document
from haystack.pipelines import SearchSummarizationPipeline
from haystack.nodes import DenseRetriever


@pytest.mark.parametrize(
    "retriever,document_store", [("embedding", "memory"), ("bm25", "elasticsearch")], indirect=True
)
def test_summarization_pipeline(document_store, retriever, summarizer):
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
    The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest
    structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction,
    the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a
    title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first
    structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower
    in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel
    Tower is the second tallest free-standing structure in France after the Millau Viaduct.
    """
        ),
    ]
    document_store.write_documents(docs)

    if isinstance(retriever, DenseRetriever):
        document_store.update_embeddings(retriever=retriever)

    query = "Where is Eiffel Tower?"
    pipeline = SearchSummarizationPipeline(retriever=retriever, summarizer=summarizer, return_in_answer_format=True)
    output = pipeline.run(query=query, params={"Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 1
    assert " The Eiffel Tower in Paris has officially opened its doors to the public." == answers[0]["answer"]
