import pytest

from haystack import Document
from haystack.pipeline import SearchSummarizationPipeline
from haystack.retriever.dense import DensePassageRetriever, EmbeddingRetriever

DOCS = [
    Document(
        text="""PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.""",
    ),
    Document(
        text="""The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."""
    )
]

EXPECTED_SUMMARIES = [
    "California's largest electricity provider has turned off power to hundreds of thousands of customers.",
    "The Eiffel Tower is a landmark in Paris, France."
]

SPLIT_DOCS = [
    Document(
        text="""The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930."""
    ),
    Document(
        text="""It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."""
    )
]

# Documents order is very important to produce summary.
# Different order of same documents produce different summary.
EXPECTED_ONE_SUMMARIES = [
    "The Eiffel Tower is a landmark in Paris, France.",
    "The Eiffel Tower, built in 1889 in Paris, France, is the world's tallest free-standing structure."
]


@pytest.mark.slow
@pytest.mark.summarizer
def test_summarization(summarizer):
    summarized_docs = summarizer.predict(documents=DOCS)
    assert len(summarized_docs) == len(DOCS)
    for expected_summary, summary in zip(EXPECTED_SUMMARIES, summarized_docs):
        assert expected_summary == summary.text


@pytest.mark.slow
@pytest.mark.summarizer
def test_summarization_one_summary(summarizer):
    summarized_docs = summarizer.predict(documents=SPLIT_DOCS, generate_single_summary=True)
    assert len(summarized_docs) == 1
    assert EXPECTED_ONE_SUMMARIES[0] == summarized_docs[0].text


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.summarizer
@pytest.mark.parametrize(
    "retriever,document_store",
    [("embedding", "memory"), ("embedding", "faiss"), ("elasticsearch", "elasticsearch")],
    indirect=True,
)
def test_summarization_pipeline(document_store, retriever, summarizer):
    document_store.write_documents(DOCS)

    if isinstance(retriever, EmbeddingRetriever) or isinstance(retriever, DensePassageRetriever):
        document_store.update_embeddings(retriever=retriever)

    query = "Where is Eiffel Tower?"
    pipeline = SearchSummarizationPipeline(retriever=retriever, summarizer=summarizer)
    output = pipeline.run(query=query, top_k_retriever=1, return_in_answer_format=True)
    answers = output["answers"]
    assert len(answers) == 1
    assert "The Eiffel Tower is a landmark in Paris, France." == answers[0]["answer"]


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.summarizer
@pytest.mark.parametrize(
    "retriever,document_store",
    [("embedding", "memory"), ("embedding", "faiss"), ("elasticsearch", "elasticsearch")],
    indirect=True,
)
def test_summarization_pipeline_one_summary(document_store, retriever, summarizer):
    document_store.write_documents(SPLIT_DOCS)

    if isinstance(retriever, EmbeddingRetriever) or isinstance(retriever, DensePassageRetriever):
        document_store.update_embeddings(retriever=retriever)

    query = "Where is Eiffel Tower?"
    pipeline = SearchSummarizationPipeline(retriever=retriever, summarizer=summarizer)
    output = pipeline.run(query=query, top_k_retriever=2, generate_single_summary=True, return_in_answer_format=True)
    answers = output["answers"]
    assert len(answers) == 1
    assert answers[0]["answer"] in EXPECTED_ONE_SUMMARIES
