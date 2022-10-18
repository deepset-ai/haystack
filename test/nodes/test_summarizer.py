from copy import deepcopy

import pytest

from haystack.schema import Document
from haystack.pipelines import SearchSummarizationPipeline
from haystack.nodes import DensePassageRetriever, EmbeddingRetriever, TransformersSummarizer

DOCS = [
    Document(
        content="""PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
    ),
    Document(
        content="""The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."""
    ),
]

EXPECTED_SUMMARIES = [
    "California's largest electricity provider, PG&E, has shut down power supplies to thousands of customers.",
    " The Eiffel Tower in Paris has officially opened its doors to the public.",
]

SPLIT_DOCS = [
    Document(
        content="""The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930."""
    ),
    Document(
        content="""It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."""
    ),
]

# Documents order is very important to produce summary.
# Different order of same documents produce different summary.
EXPECTED_ONE_SUMMARIES = [
    " The Eiffel Tower in Paris has officially opened its doors to the public.",
    " The Eiffel Tower in Paris has become the tallest man-made structure in the world.",
]


@pytest.mark.integration
@pytest.mark.summarizer
def test_summarization(summarizer):
    summarized_docs = summarizer.predict(documents=DOCS)
    assert len(summarized_docs) == len(DOCS)
    for expected_summary, summary in zip(EXPECTED_SUMMARIES, summarized_docs):
        assert expected_summary == summary.content


@pytest.mark.integration
@pytest.mark.summarizer
def test_summarization_one_summary(summarizer):
    summarized_docs = summarizer.predict(documents=SPLIT_DOCS, generate_single_summary=True)
    assert len(summarized_docs) == 1
    assert EXPECTED_ONE_SUMMARIES[0] == summarized_docs[0].content


@pytest.mark.integration
@pytest.mark.summarizer
def test_summarization_batch_single_doc_list(summarizer):
    summarized_docs = summarizer.predict_batch(documents=DOCS)
    assert len(summarized_docs) == len(DOCS)
    for expected_summary, summary in zip(EXPECTED_SUMMARIES, summarized_docs):
        assert expected_summary == summary.content


@pytest.mark.integration
@pytest.mark.summarizer
def test_summarization_batch_multiple_doc_lists(summarizer):
    summarized_docs = summarizer.predict_batch(documents=[DOCS, DOCS])
    assert len(summarized_docs) == 2  # Number of document lists
    assert len(summarized_docs[0]) == len(DOCS)
    for expected_summary, summary in zip(EXPECTED_SUMMARIES, summarized_docs[0]):
        assert expected_summary == summary.content


@pytest.mark.integration
@pytest.mark.summarizer
@pytest.mark.parametrize(
    "retriever,document_store", [("embedding", "memory"), ("elasticsearch", "elasticsearch")], indirect=True
)
def test_summarization_pipeline(document_store, retriever, summarizer):
    document_store.write_documents(DOCS)

    if isinstance(retriever, EmbeddingRetriever) or isinstance(retriever, DensePassageRetriever):
        document_store.update_embeddings(retriever=retriever)

    query = "Where is Eiffel Tower?"
    pipeline = SearchSummarizationPipeline(retriever=retriever, summarizer=summarizer, return_in_answer_format=True)
    output = pipeline.run(query=query, params={"Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 1
    assert " The Eiffel Tower in Paris has officially opened its doors to the public." == answers[0]["answer"]


@pytest.mark.integration
@pytest.mark.summarizer
@pytest.mark.parametrize(
    "retriever,document_store", [("embedding", "memory"), ("elasticsearch", "elasticsearch")], indirect=True
)
def test_summarization_pipeline_one_summary(document_store, retriever, summarizer):
    document_store.write_documents(SPLIT_DOCS)

    if isinstance(retriever, EmbeddingRetriever) or isinstance(retriever, DensePassageRetriever):
        document_store.update_embeddings(retriever=retriever)

    query = "Where is Eiffel Tower?"
    pipeline = SearchSummarizationPipeline(retriever=retriever, summarizer=summarizer, return_in_answer_format=True)
    output = pipeline.run(
        query=query, params={"Retriever": {"top_k": 2}, "Summarizer": {"generate_single_summary": True}}
    )
    answers = output["answers"]
    assert len(answers) == 1
    assert answers[0]["answer"] in EXPECTED_ONE_SUMMARIES


@pytest.mark.summarizer
def test_metadata_summarizer(summarizer):
    docs = [
        Document(
            content="""PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.""",
            meta={
                "sub_content": "Pegasus Example",
                "topic": "California's Electricity",
                "context": "Dummy - PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires.",
            },
        ),
        Document(
            content="""The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.""",
            meta={"sub_content": "Paris best tour best tour", "topic": "Eiffel tower"},
        ),
    ]
    # Original input is overwritten after the "predict". So adding the same input as check_output to assess the output
    check_output = deepcopy(docs)

    summary = summarizer.predict(documents=docs)

    assert len(summary[0].meta) == len(check_output[0].meta)
    assert len(summary[1].meta) - 1 == len(check_output[1].meta)
    assert (
        summary[0].meta["context"]
        == """PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
    )

    summary = summarizer.predict(documents=docs, generate_single_summary=True)

    assert len(summary) == 1
    summary[0].meta.pop("context")
    assert not summary[0].meta  # Remaining metadata is not returned in case of a single summary
