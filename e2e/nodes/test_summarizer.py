from haystack.schema import Document
from haystack.nodes import TransformersSummarizer


DOCS = [
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

EXPECTED_SUMMARIES = [
    "Hundreds of thousands of customers in California have been left without power after a power cut.",
    "The Eiffel Tower in Paris is the world's tallest man-made structure.",
]


def test_summarization():
    summarizer = TransformersSummarizer(model_name_or_path="sshleifer/distilbart-xsum-12-1", use_gpu=False)

    summarized_docs = summarizer.predict(documents=DOCS)
    assert len(summarized_docs) == len(DOCS)
    for expected_summary, summary in zip(EXPECTED_SUMMARIES, summarized_docs):
        assert expected_summary.strip() == summary.meta["summary"].strip()


def test_summarization_batch():
    summarizer = TransformersSummarizer(model_name_or_path="sshleifer/distilbart-xsum-12-1", use_gpu=False)

    summarized_docs = summarizer.predict_batch(documents=[DOCS, DOCS])
    assert len(summarized_docs) == 2  # Number of document lists
    assert len(summarized_docs[0]) == len(DOCS)
    assert len(summarized_docs[1]) == len(DOCS)
    for expected_summary, summary_1, summary_2 in zip(EXPECTED_SUMMARIES, summarized_docs[0], summarized_docs[1]):
        assert expected_summary.strip() == summary_1.meta["summary"].strip()
        assert expected_summary.strip() == summary_2.meta["summary"].strip()
