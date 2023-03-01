from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import SearchSummarizationPipeline
from haystack.nodes import BM25Retriever, TransformersSummarizer


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
