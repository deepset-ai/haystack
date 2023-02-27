import pytest

from haystack.schema import Document
from haystack.pipelines import SearchSummarizationPipeline
from haystack.nodes import EmbeddingRetriever, TransformersSummarizer, BM25Retriever
from haystack.nodes.other.document_merger import DocumentMerger
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore

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


@pytest.fixture
def summarizer():
    return TransformersSummarizer(model_name_or_path="sshleifer/distilbart-xsum-12-6", use_gpu=False)


@pytest.fixture
def retriever(request):
    if request.param == "bm25":
        retriever = BM25Retriever(document_store=ElasticsearchDocumentStore())
        yield retriever
        retriever.document_store.delete_documents()

    elif request.param == "embedding":
        retriever = EmbeddingRetriever(
            document_store=InMemoryDocumentStore(), embedding_model="deepset/sentence_bert", use_gpu=False
        )
        yield retriever
        retriever.document_store.delete_documents()


def test_summarization(summarizer):
    summarized_docs = summarizer.predict(documents=DOCS)
    assert len(summarized_docs) == len(DOCS)
    for expected_summary, summary in zip(EXPECTED_SUMMARIES, summarized_docs):
        assert expected_summary == summary.meta["summary"]


def test_summarization_batch_single_doc_list(summarizer):
    summarized_docs = summarizer.predict_batch(documents=DOCS)
    assert len(summarized_docs) == len(DOCS)
    for expected_summary, summary in zip(EXPECTED_SUMMARIES, summarized_docs):
        assert expected_summary == summary.meta["summary"]


def test_summarization_batch_multiple_doc_lists(summarizer):
    summarized_docs = summarizer.predict_batch(documents=[DOCS, DOCS])
    assert len(summarized_docs) == 2  # Number of document lists
    assert len(summarized_docs[0]) == len(DOCS)
    for expected_summary, summary in zip(EXPECTED_SUMMARIES, summarized_docs[0]):
        assert expected_summary == summary.meta["summary"]


@pytest.mark.parametrize("retriever", ["bm25", "embedding"], indirect=True)
def test_summarization_pipeline(retriever, summarizer):
    retriever.document_store.write_documents(DOCS)

    if isinstance(retriever, EmbeddingRetriever):
        retriever.document_store.update_embeddings(retriever=retriever)

    query = "Where is Eiffel Tower?"
    pipeline = SearchSummarizationPipeline(retriever=retriever, summarizer=summarizer, return_in_answer_format=True)
    output = pipeline.run(query=query, params={"Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 1
    assert " The Eiffel Tower in Paris has officially opened its doors to the public." == answers[0]["answer"]


#
# Document Merger + Summarizer tests
#


def test_summarization_one_summary(summarizer):
    dm = DocumentMerger()
    merged_document = dm.merge(documents=SPLIT_DOCS)
    summarized_docs = summarizer.predict(documents=merged_document)
    assert len(summarized_docs) == 1
    assert EXPECTED_ONE_SUMMARIES[0] == summarized_docs[0].meta["summary"]


@pytest.mark.parametrize("retriever", ["bm25", "embedding"], indirect=True)
def test_summarization_pipeline_one_summary(retriever, summarizer):
    retriever.document_store.write_documents(SPLIT_DOCS)

    if isinstance(retriever, EmbeddingRetriever):
        retriever.document_store.update_embeddings(retriever=retriever)

    query = "Where is Eiffel Tower?"
    pipeline = SearchSummarizationPipeline(
        retriever=retriever, summarizer=summarizer, generate_single_summary=True, return_in_answer_format=True
    )
    output = pipeline.run(query=query, params={"Retriever": {"top_k": 2}})
    answers = output["answers"]
    assert len(answers) == 1
    assert answers[0]["answer"] in EXPECTED_ONE_SUMMARIES
