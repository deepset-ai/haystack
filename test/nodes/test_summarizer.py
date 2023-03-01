import pytest

import haystack
from haystack.utils.torch_utils import ListDataset
from haystack.schema import Document
from haystack.nodes import TransformersSummarizer


DOCS = [Document(content=doc) for doc in ["First test doc", "Second test doc"]]
EXPECTED_SUMMARIES = ["First summary", "Second summary"]
SUMMARIZED_DOCS = [
    Document(content=doc.content, meta={"summary": summary}) for doc, summary in zip(DOCS, EXPECTED_SUMMARIES)
]


class MockHFPipeline:
    def __init__(self, *a, **k):
        pass

    def __call__(self, docs, *a, **k):
        summaries = [{"summary_text": summary} for summary in EXPECTED_SUMMARIES]
        if isinstance(docs, ListDataset):
            return [summaries for _ in docs]
        return summaries

    def tokenizer(self, *a, **k):
        return {"input_ids": []}


@pytest.fixture
def mock_models(monkeypatch):
    monkeypatch.setattr(haystack.nodes.summarizer.transformers, "pipeline", MockHFPipeline)


@pytest.fixture
def summarizer(mock_models) -> TransformersSummarizer:
    return TransformersSummarizer(model_name_or_path="irrelevant/anyway", use_gpu=False)


@pytest.mark.unit
def test_summarization_no_docs(summarizer):
    with pytest.raises(ValueError, match="at least one document"):
        summarizer.predict(documents=[])
    with pytest.raises(ValueError, match="at least one document"):
        summarizer.predict_batch(documents=[])


@pytest.mark.unit
def test_summarization_no_docs(summarizer):
    summarizer.min_length = 10
    summarizer.max_length = 1
    with pytest.raises(ValueError, match="min_length cannot be greater than max_length"):
        summarizer.predict(documents=DOCS)


@pytest.mark.unit
def test_summarization_one_doc(summarizer):
    summarized_docs = summarizer.predict(documents=[DOCS[0]])
    assert len(summarized_docs) == 1
    assert EXPECTED_SUMMARIES[0] == summarized_docs[0].meta["summary"]


@pytest.mark.unit
def test_summarization_more_docs(summarizer):
    summarized_docs = summarizer.predict(documents=DOCS)
    assert len(summarized_docs) == len(DOCS)
    for expected_summary, summary in zip(EXPECTED_SUMMARIES, summarized_docs):
        assert expected_summary == summary.meta["summary"]


@pytest.mark.unit
def test_summarization_batch_single_doc_list(summarizer):
    summarized_docs = summarizer.predict_batch(documents=DOCS)
    assert len(summarized_docs) == len(DOCS)
    for expected_summary, summary in zip(EXPECTED_SUMMARIES, summarized_docs):
        assert expected_summary == summary.meta["summary"]


@pytest.mark.unit
def test_summarization_batch_multiple_doc_lists(summarizer):
    summarized_docs = summarizer.predict_batch(documents=[DOCS, DOCS])
    assert len(summarized_docs) == 2  # Number of document lists
    assert len(summarized_docs[0]) == len(DOCS)
    for expected_summary, summary in zip(EXPECTED_SUMMARIES, summarized_docs[0]):
        assert expected_summary == summary.meta["summary"]
