import pytest

from haystack.pipelines import TranslationWrapperPipeline, ExtractiveQAPipeline

from haystack.schema import Answer


@pytest.mark.integration
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_answers(reader, retriever_with_docs, document_store_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    prediction = pipeline.run(query="Who lives in Berlin?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}})
    assert prediction is not None
    assert type(prediction["answers"][0]) == Answer
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0].answer == "Carla"
    assert prediction["answers"][0].score <= 1
    assert prediction["answers"][0].score >= 0
    assert prediction["answers"][0].meta["meta_field"] == "test1"
    assert prediction["answers"][0].context == "My name is Carla and I live in Berlin"

    assert len(prediction["answers"]) == 3


@pytest.mark.integration
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_answers_without_normalized_scores(reader_without_normalized_scores, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader_without_normalized_scores, retriever=retriever_with_docs)
    prediction = pipeline.run(query="Who lives in Berlin?", params={"Reader": {"top_k": 3}})
    assert prediction is not None
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0].answer == "Carla"
    assert prediction["answers"][0].score <= 9
    assert prediction["answers"][0].score >= 8
    assert prediction["answers"][0].meta["meta_field"] == "test1"
    assert prediction["answers"][0].context == "My name is Carla and I live in Berlin"

    assert len(prediction["answers"]) == 3


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_offsets(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    prediction = pipeline.run(query="Who lives in Berlin?", params={"Retriever": {"top_k": 5}})

    start = prediction["answers"][0].offsets_in_context[0].start
    end = prediction["answers"][0].offsets_in_context[0].end

    assert start == 11
    assert end == 16

    assert prediction["answers"][0].context[start:end] == prediction["answers"][0].answer


@pytest.mark.integration
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_answers_single_result(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    query = "testing finder"
    prediction = pipeline.run(query=query, params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}})
    assert prediction is not None
    assert len(prediction["answers"]) == 1
