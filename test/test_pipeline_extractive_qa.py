import pytest

from haystack.pipeline import (
    TranslationWrapperPipeline,
    ExtractiveQAPipeline
)
from haystack.pipelines.base import EvaluationResult

from haystack.schema import Answer, Document, Label, MultiLabel, Span


@pytest.mark.slow
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_answers(reader, retriever_with_docs, document_store_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    prediction = pipeline.run(
        query="Who lives in Berlin?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}},
    )
    assert prediction is not None
    assert type(prediction["answers"][0]) == Answer
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0].answer == "Carla"
    assert prediction["answers"][0].score <= 1
    assert prediction["answers"][0].score >= 0
    assert prediction["answers"][0].meta["meta_field"] == "test1"
    assert (
        prediction["answers"][0].context == "My name is Carla and I live in Berlin"
    )

    assert len(prediction["answers"]) == 3


@pytest.mark.slow
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_answers_without_normalized_scores(reader_without_normalized_scores, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader_without_normalized_scores, retriever=retriever_with_docs)
    prediction = pipeline.run(
        query="Who lives in Berlin?", params={"Reader": {"top_k": 3}}
    )
    assert prediction is not None
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0].answer == "Carla"
    assert prediction["answers"][0].score <= 11
    assert prediction["answers"][0].score >= 10
    assert prediction["answers"][0].meta["meta_field"] == "test1"
    assert (
            prediction["answers"][0].context == "My name is Carla and I live in Berlin"
    )

    assert len(prediction["answers"]) == 3


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_offsets(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    prediction = pipeline.run(query="Who lives in Berlin?", params={"Retriever": {"top_k": 5}})

    start = prediction["answers"][0].offsets_in_context[0].start
    end = prediction["answers"][0].offsets_in_context[0].end

    assert start == 11
    assert end == 16

    assert (
        prediction["answers"][0].context[start:end]
        == prediction["answers"][0].answer
    )


@pytest.mark.slow
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_answers_single_result(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    query = "testing finder"
    prediction = pipeline.run(query=query, params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}})
    assert prediction is not None
    assert len(prediction["answers"]) == 1


@pytest.mark.slow
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_answers_with_translator(
    reader, retriever_with_docs, en_to_de_translator, de_to_en_translator
):
    base_pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    pipeline = TranslationWrapperPipeline(
        input_translator=de_to_en_translator,
        output_translator=en_to_de_translator,
        pipeline=base_pipeline,
    )

    prediction = pipeline.run(query="Wer lebt in Berlin?", params={"Reader": {"top_k": 3}})
    assert prediction is not None
    assert prediction["query"] == "Wer lebt in Berlin?"
    assert "Carla" in prediction["answers"][0].answer
    assert prediction["answers"][0].score <= 1
    assert prediction["answers"][0].score >= 0
    assert prediction["answers"][0].meta["meta_field"] == "test1"
    assert (
        prediction["answers"][0].context == "My name is Carla and I live in Berlin"
    )


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_extractive_qa_eval(reader, retriever_with_docs, tmp_path):
    queries = ["Who lives in Berlin?"]
    labels = [
        MultiLabel(labels=[Label(query="Who lives in Berlin?", answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]), 
            document=Document(id='a0747b83aea0b60c4b114b15476dd32d', content_type="text", content='My name is Carla and I live in Berlin'), 
            is_correct_answer=True, is_correct_document=True, origin="gold-label")])
    ]

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result = pipeline.eval(
        queries=queries, 
        labels=labels,
        params={"Retriever": {"top_k": 5}}, 
    )

    metrics = eval_result.calculate_metrics()

    reader_result = eval_result["Reader"]
    retriever_result = eval_result["Retriever"]

    assert reader_result[reader_result['rank'] == 1]["answer"].iloc[0] in reader_result[reader_result['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_result[retriever_result['rank'] == 1]["id"].iloc[0] in retriever_result[retriever_result['rank'] == 1]["gold_document_ids"].iloc[0]
    assert metrics["MatchInTop1"] == 1.0

    eval_result.save(tmp_path)
    saved_eval_result = EvaluationResult.load(tmp_path)
    metrics = saved_eval_result.calculate_metrics()

    assert reader_result[reader_result['rank'] == 1]["answer"].iloc[0] in reader_result[reader_result['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_result[retriever_result['rank'] == 1]["id"].iloc[0] in retriever_result[retriever_result['rank'] == 1]["gold_document_ids"].iloc[0]
    assert metrics["MatchInTop1"] == 1.0


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_extractive_qa_eval_multiple_queries(reader, retriever_with_docs, tmp_path):
    queries = ["Who lives in Berlin?", "Who lives in Munich?"]
    labels = [
        MultiLabel(labels=[Label(query="Who lives in Berlin?", answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]), 
            document=Document(id='a0747b83aea0b60c4b114b15476dd32d', content_type="text", content='My name is Carla and I live in Berlin'), 
            is_correct_answer=True, is_correct_document=True, origin="gold-label")]),
        MultiLabel(labels=[Label(query="Who lives in Munich?", answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]), 
            document=Document(id='something_else', content_type="text", content='My name is Carla and I live in Munich'), 
            is_correct_answer=True, is_correct_document=True, origin="gold-label")])
    ]

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result = pipeline.eval(
        queries=queries, 
        labels=labels,
        params={"Retriever": {"top_k": 5}}, 
    )

    metrics = eval_result.calculate_metrics()

    reader_result = eval_result["Reader"]
    retriever_result = eval_result["Retriever"]

    reader_berlin = reader_result[reader_result['query'] == "Who lives in Berlin?"]
    reader_munich = reader_result[reader_result['query'] == "Who lives in Munich?"]

    retriever_berlin = retriever_result[retriever_result['query'] == "Who lives in Berlin?"]
    retriever_munich = retriever_result[retriever_result['query'] == "Who lives in Munich?"]

    assert reader_berlin[reader_berlin['rank'] == 1]["answer"].iloc[0] in reader_berlin[reader_berlin['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_berlin[retriever_berlin['rank'] == 1]["id"].iloc[0] in retriever_berlin[retriever_berlin['rank'] == 1]["gold_document_ids"].iloc[0]
    assert reader_munich[reader_munich['rank'] == 1]["answer"].iloc[0] not in reader_munich[reader_munich['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_munich[retriever_munich['rank'] == 1]["id"].iloc[0] not in retriever_munich[retriever_munich['rank'] == 1]["gold_document_ids"].iloc[0]
    assert metrics["MatchInTop1"] == 0.5

    eval_result.save(tmp_path)
    saved_eval_result = EvaluationResult.load(tmp_path)
    metrics = saved_eval_result.calculate_metrics()

    assert reader_berlin[reader_berlin['rank'] == 1]["answer"].iloc[0] in reader_berlin[reader_berlin['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_berlin[retriever_berlin['rank'] == 1]["id"].iloc[0] in retriever_berlin[retriever_berlin['rank'] == 1]["gold_document_ids"].iloc[0]
    assert reader_munich[reader_munich['rank'] == 1]["answer"].iloc[0] not in reader_munich[reader_munich['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_munich[retriever_munich['rank'] == 1]["id"].iloc[0] not in retriever_munich[retriever_munich['rank'] == 1]["gold_document_ids"].iloc[0]
    assert metrics["MatchInTop1"] == 0.5
