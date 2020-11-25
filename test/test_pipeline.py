import pytest

from haystack.pipeline import ExtractiveQAPipeline, Pipeline


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_with_docs", ["elasticsearch"], indirect=True)
def test_graph_creation(reader, retriever_with_docs, document_store_with_docs):
    pipeline = Pipeline()
    pipeline.add_node(name="ES", component=retriever_with_docs, inputs=["Query"])

    with pytest.raises(AssertionError):
        pipeline.add_node(name="Reader", component=retriever_with_docs, inputs=["ES.output_2"])

    with pytest.raises(AssertionError):
        pipeline.add_node(name="Reader", component=retriever_with_docs, inputs=["ES.wrong_edge_label"])

    with pytest.raises(Exception):
        pipeline.add_node(name="Reader", component=retriever_with_docs, inputs=["InvalidNode"])


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_answers(reader, retriever_with_docs, document_store_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    prediction = pipeline.run(question="Who lives in Berlin?", top_k_retriever=10, top_k_reader=3)
    assert prediction is not None
    assert prediction["question"] == "Who lives in Berlin?"
    assert prediction["answers"][0]["answer"] == "Carla"
    assert prediction["answers"][0]["probability"] <= 1
    assert prediction["answers"][0]["probability"] >= 0
    assert prediction["answers"][0]["meta"]["meta_field"] == "test1"
    assert prediction["answers"][0]["context"] == "My name is Carla and I live in Berlin"

    assert len(prediction["answers"]) == 3


@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_offsets(reader, retriever_with_docs, document_store_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    prediction = pipeline.run(question="Who lives in Berlin?", top_k_retriever=10, top_k_reader=5)

    assert prediction["answers"][0]["offset_start"] == 11
    assert prediction["answers"][0]["offset_end"] == 16
    start = prediction["answers"][0]["offset_start"]
    end = prediction["answers"][0]["offset_end"]
    assert prediction["answers"][0]["context"][start:end] == prediction["answers"][0]["answer"]


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_extractive_qa_answers_single_result(reader, retriever_with_docs, document_store_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    query = "testing finder"
    prediction = pipeline.run(question=query, top_k_retriever=1, top_k_reader=1)
    assert prediction is not None
    assert len(prediction["answers"]) == 1

