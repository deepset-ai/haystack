import math

import pytest

from haystack import Document
from haystack.reader.base import BaseReader
from haystack.reader.farm import FARMReader


def test_reader_basic(reader):
    assert reader is not None
    assert isinstance(reader, BaseReader)


def test_output(prediction):
    assert prediction is not None
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0]["answer"] == "Carla"
    assert prediction["answers"][0]["offset_start"] == 11
    assert prediction["answers"][0]["offset_end"] == 16
    assert prediction["answers"][0]["probability"] <= 1
    assert prediction["answers"][0]["probability"] >= 0
    assert prediction["answers"][0]["context"] == "My name is Carla and I live in Berlin"
    assert len(prediction["answers"]) == 5


@pytest.mark.slow
def test_no_answer_output(no_answer_prediction):
    assert no_answer_prediction is not None
    assert no_answer_prediction["query"] == "What is the meaning of life?"
    assert math.isclose(no_answer_prediction["no_ans_gap"], -13.048564434051514, rel_tol=0.0001)
    assert no_answer_prediction["answers"][0]["answer"] is None
    assert no_answer_prediction["answers"][0]["offset_start"] == 0
    assert no_answer_prediction["answers"][0]["offset_end"] == 0
    assert no_answer_prediction["answers"][0]["probability"] <= 1
    assert no_answer_prediction["answers"][0]["probability"] >= 0
    assert no_answer_prediction["answers"][0]["context"] == None
    assert no_answer_prediction["answers"][0]["document_id"] == None
    answers = [x["answer"] for x in no_answer_prediction["answers"]]
    assert answers.count(None) == 1
    assert len(no_answer_prediction["answers"]) == 5


# TODO Directly compare farm and transformers reader outputs
# TODO checks to see that model is responsive to input arguments e.g. context_window_size - topk


@pytest.mark.slow
def test_prediction_attributes(prediction):
    # TODO FARM's prediction also has no_ans_gap
    attributes_gold = ["query", "answers"]
    for ag in attributes_gold:
        assert ag in prediction


def test_answer_attributes(prediction):
    # TODO Transformers answer also has meta key
    # TODO FARM answer has offset_start_in_doc, offset_end_in_doc
    answer = prediction["answers"][0]
    attributes_gold = ['answer', 'score', 'probability', 'context', 'offset_start', 'offset_end', 'document_id']
    for ag in attributes_gold:
        assert ag in answer


@pytest.mark.slow
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("window_size", [10, 15, 20])
def test_context_window_size(reader, test_docs_xs, window_size):
    docs = [Document.from_dict(d) if isinstance(d, dict) else d for d in test_docs_xs]

    assert isinstance(reader, FARMReader)

    old_window_size = reader.inferencer.model.prediction_heads[0].context_window_size
    reader.inferencer.model.prediction_heads[0].context_window_size = window_size

    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=5)
    for answer in prediction["answers"]:
        # If the extracted answer is larger than the context window, the context window is expanded.
        # If the extracted answer is odd in length, the resulting context window is one less than context_window_size
        # due to rounding (See FARM's QACandidate)
        # TODO Currently the behaviour of context_window_size in FARMReader and TransformerReader is different
        if len(answer["answer"]) <= window_size:
            assert len(answer["context"]) in [window_size, window_size - 1]
        else:
            assert len(answer["answer"]) == len(answer["context"])

    reader.inferencer.model.prediction_heads[0].context_window_size = old_window_size

    # TODO Need to test transformers reader
    # TODO Currently the behaviour of context_window_size in FARMReader and TransformerReader is different


@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("top_k", [2, 5, 10])
def test_top_k(reader, test_docs_xs, top_k):
    docs = [Document.from_dict(d) if isinstance(d, dict) else d for d in test_docs_xs]

    assert isinstance(reader, FARMReader)

    old_top_k_per_candidate = reader.top_k_per_candidate
    reader.top_k_per_candidate = 4
    reader.inferencer.model.prediction_heads[0].n_best = reader.top_k_per_candidate + 1
    try:
        old_top_k_per_sample = reader.inferencer.model.prediction_heads[0].n_best_per_sample
        reader.inferencer.model.prediction_heads[0].n_best_per_sample = 4
    except:
        print("WARNING: Could not set `top_k_per_sample` in FARM. Please update FARM version.")

    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=top_k)
    assert len(prediction["answers"]) == top_k

    reader.top_k_per_candidate = old_top_k_per_candidate
    reader.inferencer.model.prediction_heads[0].n_best = reader.top_k_per_candidate + 1
    try:
        reader.inferencer.model.prediction_heads[0].n_best_per_sample = old_top_k_per_sample
    except:
        print("WARNING: Could not set `top_k_per_sample` in FARM. Please update FARM version.")


def test_farm_reader_update_params(test_docs_xs):
    reader = FARMReader(
        model_name_or_path="deepset/roberta-base-squad2",
        use_gpu=False,
        no_ans_boost=0,
        num_processes=0
    )

    docs = [Document.from_dict(d) if isinstance(d, dict) else d for d in test_docs_xs]

    # original reader
    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)
    assert len(prediction["answers"]) == 3
    assert prediction["answers"][0]["answer"] == "Carla"

    # update no_ans_boost
    reader.update_parameters(
        context_window_size=100, no_ans_boost=100, return_no_answer=True, max_seq_len=384, doc_stride=128
    )
    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)
    assert len(prediction["answers"]) == 3
    assert prediction["answers"][0]["answer"] is None

    # update no_ans_boost
    reader.update_parameters(
        context_window_size=100, no_ans_boost=0, return_no_answer=False, max_seq_len=384, doc_stride=128
    )
    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)
    assert len(prediction["answers"]) == 3
    assert None not in [ans["answer"] for ans in prediction["answers"]]

    # update context_window_size
    reader.update_parameters(context_window_size=6, no_ans_boost=-10, max_seq_len=384, doc_stride=128)
    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)
    assert len(prediction["answers"]) == 3
    assert len(prediction["answers"][0]["context"]) == 6

    # update doc_stride with invalid value
    with pytest.raises(Exception):
        reader.update_parameters(context_window_size=100, no_ans_boost=-10, max_seq_len=384, doc_stride=999)
        reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)

    # update max_seq_len with invalid value
    with pytest.raises(Exception):
        reader.update_parameters(context_window_size=6, no_ans_boost=-10, max_seq_len=99, doc_stride=128)
        reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)
