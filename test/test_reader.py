import math

from haystack import Document
from haystack.reader.base import BaseReader
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader



def test_reader_basic(reader):
    assert reader is not None
    assert isinstance(reader, BaseReader)


def test_output(prediction):
    assert prediction is not None
    assert prediction["question"] == "Who lives in Berlin?"
    assert prediction["answers"][0]["answer"] == "Carla"
    assert prediction["answers"][0]["offset_start"] == 11
    assert prediction["answers"][0]["offset_end"] == 16
    assert prediction["answers"][0]["probability"] <= 1
    assert prediction["answers"][0]["probability"] >= 0
    assert prediction["answers"][0]["context"] == "My name is Carla and I live in Berlin"
    assert len(prediction["answers"]) == 5


def test_no_answer_output(no_answer_prediction):
    assert no_answer_prediction is not None
    assert no_answer_prediction["question"] == "What is the meaning of life?"
    assert math.isclose(no_answer_prediction["no_ans_gap"], -14.4729533, rel_tol=0.0001)
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

def test_prediction_attributes(prediction):
    # TODO FARM's prediction also has no_ans_gap
    attributes_gold = ["question", "answers"]
    for ag in attributes_gold:
        assert ag in prediction


def test_answer_attributes(prediction):
    # TODO Transformers answer also has meta key
    # TODO FARM answer has offset_start_in_doc, offset_end_in_doc
    answer = prediction["answers"][0]
    attributes_gold = ['answer', 'score', 'probability', 'context', 'offset_start', 'offset_end', 'document_id']
    for ag in attributes_gold:
        assert ag in answer


def test_context_window_size(test_docs_xs):
    # TODO parametrize window_size and farm/transformers reader using pytest
    docs = [Document.from_dict(d) if isinstance(d, dict) else d for d in test_docs_xs]
    for window_size in [10, 15, 20]:
        farm_reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", num_processes=0,
                              use_gpu=False, top_k_per_sample=5, no_ans_boost=None, context_window_size=window_size)
        prediction = farm_reader.predict(question="Who lives in Berlin?", documents=docs, top_k=5)
        for answer in prediction["answers"]:
            # If the extracted answer is larger than the context window, the context window is expanded.
            # If the extracted answer is odd in length, the resulting context window is one less than context_window_size
            # due to rounding (See FARM's QACandidate)
            # TODO Currently the behaviour of context_window_size in FARMReader and TransformerReader is different
            if len(answer["answer"]) <= window_size:
                assert len(answer["context"]) in [window_size, window_size-1]
            else:
                assert len(answer["answer"]) == len(answer["context"])

        # TODO Need to test transformers reader
        # TODO Currently the behaviour of context_window_size in FARMReader and TransformerReader is different


def test_top_k(test_docs_xs):
    # TODO parametrize top_k and farm/transformers reader using pytest
    # TODO transformers reader was crashing when tested on this

    docs = [Document.from_dict(d) if isinstance(d, dict) else d for d in test_docs_xs]
    farm_reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", num_processes=0,
                             use_gpu=False, top_k_per_sample=4, no_ans_boost=None, top_k_per_candidate=4)
    for top_k in [2, 5, 10]:
        prediction = farm_reader.predict(question="Who lives in Berlin?", documents=docs, top_k=top_k)
        assert len(prediction["answers"]) == top_k



