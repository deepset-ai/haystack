import math
import os
from pathlib import Path

import pytest
from haystack.modeling.data_handler.inputs import QAInput, Question

from haystack.schema import Document, Answer
from haystack.nodes.reader.base import BaseReader
from haystack.nodes.reader.farm import FARMReader


def test_reader_basic(reader):
    assert reader is not None
    assert isinstance(reader, BaseReader)


def test_output(prediction):
    assert prediction is not None
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0].answer == "Carla"
    assert prediction["answers"][0].offsets_in_context[0].start == 11
    assert prediction["answers"][0].offsets_in_context[0].end == 16
    assert prediction["answers"][0].score <= 1
    assert prediction["answers"][0].score >= 0
    assert prediction["answers"][0].context == "My name is Carla and I live in Berlin"
    assert len(prediction["answers"]) == 5


def test_output_batch_single_query_single_doc_list(reader, docs):
    prediction = reader.predict_batch(queries=["Who lives in Berlin?"], documents=docs, top_k=5)
    assert prediction is not None
    assert prediction["queries"] == ["Who lives in Berlin?"]
    # Expected output: List of lists of answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], Answer)
    assert len(prediction["answers"]) == 5  # Predictions for 5 docs


def test_output_batch_single_query_multiple_doc_lists(reader, docs):
    prediction = reader.predict_batch(queries=["Who lives in Berlin?"], documents=[docs, docs], top_k=5)
    assert prediction is not None
    assert prediction["queries"] == ["Who lives in Berlin?"]
    # Expected output: List of lists of answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], Answer)
    assert len(prediction["answers"]) == 2  # Predictions for 2 collection of docs
    assert len(prediction["answers"][0]) == 5  # top-k of 5 per collection of docs


def test_output_batch_multiple_queries_single_doc_list(reader, docs):
    prediction = reader.predict_batch(
        queries=["Who lives in Berlin?", "Who lives in New York?"], documents=docs, top_k=5
    )
    assert prediction is not None
    assert prediction["queries"] == ["Who lives in Berlin?", "Who lives in New York?"]
    # Expected output: List of lists of lists of answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], list)
    assert isinstance(prediction["answers"][0][0][0], Answer)
    assert len(prediction["answers"]) == 2  # Predictions for 2 queries
    assert len(prediction["answers"][0]) == 5  # Predictions for 5 documents


def test_output_batch_multiple_queries_multiple_doc_lists(reader, docs):
    prediction = reader.predict_batch(
        queries=["Who lives in Berlin?", "Who lives in New York?"], documents=[docs, docs], top_k=5
    )
    assert prediction is not None
    assert prediction["queries"] == ["Who lives in Berlin?", "Who lives in New York?"]
    # Expected output: List of lists answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], Answer)
    assert len(prediction["answers"]) == 2  # Predictions for 2 collections of documents
    assert len(prediction["answers"][0]) == 5  # top-k of 5 for collection of docs


@pytest.mark.integration
def test_no_answer_output(no_answer_prediction):
    assert no_answer_prediction is not None
    assert no_answer_prediction["query"] == "What is the meaning of life?"
    assert math.isclose(no_answer_prediction["no_ans_gap"], 0.9094805717468262, rel_tol=0.0001)
    assert no_answer_prediction["answers"][0].answer == ""
    assert no_answer_prediction["answers"][0].offsets_in_context[0].start == 0
    assert no_answer_prediction["answers"][0].offsets_in_context[0].end == 0
    assert no_answer_prediction["answers"][0].score <= 1
    assert no_answer_prediction["answers"][0].score >= 0
    assert no_answer_prediction["answers"][0].context == None
    assert no_answer_prediction["answers"][0].document_id == None
    answers = [x.answer for x in no_answer_prediction["answers"]]
    assert answers.count("") == 1
    assert len(no_answer_prediction["answers"]) == 5


# TODO Directly compare farm and transformers reader outputs
# TODO checks to see that model is responsive to input arguments e.g. context_window_size - topk


@pytest.mark.integration
def test_prediction_attributes(prediction):
    # TODO FARM's prediction also has no_ans_gap
    attributes_gold = ["query", "answers"]
    for ag in attributes_gold:
        assert ag in prediction


@pytest.mark.integration
def test_model_download_options():
    # download disabled and model is not cached locally
    with pytest.raises(OSError):
        impossible_reader = FARMReader("mfeb/albert-xxlarge-v2-squad2", local_files_only=True, num_processes=0)


def test_answer_attributes(prediction):
    # TODO Transformers answer also has meta key
    answer = prediction["answers"][0]
    assert type(answer) == Answer
    attributes_gold = ["answer", "score", "context", "offsets_in_context", "offsets_in_document", "type"]
    for ag in attributes_gold:
        assert getattr(answer, ag, None) is not None


@pytest.mark.integration
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("window_size", [10, 15, 20])
def test_context_window_size(reader, docs, window_size):
    assert isinstance(reader, FARMReader)

    old_window_size = reader.inferencer.model.prediction_heads[0].context_window_size
    reader.inferencer.model.prediction_heads[0].context_window_size = window_size

    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=5)
    for answer in prediction["answers"]:
        # If the extracted answer is larger than the context window, the context window is expanded.
        # If the extracted answer is odd in length, the resulting context window is one less than context_window_size
        # due to rounding (See FARM's QACandidate)
        # TODO Currently the behaviour of context_window_size in FARMReader and TransformerReader is different
        if len(answer.answer) <= window_size:
            assert len(answer.context) in [window_size, window_size - 1]
        else:
            # If the extracted answer is larger than the context window and is odd in length,
            # the resulting context window is one more than the answer length
            assert len(answer.context) in [len(answer.answer), len(answer.answer) + 1]

    reader.inferencer.model.prediction_heads[0].context_window_size = old_window_size

    # TODO Need to test transformers reader
    # TODO Currently the behaviour of context_window_size in FARMReader and TransformerReader is different


@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("top_k", [2, 5, 10])
def test_top_k(reader, docs, top_k):

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


def test_farm_reader_invalid_params():
    # invalid max_seq_len (greater than model maximum seq length)
    with pytest.raises(Exception):
        reader = FARMReader(model_name_or_path="deepset/tinyroberta-squad2", use_gpu=False, max_seq_len=513)

    # invalid max_seq_len (max_seq_len >= doc_stride)
    with pytest.raises(Exception):
        reader = FARMReader(
            model_name_or_path="deepset/tinyroberta-squad2", use_gpu=False, max_seq_len=129, doc_stride=128
        )

    # invalid doc_stride (doc_stride >= (max_seq_len - max_query_length))
    with pytest.raises(Exception):
        reader = FARMReader(model_name_or_path="deepset/tinyroberta-squad2", use_gpu=False, doc_stride=999)


def test_farm_reader_update_params(docs):
    reader = FARMReader(
        model_name_or_path="deepset/bert-medium-squad2-distilled", use_gpu=False, no_ans_boost=0, num_processes=0
    )

    # original reader
    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)
    assert len(prediction["answers"]) == 3
    assert prediction["answers"][0].answer == "Carla"

    # update no_ans_boost
    reader.update_parameters(
        context_window_size=100, no_ans_boost=100, return_no_answer=True, max_seq_len=384, doc_stride=128
    )
    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)
    assert len(prediction["answers"]) == 3
    assert prediction["answers"][0].answer == ""

    # update no_ans_boost
    reader.update_parameters(
        context_window_size=100, no_ans_boost=0, return_no_answer=False, max_seq_len=384, doc_stride=128
    )
    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)
    assert len(prediction["answers"]) == 3
    assert None not in [ans.answer for ans in prediction["answers"]]

    # update context_window_size
    reader.update_parameters(context_window_size=6, no_ans_boost=-10, max_seq_len=384, doc_stride=128)
    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)
    assert len(prediction["answers"]) == 3
    assert len(prediction["answers"][0].context) == 6

    # update doc_stride with invalid value
    with pytest.raises(Exception):
        reader.update_parameters(context_window_size=100, no_ans_boost=-10, max_seq_len=384, doc_stride=999)
        reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)

    # update max_seq_len with invalid value
    with pytest.raises(Exception):
        reader.update_parameters(context_window_size=6, no_ans_boost=-10, max_seq_len=99, doc_stride=128)
        reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)

    # update max_seq_len with invalid value (greater than the model maximum sequence length)
    with pytest.raises(Exception):
        invalid_max_seq_len = reader.inferencer.processor.tokenizer.model_max_length + 1
        reader.update_parameters(
            context_window_size=100, no_ans_boost=-10, max_seq_len=invalid_max_seq_len, doc_stride=128
        )
        reader.predict(query="Who lives in Berlin?", documents=docs, top_k=3)


@pytest.mark.parametrize("use_confidence_scores", [True, False])
def test_farm_reader_uses_same_sorting_as_QAPredictionHead(use_confidence_scores):
    reader = FARMReader(
        model_name_or_path="deepset/bert-medium-squad2-distilled",
        use_gpu=False,
        num_processes=0,
        return_no_answer=True,
        use_confidence_scores=use_confidence_scores,
    )

    text = """Beer is one of the oldest[1][2][3] and most widely consumed[4] alcoholic drinks in the world, and the third most popular drink overall after water and tea.[5] It is produced by the brewing and fermentation of starches, mainly derived from cereal grains—most commonly from malted barley, though wheat, maize (corn), rice, and oats are also used. During the brewing process, fermentation of the starch sugars in the wort produces ethanol and carbonation in the resulting beer.[6] Most modern beer is brewed with hops, which add bitterness and other flavours and act as a natural preservative and stabilizing agent. Other flavouring agents such as gruit, herbs, or fruits may be included or used instead of hops. In commercial brewing, the natural carbonation effect is often removed during processing and replaced with forced carbonation.[7]
Some of humanity's earliest known writings refer to the production and distribution of beer: the Code of Hammurabi included laws regulating beer and beer parlours,[8] and "The Hymn to Ninkasi", a prayer to the Mesopotamian goddess of beer, served as both a prayer and as a method of remembering the recipe for beer in a culture with few literate people.[9][10]
Beer is distributed in bottles and cans and is also commonly available on draught, particularly in pubs and bars. The brewing industry is a global business, consisting of several dominant multinational companies and many thousands of smaller producers ranging from brewpubs to regional breweries. The strength of modern beer is usually around 4% to 6% alcohol by volume (ABV), although it may vary between 0.5% and 20%, with some breweries creating examples of 40% ABV and above.[11]
Beer forms part of the culture of many nations and is associated with social traditions such as beer festivals, as well as a rich pub culture involving activities like pub crawling, pub quizzes and pub games.
When beer is distilled, the resulting liquor is a form of whisky.[12]
"""

    docs = [Document(text)]
    query = "What is the third most popular drink?"

    reader_predictions = reader.predict(query=query, documents=docs, top_k=5)

    farm_input = [QAInput(doc_text=d.content, questions=Question(query)) for d in docs]
    inferencer_predictions = reader.inferencer.inference_from_objects(farm_input, return_json=False)

    for answer, qa_cand in zip(reader_predictions["answers"], inferencer_predictions[0].prediction):
        assert answer.answer == ("" if qa_cand.answer_type == "no_answer" else qa_cand.answer)
        assert answer.offsets_in_document[0].start == qa_cand.offset_answer_start
        assert answer.offsets_in_document[0].end == qa_cand.offset_answer_end
        if use_confidence_scores:
            assert answer.score == qa_cand.confidence
        else:
            assert answer.score == qa_cand.score


@pytest.mark.parametrize("model_name", ["deepset/tinyroberta-squad2", "deepset/bert-medium-squad2-distilled"])
def test_farm_reader_onnx_conversion_and_inference(model_name, tmpdir, docs):
    FARMReader.convert_to_onnx(model_name=model_name, output_path=Path(tmpdir, "onnx"))
    assert os.path.exists(Path(tmpdir, "onnx", "model.onnx"))
    assert os.path.exists(Path(tmpdir, "onnx", "processor_config.json"))
    assert os.path.exists(Path(tmpdir, "onnx", "onnx_model_config.json"))
    assert os.path.exists(Path(tmpdir, "onnx", "language_model_config.json"))

    reader = FARMReader(str(Path(tmpdir, "onnx")))
    result = reader.predict(query="Where does Paul live?", documents=[docs[0]])
    assert result["answers"][0].answer == "New York"
