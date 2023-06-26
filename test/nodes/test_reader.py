import math
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from huggingface_hub import snapshot_download
from haystack.modeling.data_handler.inputs import QAInput, Question

from haystack.schema import Document, Answer, Label, MultiLabel, Span
from haystack.nodes.reader.base import BaseReader
from haystack.nodes import FARMReader, TransformersReader


def _joinpath(rootdir, targetdir):
    return os.path.join(os.sep, rootdir + os.sep, targetdir)


# TODO Fix bug in test_no_answer_output when using
# @pytest.fixture(params=["farm", "transformers"])
@pytest.fixture(params=["farm"])
def no_answer_reader(request):
    if request.param == "farm":
        return FARMReader(
            model_name_or_path="deepset/bert-medium-squad2-distilled",
            use_gpu=False,
            top_k_per_sample=5,
            no_ans_boost=0,
            return_no_answer=True,
            num_processes=0,
        )
    if request.param == "transformers":
        return TransformersReader(
            model_name_or_path="deepset/bert-medium-squad2-distilled",
            tokenizer="deepset/bert-medium-squad2-distilled",
            use_gpu=-1,
            top_k_per_candidate=5,
            return_no_answers=True,
        )


def test_reader_basic(reader):
    assert reader is not None
    assert isinstance(reader, BaseReader)


def test_output(reader, docs):
    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=5)
    assert prediction is not None
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0].answer == "Carla"
    assert prediction["answers"][0].offsets_in_context[0].start == 11
    assert prediction["answers"][0].offsets_in_context[0].end == 16
    assert prediction["answers"][0].offsets_in_document[0].start == 11
    assert prediction["answers"][0].offsets_in_document[0].end == 16
    assert prediction["answers"][0].type == "extractive"
    assert 0 <= prediction["answers"][0].score <= 1
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


def test_output_batch_single_query_single_nested_doc_list(reader, docs):
    prediction = reader.predict_batch(queries=["Who lives in Berlin?"], documents=[docs], top_k=5)
    assert prediction is not None
    assert prediction["queries"] == ["Who lives in Berlin?"]
    # Expected output: List of lists answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], Answer)
    assert len(prediction["answers"]) == 1  # Predictions for 1 collections of documents
    assert len(prediction["answers"][0]) == 5  # top-k of 5 for collection of docs


@pytest.mark.integration
def test_no_answer_output(no_answer_reader, docs):
    no_answer_prediction = no_answer_reader.predict(query="What is the meaning of life?", documents=docs, top_k=5)
    assert no_answer_prediction is not None
    assert no_answer_prediction["query"] == "What is the meaning of life?"
    assert math.isclose(no_answer_prediction["no_ans_gap"], 0.9094805717468262, rel_tol=0.0001)
    assert no_answer_prediction["answers"][0].answer == ""
    assert no_answer_prediction["answers"][0].offsets_in_context[0].start == 0
    assert no_answer_prediction["answers"][0].offsets_in_context[0].end == 0
    assert no_answer_prediction["answers"][0].score <= 1
    assert no_answer_prediction["answers"][0].score >= 0
    assert no_answer_prediction["answers"][0].context == None
    assert no_answer_prediction["answers"][0].document_ids == None
    answers = [x.answer for x in no_answer_prediction["answers"]]
    assert answers.count("") == 1
    assert len(no_answer_prediction["answers"]) == 5


@pytest.mark.integration
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_deduplication_for_overlapping_documents(reader):
    docs = [
        Document(
            content="My name is Carla. I live in Berlin.",
            id="doc1",
            meta={"_split_id": 0, "_split_overlap": [{"doc_id": "doc2", "range": (18, 35)}]},
        ),
        Document(
            content="I live in Berlin. My friends call me Carla.",
            id="doc2",
            meta={"_split_id": 1, "_split_overlap": [{"doc_id": "doc1", "range": (0, 17)}]},
        ),
    ]
    prediction = reader.predict(query="Where does Carla live?", documents=docs, top_k=5)

    # Check that there are no duplicate answers
    assert len(set(ans.answer for ans in prediction["answers"])) == len(prediction["answers"])


@pytest.mark.integration
def test_model_download_options():
    # download disabled and model is not cached locally
    with pytest.raises(OSError):
        impossible_reader = FARMReader("mfeb/albert-xxlarge-v2-squad2", local_files_only=True, num_processes=0)


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


# There are 5 different ways to load a FARMReader model.
# 1. HuggingFace Hub (online load)
# 2. HuggingFace downloaded (local load)
# 3. HF Model saved as FARM Model (same works for trained FARM model) (local load)
# 4. FARM Model converted to transformers (same as hf local model) (local load)
# 5. ONNX Model load (covered by test_farm_reader_onnx_conversion_and_inference)
@pytest.mark.integration
def test_farm_reader_load_hf_online():
    # Test Case: 1. HuggingFace Hub (online load)

    hf_model = "hf-internal-testing/tiny-random-RobertaForQuestionAnswering"
    _ = FARMReader(model_name_or_path=hf_model, use_gpu=False, no_ans_boost=0, num_processes=0)


@pytest.mark.integration
def test_farm_reader_load_hf_local(tmp_path):
    # Test Case: 2. HuggingFace downloaded (local load)

    hf_model = "hf-internal-testing/tiny-random-RobertaForQuestionAnswering"
    local_model_path = "locally_saved_hf"

    local_model_path = str(Path.joinpath(tmp_path, local_model_path))
    model_path = snapshot_download(repo_id=hf_model, revision="main", cache_dir=local_model_path)
    _ = FARMReader(model_name_or_path=model_path, use_gpu=False, no_ans_boost=0, num_processes=0)


@pytest.mark.integration
def test_farm_reader_load_farm_local(tmp_path):
    # Test Case: 3. HF Model saved as FARM Model (same works for trained FARM model) (local load)

    hf_model = "hf-internal-testing/tiny-random-RobertaForQuestionAnswering"
    local_model_path = f"{tmp_path}/locally_saved_farm"
    reader = FARMReader(model_name_or_path=hf_model, use_gpu=False, no_ans_boost=0, num_processes=0)
    reader.save(Path(local_model_path))
    _ = FARMReader(model_name_or_path=local_model_path, use_gpu=False, no_ans_boost=0, num_processes=0)


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


@pytest.mark.parametrize(
    "model_name",
    ["deepset/tinyroberta-squad2", "deepset/bert-medium-squad2-distilled", "deepset/xlm-roberta-base-squad2-distilled"],
)
def test_farm_reader_onnx_conversion_and_inference(model_name, tmpdir, docs):
    FARMReader.convert_to_onnx(model_name=model_name, output_path=Path(tmpdir, "onnx"))
    assert os.path.exists(Path(tmpdir, "onnx", "model.onnx"))
    assert os.path.exists(Path(tmpdir, "onnx", "processor_config.json"))
    assert os.path.exists(Path(tmpdir, "onnx", "onnx_model_config.json"))
    assert os.path.exists(Path(tmpdir, "onnx", "language_model_config.json"))

    reader = FARMReader(str(Path(tmpdir, "onnx")))
    result = reader.predict(query="Where does Paul live?", documents=[docs[0]])
    assert result["answers"][0].answer == "New York"


LABELS = [
    MultiLabel(
        labels=[
            Label(
                query="Who lives in Berlin?",
                answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                document=Document(
                    id="a0747b83aea0b60c4b114b15476dd32d", content_type="text", content=""  # empty document
                ),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
    MultiLabel(
        labels=[
            Label(
                query="Who lives in Munich?",
                answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                document=Document(
                    id="something_else", content_type="text", content="My name is Carla and I live in Munich"
                ),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            )
        ]
    ),
]


def test_reader_skips_empty_documents(reader):
    predictions, _ = reader.run(query=LABELS[0].labels[0].query, documents=[LABELS[0].labels[0].document])
    assert predictions["answers"] == []  # no answer given for query as document is empty
    predictions, _ = reader.run_batch(
        queries=[l.labels[0].query for l in LABELS], documents=[[l.labels[0].document] for l in LABELS]
    )
    assert predictions["answers"][0] == []  # no answer given for 1st query as document is empty
    assert predictions["answers"][1][0].answer == "Carla"  # answer given for 2nd query as usual


@pytest.mark.parametrize("no_answer_reader", ["farm", "transformers"], indirect=True)
def test_no_answer_reader_skips_empty_documents(no_answer_reader):
    predictions, _ = no_answer_reader.run(query=LABELS[0].labels[0].query, documents=[LABELS[0].labels[0].document])
    assert predictions["answers"][0].answer == ""  # Return no_answer as document is empty
    predictions, _ = no_answer_reader.run_batch(
        queries=[l.labels[0].query for l in LABELS], documents=[[l.labels[0].document] for l in LABELS]
    )
    assert predictions["answers"][0][0].answer == ""  # Return no_answer for 1st query as document is empty
    assert predictions["answers"][1][1].answer == "Carla"  # answer given for 2nd query as usual


@pytest.mark.integration
def test_reader_training_returns_eval(tmp_path, samples_path):
    max_seq_len = 16
    max_query_length = 8
    reader = FARMReader(
        model_name_or_path="deepset/tinyroberta-squad2",
        use_gpu=False,
        num_processes=0,
        max_seq_len=max_seq_len,
        doc_stride=2,
        max_query_length=max_query_length,
    )

    save_dir = f"{tmp_path}/test_dpr_training"
    reader.train(
        data_dir=str(samples_path / "squad"),
        train_filename="tiny.json",
        dev_filename="tiny.json",
        n_epochs=1,
        batch_size=1,
        grad_acc_steps=1,
        evaluate_every=0,
        save_dir=save_dir,
        max_seq_len=max_seq_len,
        max_query_length=max_query_length,
    )
    assert reader.inferencer.model.training is False


@pytest.mark.integration
def test_reader_training(tmp_path, samples_path):
    max_seq_len = 16
    max_query_length = 8
    reader = FARMReader(
        model_name_or_path="deepset/tinyroberta-squad2",
        use_gpu=False,
        num_processes=0,
        max_seq_len=max_seq_len,
        doc_stride=2,
        max_query_length=max_query_length,
    )

    save_dir = f"{tmp_path}/test_dpr_training"
    reader.train(
        data_dir=str(samples_path / "squad"),
        train_filename="tiny.json",
        dev_filename="tiny.json",
        test_filename="tiny.json",
        n_epochs=1,
        batch_size=1,
        grad_acc_steps=1,
        save_dir=save_dir,
        evaluate_every=2,
        max_seq_len=max_seq_len,
        max_query_length=max_query_length,
    )


@pytest.mark.integration
def test_reader_long_document(reader):
    # Check that long documents with >2^16 characters do not result in negative offsets
    docs = [Document(content=("abbreviation " * 2550) + "Christelle lives in Madrid.")]
    res = reader.predict(query="Where does Christelle live?", documents=docs)
    assert res["answers"][0].offsets_in_document[0].start >= 0
    assert res["answers"][0].offsets_in_document[0].end >= 0


@pytest.mark.unit
@patch("haystack.nodes.reader.farm.QAInferencer")
def test_farmreader_predict_preprocessor_batching(mocked_qa_inferencer, docs):
    reader = FARMReader(model_name_or_path="mocked_model", preprocessing_batch_size=2)
    reader.predict(query="sample query", documents=docs)

    # We expect 3 calls to the QAInferencer (5 docs / 2 batch_size)
    assert reader.inferencer.inference_from_objects.call_count == 3


@pytest.mark.unit
@patch("haystack.nodes.reader.farm.QAInferencer")
def test_farmreader_predict_batch_preprocessor_batching(mocked_qa_inferencer, docs):
    reader = FARMReader(model_name_or_path="mocked_model", preprocessing_batch_size=2)
    reader.predict_batch(queries=["sample query 1", "sample_query_2"], documents=docs)

    # We expect 5 calls to the QAInferencer (2 queries * 5 docs / 2 batch_size)
    assert reader.inferencer.inference_from_objects.call_count == 5
