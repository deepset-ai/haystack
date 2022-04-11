import pytest

from haystack.schema import Document, Answer
from haystack.nodes.reader.farm import FARMReader
from haystack.nodes.reader.transformers import TransformersReader


TEST_DOCS = [
    Document(
        content="My name is Carla and I live in Berlin",
        meta={"meta_field": "test1", "name": "filename1", "date_field": "2020-03-01", "numeric_field": 5.5},
    ),
    Document(
        content="My name is Matteo and I live in Rome",
        meta={"meta_field": "test2", "name": "filename2", "date_field": "2019-01-01", "numeric_field": 0.0},
    ),
    Document(
        content="My name is Christelle and I live in Paris",
        meta={"meta_field": "test3", "name": "filename3", "date_field": "2018-10-01", "numeric_field": 4.5},
    ),
    Document(
        content="My name is Camila and I live in Madrid",
        meta={"meta_field": "test4", "name": "filename4", "date_field": "2021-02-01", "numeric_field": 3.0},
    ),
]
UNANSWERABLE_QUERY = "Why?"
ANSWERABLE_QUERY = "Who lives in Madrid?"
ANSWER = "Camila"
NO_ANSWER = ""


@pytest.fixture(params=["farm", "transformers"], scope="session")
def reader(request):
    if request.param == "farm":
        return FARMReader(
            model_name_or_path="distilbert-base-uncased-distilled-squad",
            use_gpu=False,
            top_k_per_sample=5,
            num_processes=0,
        )
    if request.param == "transformers":
        return TransformersReader(
            model_name_or_path="distilbert-base-uncased-distilled-squad",
            tokenizer="distilbert-base-uncased",
            use_gpu=-1,
        )


@pytest.mark.integration
def test_output(reader):
    prediction = reader.predict(query=ANSWERABLE_QUERY, documents=TEST_DOCS, top_k=5)

    assert prediction is not None
    assert prediction["query"] == ANSWERABLE_QUERY
    assert prediction["answers"][0].answer == ANSWER
    assert prediction["answers"][0].offsets_in_context[0].start == 11
    assert prediction["answers"][0].offsets_in_context[0].end == 17
    assert prediction["answers"][0].score <= 1
    assert prediction["answers"][0].score >= 0
    assert prediction["answers"][0].context == "My name is Carla and I live in Berlin"
    assert len(prediction["answers"]) == 5


@pytest.mark.integration
def test_prediction_attributes(reader):
    # TODO FARM's prediction also has no_ans_gap
    prediction = reader.predict(query=UNANSWERABLE_QUERY, documents=TEST_DOCS, top_k=5)

    attributes_gold = ["query", "answers"]
    for ag in attributes_gold:
        assert ag in prediction


def test_model_download_options():
    """Download is disabled and the model is not cached locally"""
    with pytest.raises(OSError):
        FARMReader("mfeb/albert-xxlarge-v2-squad2", local_files_only=True, num_processes=0)


@pytest.mark.integration
def test_answer_attributes(reader):
    # TODO Transformers answer also has meta key
    prediction = reader.predict(query=UNANSWERABLE_QUERY, documents=TEST_DOCS, top_k=5)

    answer = prediction["answers"][0]
    assert type(answer) == Answer
    attributes_gold = ["answer", "score", "context", "offsets_in_context", "offsets_in_document", "type"]
    for ag in attributes_gold:
        assert getattr(answer, ag, None) is not None


#
# FARMReader specific
#


@pytest.mark.integration
def test_farmreader_no_answer_output():
    reader = FARMReader(
        model_name_or_path="distilbert-base-uncased-distilled-squad",
        use_gpu=False,
        top_k_per_sample=5,
        return_no_answer=True,
        num_processes=0,
    )
    prediction = reader.predict(query=UNANSWERABLE_QUERY, documents=TEST_DOCS, top_k=5)

    print(prediction)

    assert prediction is not None
    assert prediction["query"] == UNANSWERABLE_QUERY
    assert prediction["answers"][0].answer == NO_ANSWER
    assert prediction["answers"][0].offsets_in_context[0].start == 0
    assert prediction["answers"][0].offsets_in_context[0].end == 0
    assert prediction["answers"][0].score <= 1
    assert prediction["answers"][0].score >= 0
    assert prediction["answers"][0].context == None
    assert prediction["answers"][0].document_id == None
    answers = [x.answer for x in prediction["answers"]]
    assert answers.count("") == 1
    assert len(prediction["answers"]) == 5


# TODO Directly compare farm and transformers reader outputs
# TODO checks to see that model is responsive to input arguments e.g. context_window_size - topk


@pytest.mark.integration
@pytest.mark.parametrize("window_size", [10, 15, 20])
def test_farmreader_context_window_size(reader, window_size):
    reader = FARMReader(
        model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=False, top_k_per_sample=5, num_processes=0
    )
    prediction = reader.predict(query=UNANSWERABLE_QUERY, documents=TEST_DOCS, top_k=5)

    old_window_size = reader.inferencer.model.prediction_heads[0].context_window_size
    reader.inferencer.model.prediction_heads[0].context_window_size = window_size

    prediction = reader.predict(query="Who lives in Berlin?", documents=TEST_DOCS, top_k=5)
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


@pytest.mark.integration
@pytest.mark.parametrize("top_k", [2, 5, 10])
def test_farmreader_top_k(reader, top_k):
    reader = FARMReader(
        model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=False, top_k_per_sample=5, num_processes=0
    )

    old_top_k_per_candidate = reader.top_k_per_candidate
    reader.top_k_per_candidate = 4
    reader.inferencer.model.prediction_heads[0].n_best = reader.top_k_per_candidate + 1
    old_top_k_per_sample = reader.inferencer.model.prediction_heads[0].n_best_per_sample
    reader.inferencer.model.prediction_heads[0].n_best_per_sample = 4

    prediction = reader.predict(query=ANSWERABLE_QUERY, documents=TEST_DOCS, top_k=top_k)
    assert len(prediction["answers"]) == top_k

    reader.top_k_per_candidate = old_top_k_per_candidate
    reader.inferencer.model.prediction_heads[0].n_best = reader.top_k_per_candidate + 1
    reader.inferencer.model.prediction_heads[0].n_best_per_sample = old_top_k_per_sample


@pytest.mark.integration
def test_farm_reader_update_params():
    reader = FARMReader(
        model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=False, no_ans_boost=0, num_processes=0
    )

    # original reader
    prediction = reader.predict(query=ANSWERABLE_QUERY, documents=TEST_DOCS, top_k=3)
    assert len(prediction["answers"]) == 3
    assert prediction["answers"][0].answer == ANSWER

    # update no_ans_boost
    reader.update_parameters(
        context_window_size=100, no_ans_boost=100, return_no_answer=True, max_seq_len=384, doc_stride=128
    )
    prediction = reader.predict(query=ANSWERABLE_QUERY, documents=TEST_DOCS, top_k=3)
    assert len(prediction["answers"]) == 3
    assert prediction["answers"][0].answer == NO_ANSWER

    # update no_ans_boost
    reader.update_parameters(
        context_window_size=100, no_ans_boost=0, return_no_answer=False, max_seq_len=384, doc_stride=128
    )
    prediction = reader.predict(query=ANSWERABLE_QUERY, documents=TEST_DOCS, top_k=3)
    assert len(prediction["answers"]) == 3
    assert None not in [ans.answer for ans in prediction["answers"]]

    # update context_window_size
    reader.update_parameters(context_window_size=6, no_ans_boost=-10, max_seq_len=384, doc_stride=128)
    prediction = reader.predict(query=ANSWERABLE_QUERY, documents=TEST_DOCS, top_k=3)
    assert len(prediction["answers"]) == 3
    assert len(prediction["answers"][0].context) == 6

    # update doc_stride with invalid value
    with pytest.raises(Exception):
        reader.update_parameters(context_window_size=100, no_ans_boost=-10, max_seq_len=384, doc_stride=999)
        reader.predict(query=ANSWERABLE_QUERY, documents=TEST_DOCS, top_k=3)

    # update max_seq_len with invalid value
    with pytest.raises(Exception):
        reader.update_parameters(context_window_size=6, no_ans_boost=-10, max_seq_len=99, doc_stride=128)
        reader.predict(query=ANSWERABLE_QUERY, documents=TEST_DOCS, top_k=3)
