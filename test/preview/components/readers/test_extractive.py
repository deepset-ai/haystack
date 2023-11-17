from math import ceil, exp
from typing import List
from unittest.mock import patch, Mock
import pytest

import torch
from transformers import pipeline

from haystack.preview.components.readers import ExtractiveReader
from haystack.preview import Document


@pytest.fixture
def mock_tokenizer():
    def mock_tokenize(
        texts: List[str],
        text_pairs: List[str],
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
        return_overflowing_tokens: bool,
        stride: int,
    ):
        assert padding
        assert truncation
        assert return_tensors == "pt"
        assert return_overflowing_tokens

        tokens = Mock()

        num_splits = [ceil(len(text + pair) / max_length) for text, pair in zip(texts, text_pairs)]
        tokens.overflow_to_sample_mapping = [i for i, num in enumerate(num_splits) for _ in range(num)]
        num_samples = sum(num_splits)
        tokens.encodings = [Mock() for _ in range(num_samples)]
        sequence_ids = [0] * 16 + [1] * 16 + [None] * (max_length - 32)
        for encoding in tokens.encodings:
            encoding.sequence_ids = sequence_ids
            encoding.token_to_chars = lambda i: (i - 16, i - 15)
        tokens.input_ids = torch.zeros(num_samples, max_length, dtype=torch.int)
        attention_mask = torch.zeros(num_samples, max_length, dtype=torch.int)
        attention_mask[:32] = 1
        tokens.attention_mask = attention_mask
        return tokens

    with patch("haystack.preview.components.readers.extractive.AutoTokenizer.from_pretrained") as tokenizer:
        tokenizer.return_value = mock_tokenize
        yield tokenizer


@pytest.fixture()
def mock_reader(mock_tokenizer):
    class MockModel(torch.nn.Module):
        def to(self, device):
            assert device == "cpu:0"
            self.device_set = True
            return self

        def forward(self, input_ids, attention_mask, *args, **kwargs):
            assert input_ids.device == torch.device("cpu")
            assert attention_mask.device == torch.device("cpu")
            assert self.device_set
            start = torch.zeros(input_ids.shape[:2])
            end = torch.zeros(input_ids.shape[:2])
            start[:, 27] = 1
            end[:, 31] = 1
            end[:, 32] = 1
            prediction = Mock()
            prediction.start_logits = start
            prediction.end_logits = end
            return prediction

    with patch("haystack.preview.components.readers.extractive.AutoModelForQuestionAnswering.from_pretrained") as model:
        model.return_value = MockModel()
        reader = ExtractiveReader(model_name_or_path="mock-model", device="cpu:0")
        reader.warm_up()
        return reader


example_queries = ["Who is the chancellor of Germany?", "Who is the head of the department?"]
example_documents = [
    [
        Document(content="Angela Merkel was the chancellor of Germany."),
        Document(content="Olaf Scholz is the chancellor of Germany"),
        Document(content="Jerry is the head of the department."),
    ]
] * 2


@pytest.mark.unit
def test_to_dict():
    component = ExtractiveReader("my-model", token="secret-token", model_kwargs={"torch_dtype": "auto"})
    data = component.to_dict()

    assert data == {
        "type": "haystack.preview.components.readers.extractive.ExtractiveReader",
        "init_parameters": {
            "model_name_or_path": "my-model",
            "device": None,
            "token": None,  # don't serialize valid tokens
            "top_k": 20,
            "confidence_threshold": None,
            "max_seq_length": 384,
            "stride": 128,
            "max_batch_size": None,
            "answers_per_seq": None,
            "no_answer": True,
            "calibration_factor": 0.1,
            "model_kwargs": {"torch_dtype": "auto"},
        },
    }


@pytest.mark.unit
def test_to_dict_empty_model_kwargs():
    component = ExtractiveReader("my-model", token="secret-token")
    data = component.to_dict()

    assert data == {
        "type": "haystack.preview.components.readers.extractive.ExtractiveReader",
        "init_parameters": {
            "model_name_or_path": "my-model",
            "device": None,
            "token": None,  # don't serialize valid tokens
            "top_k": 20,
            "confidence_threshold": None,
            "max_seq_length": 384,
            "stride": 128,
            "max_batch_size": None,
            "answers_per_seq": None,
            "no_answer": True,
            "calibration_factor": 0.1,
            "model_kwargs": {},
        },
    }


@pytest.mark.unit
def test_output(mock_reader: ExtractiveReader):
    answers = mock_reader.run(example_queries[0], example_documents[0], top_k=3)[
        "answers"
    ]  # [0] Uncomment and remove first two indices when batching support is reintroduced
    doc_ids = set()
    no_answer_prob = 1
    for doc, answer in zip(example_documents[0], answers[:3]):
        assert answer.start == 11
        assert answer.end == 16
        assert doc.content is not None
        assert answer.data == doc.content[11:16]
        assert answer.probability == pytest.approx(1 / (1 + exp(-2 * mock_reader.calibration_factor)))
        no_answer_prob *= 1 - answer.probability
        doc_ids.add(doc.id)
    assert len(doc_ids) == 3
    assert answers[-1].probability == pytest.approx(no_answer_prob)


@pytest.mark.unit
def test_flatten_documents(mock_reader: ExtractiveReader):
    queries, docs, query_ids = mock_reader._flatten_documents(example_queries, example_documents)
    i = 0
    for j, query in enumerate(example_queries):
        for doc in example_documents[j]:
            assert queries[i] == query
            assert docs[i] == doc
            assert query_ids[i] == j
            i += 1
    assert len(docs) == len(queries) == len(query_ids) == i


@pytest.mark.unit
def test_preprocess(mock_reader: ExtractiveReader):
    _, _, seq_ids, _, query_ids, doc_ids = mock_reader._preprocess(
        example_queries * 3, example_documents[0], 384, [1, 1, 1], 0
    )
    expected_seq_ids = torch.full((3, 384), -1, dtype=torch.int)
    expected_seq_ids[:, :16] = 0
    expected_seq_ids[:, 16:32] = 1
    assert torch.equal(seq_ids, expected_seq_ids)
    assert query_ids == [1, 1, 1]
    assert doc_ids == [0, 1, 2]


def test_preprocess_splitting(mock_reader: ExtractiveReader):
    _, _, seq_ids, _, query_ids, doc_ids = mock_reader._preprocess(
        example_queries * 4, example_documents[0] + [Document(content="a" * 64)], 96, [1, 1, 1, 1], 0
    )
    assert seq_ids.shape[0] == 5
    assert query_ids == [1, 1, 1, 1, 1]
    assert doc_ids == [0, 1, 2, 3, 3]


@pytest.mark.unit
def test_postprocess(mock_reader: ExtractiveReader):
    start = torch.zeros((2, 8))
    start[0, 3] = 4
    start[0, 1] = 5  # test attention_mask
    start[0, 4] = 3
    start[1, 2] = 1

    end = torch.zeros((2, 8))
    end[0, 1] = 5  # test attention_mask
    end[0, 2] = 4  # test that end can't be before start
    end[0, 3] = 3
    end[0, 4] = 2
    end[1, :] = -10
    end[1, 4] = -1

    sequence_ids = torch.ones((2, 8))
    attention_mask = torch.ones((2, 8))
    attention_mask[0, :2] = 0
    encoding = Mock()
    encoding.token_to_chars = lambda i: (int(i), int(i) + 1)

    start_candidates, end_candidates, probs = mock_reader._postprocess(
        start, end, sequence_ids, attention_mask, 3, [encoding, encoding]
    )

    assert len(start_candidates) == len(end_candidates) == len(probs) == 2
    assert len(start_candidates[0]) == len(end_candidates[0]) == len(probs[0]) == 3
    assert start_candidates[0][0] == 3
    assert end_candidates[0][0] == 4
    assert start_candidates[0][1] == 3
    assert end_candidates[0][1] == 5
    assert start_candidates[0][2] == 4
    assert end_candidates[0][2] == 5
    assert probs[0][0] == pytest.approx(1 / (1 + exp(-7 * mock_reader.calibration_factor)))
    assert probs[0][1] == pytest.approx(1 / (1 + exp(-6 * mock_reader.calibration_factor)))
    assert probs[0][2] == pytest.approx(1 / (1 + exp(-5 * mock_reader.calibration_factor)))
    assert start_candidates[1][0] == 2
    assert end_candidates[1][0] == 5
    assert probs[1][0] == pytest.approx(1 / 2)


@pytest.mark.unit
def test_nest_answers(mock_reader: ExtractiveReader):
    start = list(range(5))
    end = [i + 5 for i in start]
    start = [start] * 6  # type: ignore
    end = [end] * 6  # type: ignore
    probabilities = torch.arange(5).unsqueeze(0) / 5 + torch.arange(6).unsqueeze(-1) / 25
    query_ids = [0] * 3 + [1] * 3
    document_ids = list(range(3)) * 2
    nested_answers = mock_reader._nest_answers(
        start, end, probabilities, example_documents[0], example_queries, 5, 3, None, query_ids, document_ids, True  # type: ignore
    )
    expected_no_answers = [0.2 * 0.16 * 0.12, 0]
    for query, answers, expected_no_answer, probabilities in zip(
        example_queries, nested_answers, expected_no_answers, [probabilities[:3, -1], probabilities[3:, -1]]
    ):
        assert len(answers) == 4
        for doc, answer, probability in zip(example_documents[0], reversed(answers[:3]), probabilities):
            assert answer.query == query
            assert answer.document == doc
            assert answer.probability == pytest.approx(probability)
        no_answer = answers[-1]
        assert no_answer.query == query
        assert no_answer.document is None
        assert no_answer.probability == pytest.approx(expected_no_answer)


@pytest.mark.unit
@patch("haystack.preview.components.readers.extractive.AutoTokenizer.from_pretrained")
@patch("haystack.preview.components.readers.extractive.AutoModelForQuestionAnswering.from_pretrained")
def test_warm_up_use_hf_token(mocked_automodel, mocked_autotokenizer):
    reader = ExtractiveReader("deepset/roberta-base-squad2", token="fake-token")
    reader.warm_up()

    mocked_automodel.assert_called_once_with("deepset/roberta-base-squad2", token="fake-token")
    mocked_autotokenizer.assert_called_once_with("deepset/roberta-base-squad2", token="fake-token")


@pytest.mark.integration
def test_t5():
    reader = ExtractiveReader("TARUNBHATT/flan-t5-small-finetuned-squad")
    reader.warm_up()
    answers = reader.run(example_queries[0], example_documents[0], top_k=2)[
        "answers"
    ]  # remove indices when batching support is reintroduced
    assert answers[0].data == "Angela Merkel"
    assert answers[0].probability == pytest.approx(0.7764519453048706)
    assert answers[1].data == "Olaf Scholz"
    assert answers[1].probability == pytest.approx(0.7703777551651001)
    assert answers[2].data is None
    assert answers[2].probability == pytest.approx(0.051331606147570596)
    # Uncomment assertions below when batching is reintroduced
    # assert answers[0][2].probability == pytest.approx(0.051331606147570596)
    # assert answers[1][0].data == "Jerry"
    # assert answers[1][0].probability == pytest.approx(0.7413333654403687)
    # assert answers[1][1].data == "Olaf Scholz"
    # assert answers[1][1].probability == pytest.approx(0.7266613841056824)
    # assert answers[1][2].data is None
    # assert answers[1][2].probability == pytest.approx(0.0707035798685709)


@pytest.mark.integration
def test_roberta():
    reader = ExtractiveReader("deepset/tinyroberta-squad2")
    reader.warm_up()
    answers = reader.run(example_queries[0], example_documents[0], top_k=2)[
        "answers"
    ]  # remove indices when batching is reintroduced
    assert answers[0].data == "Olaf Scholz"
    assert answers[0].probability == pytest.approx(0.8614975214004517)
    assert answers[1].data == "Angela Merkel"
    assert answers[1].probability == pytest.approx(0.857952892780304)
    assert answers[2].data is None
    assert answers[2].probability == pytest.approx(0.019673851661650588)
    # uncomment assertions below when there is batching in v2
    # assert answers[0][0].data == "Olaf Scholz"
    # assert answers[0][0].probability == pytest.approx(0.8614975214004517)
    # assert answers[0][1].data == "Angela Merkel"
    # assert answers[0][1].probability == pytest.approx(0.857952892780304)
    # assert answers[0][2].data is None
    # assert answers[0][2].probability == pytest.approx(0.0196738764278237)
    # assert answers[1][0].data == "Jerry"
    # assert answers[1][0].probability == pytest.approx(0.7048940658569336)
    # assert answers[1][1].data == "Olaf Scholz"
    # assert answers[1][1].probability == pytest.approx(0.6604189872741699)
    # assert answers[1][2].data is None
    # assert answers[1][2].probability == pytest.approx(0.1002123719777046)


@pytest.mark.integration
def test_matches_hf_pipeline():
    reader = ExtractiveReader("deepset/tinyroberta-squad2", device="cpu")
    reader.warm_up()
    answers = reader.run(example_queries[0], [[example_documents[0][0]]][0], top_k=20, no_answer=False)[
        "answers"
    ]  # [0] Remove first two indices when batching support is reintroduced
    pipe = pipeline("question-answering", model=reader.model, tokenizer=reader.tokenizer, align_to_words=False)
    answers_hf = pipe(
        question=example_queries[0],
        context=example_documents[0][0].content,
        max_answer_len=1_000,
        handle_impossible_answer=False,
        top_k=20,
    )  # We need to disable HF postprocessing features to make the results comparable. This is related to https://github.com/huggingface/transformers/issues/26286
    assert len(answers) == len(answers_hf) == 20
    for answer, answer_hf in zip(answers, answers_hf):
        assert answer.start == answer_hf["start"]
        assert answer.end == answer_hf["end"]
        assert answer.data == answer_hf["answer"]
