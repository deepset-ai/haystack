from unittest.mock import patch
import pytest
from pathlib import Path
from urllib.error import URLError
from haystack.nodes.query_classifier import TransformersQueryClassifier, SklearnQueryClassifier
from ..conftest import fail_at_version


@pytest.mark.unit
@fail_at_version(1, 21)
def test_sklearnqueryclassifier_deprecation():
    with pytest.warns(DeprecationWarning):
        try:
            SklearnQueryClassifier(Path("fake_model"), Path("fake_vectorizer"))
        except URLError:
            pass


@pytest.mark.unit
def test_query_classifier_initialized_with_token_instead_of_use_auth_token():
    with patch("haystack.nodes.query_classifier.transformers.pipeline") as mock_transformers_pipeline:
        classifier = TransformersQueryClassifier(task="zero-shot-classification")
        assert "token" in mock_transformers_pipeline.call_args.kwargs
        assert "use_auth_token" not in mock_transformers_pipeline.call_args.kwargs


@pytest.fixture
def transformers_query_classifier():
    return TransformersQueryClassifier(
        model_name_or_path="shahrukhx01/bert-mini-finetune-question-detection",
        use_gpu=False,
        task="text-classification",
        labels=["LABEL_1", "LABEL_0"],
    )


@pytest.fixture
def zero_shot_transformers_query_classifier():
    return TransformersQueryClassifier(
        model_name_or_path="typeform/distilbert-base-uncased-mnli",
        use_gpu=False,
        task="zero-shot-classification",
        labels=["happy", "unhappy", "neutral"],
    )


def test_transformers_query_classifier(transformers_query_classifier):
    output = transformers_query_classifier.run(query="morse code")
    assert output == ({}, "output_2")

    output = transformers_query_classifier.run(query="How old is John?")
    assert output == ({}, "output_1")


def test_transformers_query_classifier_batch(transformers_query_classifier):
    queries = ["morse code", "How old is John?"]
    output = transformers_query_classifier.run_batch(queries=queries)

    assert output[0] == {"output_2": {"queries": ["morse code"]}, "output_1": {"queries": ["How old is John?"]}}


def test_zero_shot_transformers_query_classifier(zero_shot_transformers_query_classifier):
    output = zero_shot_transformers_query_classifier.run(query="What's the answer?")
    assert output == ({}, "output_3")

    output = zero_shot_transformers_query_classifier.run(query="Would you be so kind to tell me the answer?")
    assert output == ({}, "output_1")

    output = zero_shot_transformers_query_classifier.run(query="Can you give me the right answer for once??")
    assert output == ({}, "output_2")


def test_zero_shot_transformers_query_classifier_batch(zero_shot_transformers_query_classifier):
    queries = [
        "What's the answer?",
        "Would you be so kind to tell me the answer?",
        "Can you give me the right answer for once??",
    ]

    output = zero_shot_transformers_query_classifier.run_batch(queries=queries)

    assert output[0] == {
        "output_3": {"queries": ["What's the answer?"]},
        "output_1": {"queries": ["Would you be so kind to tell me the answer?"]},
        "output_2": {"queries": ["Can you give me the right answer for once??"]},
    }


def test_transformers_query_classifier_wrong_labels():
    with pytest.raises(ValueError, match="For text-classification, the provided labels must match the model labels"):
        query_classifier = TransformersQueryClassifier(
            model_name_or_path="shahrukhx01/bert-mini-finetune-question-detection",
            use_gpu=False,
            task="text-classification",
            labels=["WRONG_LABEL_1", "WRONG_LABEL_2", "WRONG_LABEL_3"],
        )


def test_transformers_query_classifier_no_labels():
    with pytest.raises(ValueError, match="The labels must be provided"):
        query_classifier = TransformersQueryClassifier(
            model_name_or_path="shahrukhx01/bert-mini-finetune-question-detection",
            use_gpu=False,
            task="text-classification",
            labels=None,
        )


def test_transformers_query_classifier_unsupported_task():
    with pytest.raises(ValueError, match="Task not supported"):
        query_classifier = TransformersQueryClassifier(
            model_name_or_path="shahrukhx01/bert-mini-finetune-question-detection",
            use_gpu=False,
            task="summarization",
            labels=["LABEL_1", "LABEL_0"],
        )
