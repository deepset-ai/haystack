import pytest

from haystack.modeling.model.language_model import get_language_model


@pytest.mark.parametrize(
    "pretrained_model_name_or_path, lm_class",
    [
        ("google/bert_uncased_L-2_H-128_A-2", "HFLanguageModel"),
        ("google/electra-small-generator", "HFLanguageModelWithPooler"),
        ("distilbert-base-uncased", "HFLanguageModelNoSegmentIds"),
        ("deepset/bert-small-mm_retrieval-passage_encoder", "DPREncoder"),
    ],
)
def test_basic_loading(pretrained_model_name_or_path, lm_class):
    lm = get_language_model(pretrained_model_name_or_path)
    mod = __import__("haystack.modeling.model.language_model", fromlist=[lm_class])
    klass = getattr(mod, lm_class)
    assert isinstance(lm, klass)


def test_basic_loading_unknown_model():
    with pytest.raises(OSError):
        get_language_model("model_that_doesnt_exist")


def test_basic_loading_with_empty_string():
    with pytest.raises(ValueError):
        get_language_model("")


def test_basic_loading_invalid_params():
    with pytest.raises(ValueError):
        get_language_model(None)
