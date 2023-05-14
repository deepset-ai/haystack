import pytest

from haystack.modeling.model.language_model import (
    get_language_model,
    HFLanguageModel,
    HFLanguageModelNoSegmentIds,
    HFLanguageModelWithPooler,
    DPREncoder,
)


@pytest.mark.integration
@pytest.mark.parametrize(
    "pretrained_model_name_or_path, lm_class",
    [
        ("google/bert_uncased_L-2_H-128_A-2", HFLanguageModel),
        ("google/electra-small-generator", HFLanguageModelWithPooler),
        ("distilbert-base-uncased", HFLanguageModelNoSegmentIds),
        ("deepset/bert-small-mm_retrieval-passage_encoder", DPREncoder),
    ],
)
def test_basic_loading(pretrained_model_name_or_path, lm_class, monkeypatch):
    monkeypatch.setattr(lm_class, "__init__", lambda self, *a, **k: None)
    lm = get_language_model(pretrained_model_name_or_path)
    assert isinstance(lm, lm_class)


@pytest.mark.unit
def test_basic_loading_unknown_model():
    with pytest.raises(OSError):
        get_language_model("model_that_doesnt_exist")


@pytest.mark.unit
def test_basic_loading_with_empty_string():
    with pytest.raises(ValueError):
        get_language_model("")


@pytest.mark.unit
def test_basic_loading_invalid_params():
    with pytest.raises(ValueError):
        get_language_model(None)
