import pytest
from transformers import BertModel, ElectraModel, DistilBertModel, DPRContextEncoder
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.electra.modeling_electra import ElectraEncoder

from haystack.modeling.model.language_model import get_language_model, HFLanguageModel, HFLanguageModelWithPooler, \
    HFLanguageModelNoSegmentIds, DPREncoder


def test_basic_loading():
    lm = get_language_model("google/bert_uncased_L-2_H-128_A-2")
    assert lm is not None and isinstance(lm, HFLanguageModel)
    assert lm.name == "Bert"
    assert lm.output_dims == 128
    assert isinstance(lm.model, BertModel)
    assert lm.language == "english"
    assert isinstance(lm.encoder, BertEncoder)


def test_basic_loading_with_model_type():
    lm = get_language_model("google/bert_uncased_L-2_H-128_A-2", "bert")
    assert lm is not None and isinstance(lm, HFLanguageModel)
    assert lm.name == "Bert"
    assert lm.output_dims == 128
    assert isinstance(lm.model, BertModel)
    assert lm.language == "english"
    assert isinstance(lm.encoder, BertEncoder)


def test_basic_loading_with_pooler():
    lm = get_language_model("google/electra-small-generator", "electra")
    assert lm is not None and isinstance(lm, HFLanguageModelWithPooler)
    assert lm.name == "Electra"
    assert lm.output_dims == 256
    assert isinstance(lm.model, ElectraModel)
    assert lm.language == "english"
    assert isinstance(lm.encoder, ElectraEncoder)


def test_basic_loading_with_pooler():
    lm = get_language_model("google/electra-small-generator", "electra")
    assert lm is not None and isinstance(lm, HFLanguageModelWithPooler)
    assert lm.name == "Electra"
    assert lm.output_dims == 256
    assert isinstance(lm.model, ElectraModel)
    assert lm.language == "english"
    assert isinstance(lm.encoder, ElectraEncoder)


def test_basic_loading_with_segment_id():
    lm = get_language_model("distilbert-base-uncased")
    assert lm is not None and isinstance(lm, HFLanguageModelNoSegmentIds)
    assert lm.name == "DistilBert"
    assert lm.output_dims == 768
    assert isinstance(lm.model, DistilBertModel)
    assert lm.language == "english"


def test_basic_loading_with_segment_id():
    lm = get_language_model("distilbert-base-uncased")
    assert lm is not None and isinstance(lm, HFLanguageModelNoSegmentIds)
    assert lm.name == "DistilBert"
    assert lm.output_dims == 768
    assert isinstance(lm.model, DistilBertModel)
    assert lm.language == "english"


def test_basic_loading_with_dpr_encoder_no_model_type():
    # TODO raises exception because the model type is not specified, but we should handle this case no?
    # why don't we? Beacuse we don't know the model type (context|question)?
    # we actually do as it is usually specified in architecture field of the Config
    with pytest.raises(BaseException):
        lm = get_language_model("facebook/dpr-ctx_encoder-single-nq-base")


def test_basic_loading_with_dpr_encoder():
    lm = get_language_model("facebook/dpr-ctx_encoder-single-nq-base", "DPRContextEncoder")
    assert lm is not None and isinstance(lm, DPREncoder)
    assert lm.name == "DPRContextEncoder"
    assert lm.output_dims == 768
    assert isinstance(lm.model, DPRContextEncoder)
    assert lm.language == "english"


def test_basic_loading_unknown_model():
    with pytest.raises(OSError):
        get_language_model("model_that_doesnt_exist")


def test_basic_loading_wrong_model_tyoe():
    # adversarial unit test, clearly this is bert model, not roberta model
    # how should we handle these cases, clients are bound to make mistakes
    lm = get_language_model("google/bert_uncased_L-2_H-128_A-2", "roberta")
    assert lm is not None and isinstance(lm, HFLanguageModel)
    assert lm.name == "Bert"
    assert lm.output_dims == 128
    assert isinstance(lm.model, BertModel)


def test_basic_loading_invalid_params():
    with pytest.raises(ValueError):
        get_language_model(None)
