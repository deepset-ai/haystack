import logging

from haystack.basics.modeling.adaptive_model import AdaptiveModel
from haystack.basics.modeling.language_model import LanguageModel
from haystack.basics.modeling.prediction_head import QuestionAnsweringHead
from haystack.basics.utils import set_all_seeds, initialize_device_settings


def test_prediction_head_load_save(tmp_path, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    lang_model = "bert-base-german-cased"

    language_model = LanguageModel.load(lang_model)
    prediction_head = QuestionAnsweringHead()

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    model.save(tmp_path)
    model_loaded = AdaptiveModel.load(tmp_path, device='cpu')
    assert model_loaded is not None
