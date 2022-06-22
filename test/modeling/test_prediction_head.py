import logging

from haystack.modeling.model.adaptive_model import AdaptiveModel
from haystack.modeling.model.language_model import get_language_model
from haystack.modeling.model.prediction_head import QuestionAnsweringHead
from haystack.modeling.utils import set_all_seeds, initialize_device_settings


def test_prediction_head_load_save(tmp_path, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    devices, n_gpu = initialize_device_settings(use_cuda=False)
    lang_model = "bert-base-german-cased"

    language_model = get_language_model(lang_model)
    prediction_head = QuestionAnsweringHead()

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=devices[0],
    )

    model.save(tmp_path)
    model_loaded = AdaptiveModel.load(tmp_path, device="cpu")
    assert model_loaded is not None
