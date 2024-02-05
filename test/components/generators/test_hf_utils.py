import pytest

from haystack.components.generators.hf_utils import check_generation_params, list_inference_deployed_models


def test_empty_dictionary():
    # no exception raised
    check_generation_params({})


def test_valid_generation_parameters():
    # these are valid parameters
    kwargs = {"max_new_tokens": 100, "temperature": 0.8}
    additional_accepted_params = None
    check_generation_params(kwargs, additional_accepted_params)


def test_invalid_generation_parameters():
    # these are invalid parameters
    kwargs = {"invalid_param": "value"}
    additional_accepted_params = None
    with pytest.raises(ValueError):
        check_generation_params(kwargs, additional_accepted_params)


def test_additional_accepted_params_empty_list():
    kwargs = {"temperature": 0.8}
    additional_accepted_params = []
    check_generation_params(kwargs, additional_accepted_params)


def test_additional_accepted_params_known_parameter():
    # both are valid parameters
    kwargs = {"temperature": 0.8}
    additional_accepted_params = ["max_new_tokens"]
    check_generation_params(kwargs, additional_accepted_params)


def test_additional_accepted_params_unknown_parameter():
    kwargs = {"strange_param": "value"}
    additional_accepted_params = ["strange_param"]
    # Although strange_param is not generation param the check_generation_params
    # does not raise exception because strange_param is passed as additional_accepted_params
    check_generation_params(kwargs, additional_accepted_params)


@pytest.mark.integration
def test_hf_free_tier_models():
    # maintainer note:
    # leave this test as integration test because it requires network access
    # and because we need to check if the default TGI models are still listed there
    # if the test fails, it's likely because the TGI free-tier models have changed or the API is down

    free_tier_models = list_inference_deployed_models()
    assert len(free_tier_models) > 0

    # first test some old models that should not be there
    assert "google/flan-t5-base" not in free_tier_models

    # and then test some new models that should be there
    # use this opportunity to test the default HuggingFaceTGIGenerator and HuggingFaceTGIChatGenerator
    # are still listed there
    assert "mistralai/Mistral-7B-v0.1" in free_tier_models
    assert "HuggingFaceH4/zephyr-7b-beta" in free_tier_models
