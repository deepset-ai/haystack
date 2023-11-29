import pytest

from haystack.components.generators.hf_utils import check_generation_params


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
