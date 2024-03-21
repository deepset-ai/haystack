from haystack.utils.object_inspection import get_parameter_info


def test_get_parameter_info_with_no_params():
    def func():
        pass

    params_info = get_parameter_info(func)
    assert params_info == {}


def test_get_parameter_info_no_default():
    def func(param_a):
        pass

    params_info = get_parameter_info(func)
    assert params_info == {"param_a": {"default_value": None, "optional": False}}


def test_get_parameter_info_with_defaults():
    def func(a, b=2, c=3):
        pass

    params_info = get_parameter_info(func)
    assert params_info == {
        "a": {"default_value": None, "optional": False},
        "b": {"default_value": 2, "optional": True},
        "c": {"default_value": 3, "optional": True},
    }


def test_get_parameter_info_with_annotations():
    def func(a: int, b: str = "hello"):
        pass

    params_info = get_parameter_info(func)
    assert params_info == {
        "a": {"default_value": None, "optional": False},
        "b": {"default_value": "hello", "optional": True},
    }
