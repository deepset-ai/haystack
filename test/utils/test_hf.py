import logging
from haystack.utils.hf import resolve_hf_device_map
from haystack.utils.device import ComponentDevice


def test_resolve_hf_device_map_only_device():
    model_kwargs = resolve_hf_device_map(device=None, model_kwargs={})
    assert model_kwargs["device_map"] == ComponentDevice.resolve_device(None).to_hf()


def test_resolve_hf_device_map_only_device_map():
    model_kwargs = resolve_hf_device_map(device=None, model_kwargs={"device_map": "cpu"})
    assert model_kwargs["device_map"] == "cpu"


def test_resolve_hf_device_map_device_and_device_map(caplog):
    with caplog.at_level(logging.WARNING):
        model_kwargs = resolve_hf_device_map(
            device=ComponentDevice.from_str("cpu"), model_kwargs={"device_map": "cuda:0"}
        )
        assert "The parameters `device` and `device_map` from `model_kwargs` are both provided." in caplog.text
    assert model_kwargs["device_map"] == "cuda:0"
