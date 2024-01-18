from haystack.utils.hf import resolve_hf_device_map
from haystack.utils import ComponentDevice, DeviceMap


def test_resolve_hf_device_map_str():
    component_device = resolve_hf_device_map(device_map="cpu:0")
    assert component_device == ComponentDevice.from_str("cpu:0")


def test_resolve_hf_device_map_dict():
    component_device = resolve_hf_device_map(device_map={"layer_1": 1, "classifier": "cpu"})
    assert component_device == ComponentDevice.from_multiple(DeviceMap.from_hf({"layer_1": 1, "classifier": "cpu"}))
