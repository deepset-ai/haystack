# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch

import pytest

from haystack.utils import ComponentDevice, Device, DeviceMap, DeviceType


def test_device_type():
    for e in DeviceType:
        assert e == DeviceType.from_str(e.value)

    with pytest.raises(ValueError, match="Unknown device type string"):
        DeviceType.from_str("tpu")


def test_device_creation():
    assert Device.cpu().type == DeviceType.CPU
    assert Device.gpu().type == DeviceType.GPU
    assert Device.mps().type == DeviceType.MPS
    assert Device.disk().type == DeviceType.DISK

    assert Device.from_str("cpu") == Device.cpu()
    assert Device.from_str("cuda:1") == Device.gpu(1)
    assert Device.from_str("disk") == Device.disk()
    assert Device.from_str("mps:0") == Device(DeviceType.MPS, 0)

    with pytest.raises(ValueError, match="Device id must be >= 0"):
        Device.gpu(-1)


def test_device_map():
    map = DeviceMap({"layer1": Device.cpu(), "layer2": Device.gpu(1), "layer3": Device.disk()})

    assert all(x in map for x in ["layer1", "layer2", "layer3"])
    assert len(map) == 3
    assert map["layer1"] == Device.cpu()
    assert map["layer2"] == Device.gpu(1)
    assert map["layer3"] == Device.disk()

    for k, d in map:
        assert k in ["layer1", "layer2", "layer3"]
        assert d in [Device.cpu(), Device.gpu(1), Device.disk()]

    map["layer1"] = Device.gpu(0)
    assert map["layer1"] == Device.gpu(0)

    map["layer2"] = Device.cpu()
    assert DeviceMap.from_hf({"layer1": 0, "layer2": "cpu", "layer3": "disk"}) == DeviceMap(
        {"layer1": Device.gpu(0), "layer2": Device.cpu(), "layer3": Device.disk()}
    )

    with pytest.raises(ValueError, match="unexpected device"):
        DeviceMap.from_hf({"layer1": 0.1})

    assert map.first_device == Device.gpu(0)
    assert DeviceMap({}).first_device is None


def test_component_device_empty_and_full():
    with pytest.raises(ValueError, match="neither be empty nor contain"):
        ComponentDevice().first_device

    with pytest.raises(ValueError, match="neither be empty nor contain"):
        ComponentDevice(Device.cpu(), DeviceMap({})).to_hf()


def test_component_device_single():
    single = ComponentDevice.from_single(Device.gpu(1))
    assert not single.has_multiple_devices
    assert single._single_device == Device.gpu(1)
    assert single._multiple_devices is None

    with pytest.raises(ValueError, match="disk device can only be used as a part of device maps"):
        ComponentDevice.from_single(Device.disk())

    assert single.to_torch_str() == "cuda:1"
    assert single.to_spacy() == 1
    assert single.to_hf() == "cuda:1"
    assert single.update_hf_kwargs({}, overwrite=False) == {"device": "cuda:1"}
    assert single.update_hf_kwargs({"device": 0}, overwrite=True) == {"device": "cuda:1"}
    assert single.first_device == ComponentDevice.from_single(single._single_device)


def test_component_device_multiple():
    multiple = ComponentDevice.from_multiple(
        DeviceMap({"layer1": Device.cpu(), "layer2": Device.gpu(1), "layer3": Device.disk()})
    )
    assert multiple.has_multiple_devices
    assert multiple._single_device is None
    assert multiple._multiple_devices == DeviceMap(
        {"layer1": Device.cpu(), "layer2": Device.gpu(1), "layer3": Device.disk()}
    )

    with pytest.raises(ValueError, match="Only single devices"):
        multiple.to_torch()

    with pytest.raises(ValueError, match="Only single devices"):
        multiple.to_torch_str()

    with pytest.raises(ValueError, match="Only single devices"):
        multiple.to_spacy()

    assert multiple.to_hf() == {"layer1": "cpu", "layer2": 1, "layer3": "disk"}
    assert multiple.update_hf_kwargs({}, overwrite=False) == {
        "device_map": {"layer1": "cpu", "layer2": 1, "layer3": "disk"}
    }
    assert multiple.update_hf_kwargs({"device_map": {None: None}}, overwrite=True) == {
        "device_map": {"layer1": "cpu", "layer2": 1, "layer3": "disk"}
    }
    assert multiple.first_device == ComponentDevice.from_single(Device.cpu())


@patch("torch.backends.mps.is_available")
@patch("torch.cuda.is_available")
def test_component_device_resolution(torch_cuda_is_available, torch_backends_mps_is_available):
    assert ComponentDevice.resolve_device(ComponentDevice.from_single(Device.cpu()))._single_device == Device.cpu()

    torch_cuda_is_available.return_value = True
    assert ComponentDevice.resolve_device(None)._single_device == Device.gpu(0)

    torch_cuda_is_available.return_value = False
    torch_backends_mps_is_available.return_value = True
    assert ComponentDevice.resolve_device(None)._single_device == Device.mps()

    torch_cuda_is_available.return_value = False
    torch_backends_mps_is_available.return_value = False
    assert ComponentDevice.resolve_device(None)._single_device == Device.cpu()

    torch_cuda_is_available.return_value = False
    torch_backends_mps_is_available.return_value = True
    os.environ["HAYSTACK_MPS_ENABLED"] = "false"
    assert ComponentDevice.resolve_device(None)._single_device == Device.cpu()
