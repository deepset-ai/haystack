# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from haystack.lazy_imports import LazyImport

with LazyImport(
    message="PyTorch must be installed to use torch.device or use GPU support in HuggingFace transformers. "
    "Run 'pip install \"transformers[torch]\"'"
) as torch_import:
    import torch


class DeviceType(Enum):
    """
    Represents device types supported by Haystack.

    This also includes devices that are not directly used by models - for example, the disk device is exclusively used
    in device maps for frameworks that support offloading model weights to disk.
    """

    CPU = "cpu"
    GPU = "cuda"
    DISK = "disk"
    MPS = "mps"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string: str) -> "DeviceType":
        """
        Create a device type from a string.

        :param string:
            The string to convert.
        :returns:
            The device type.
        """
        mapping = {e.value: e for e in DeviceType}
        _type = mapping.get(string)
        if _type is None:
            raise ValueError(f"Unknown device type string '{string}'")
        return _type


@dataclass
class Device:
    """
    A generic representation of a device.

    :param type:
        The device type.
    :param id:
        The optional device id.
    """

    type: DeviceType
    id: Optional[int] = field(default=None)

    def __init__(self, type: DeviceType, id: Optional[int] = None):  # noqa:A002
        """
        Create a generic device.

        :param type:
            The device type.
        :param id:
            The device id.
        """
        if id is not None and id < 0:
            raise ValueError(f"Device id must be >= 0, got {id}")

        self.type = type
        self.id = id

    def __str__(self):
        if self.id is None:
            return str(self.type)
        else:
            return f"{self.type}:{self.id}"

    @staticmethod
    def cpu() -> "Device":
        """
        Create a generic CPU device.

        :returns:
            The CPU device.
        """
        return Device(DeviceType.CPU)

    @staticmethod
    def gpu(id: int = 0) -> "Device":  # noqa:A002
        """
        Create a generic GPU device.

        :param id:
            The GPU id.
        :returns:
            The GPU device.
        """
        return Device(DeviceType.GPU, id)

    @staticmethod
    def disk() -> "Device":
        """
        Create a generic disk device.

        :returns:
            The disk device.
        """
        return Device(DeviceType.DISK)

    @staticmethod
    def mps() -> "Device":
        """
        Create a generic Apple Metal Performance Shader device.

        :returns:
            The MPS device.
        """
        return Device(DeviceType.MPS)

    @staticmethod
    def from_str(string: str) -> "Device":
        """
        Create a generic device from a string.

        :returns:
            The device.

        """
        device_type_str, device_id = _split_device_string(string)
        return Device(DeviceType.from_str(device_type_str), device_id)


@dataclass
class DeviceMap:
    """
    A generic mapping from strings to devices.

    The semantics of the strings are dependent on target framework. Primarily used to deploy HuggingFace models to
    multiple devices.

    :param mapping:
        Dictionary mapping strings to devices.
    """

    mapping: Dict[str, Device] = field(default_factory=dict, hash=False)

    def __getitem__(self, key: str) -> Device:
        return self.mapping[key]

    def __setitem__(self, key: str, value: Device):
        self.mapping[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.mapping

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self):
        return iter(self.mapping.items())

    def to_dict(self) -> Dict[str, str]:
        """
        Serialize the mapping to a JSON-serializable dictionary.

        :returns:
            The serialized mapping.
        """
        return {key: str(device) for key, device in self.mapping.items()}

    @property
    def first_device(self) -> Optional[Device]:
        """
        Return the first device in the mapping, if any.

        :returns:
            The first device.
        """
        if not self.mapping:
            return None
        else:
            return next(iter(self.mapping.values()))

    @staticmethod
    def from_dict(dict: Dict[str, str]) -> "DeviceMap":  # noqa:A002
        """
        Create a generic device map from a JSON-serialized dictionary.

        :param dict:
            The serialized mapping.
        :returns:
            The generic device map.
        """
        mapping = {}
        for key, device_str in dict.items():
            mapping[key] = Device.from_str(device_str)
        return DeviceMap(mapping)

    @staticmethod
    def from_hf(hf_device_map: Dict[str, Union[int, str, "torch.device"]]) -> "DeviceMap":
        """
        Create a generic device map from a HuggingFace device map.

        :param hf_device_map:
            The HuggingFace device map.
        :returns:
            The deserialized device map.
        """
        mapping = {}
        for key, device in hf_device_map.items():
            if isinstance(device, int):
                mapping[key] = Device(DeviceType.GPU, device)
            elif isinstance(device, str):
                device_type, device_id = _split_device_string(device)
                mapping[key] = Device(DeviceType.from_str(device_type), device_id)
            elif isinstance(device, torch.device):
                device_type = device.type
                device_id = device.index
                mapping[key] = Device(DeviceType.from_str(device_type), device_id)
            else:
                raise ValueError(
                    f"Couldn't convert HuggingFace device map - unexpected device '{str(device)}' for '{key}'"
                )
        return DeviceMap(mapping)


@dataclass(frozen=True)
class ComponentDevice:
    """
    A representation of a device for a component.

    This can be either a single device or a device map.
    """

    _single_device: Optional[Device] = field(default=None)
    _multiple_devices: Optional[DeviceMap] = field(default=None)

    @classmethod
    def from_str(cls, device_str: str) -> "ComponentDevice":
        """
        Create a component device representation from a device string.

        The device string can only represent a single device.

        :param device_str:
            The device string.
        :returns:
            The component device representation.
        """
        device = Device.from_str(device_str)
        return cls.from_single(device)

    @classmethod
    def from_single(cls, device: Device) -> "ComponentDevice":
        """
        Create a component device representation from a single device.

        Disks cannot be used as single devices.

        :param device:
            The device.
        :returns:
            The component device representation.
        """
        if device.type == DeviceType.DISK:
            raise ValueError("The disk device can only be used as a part of device maps")

        return cls(_single_device=device)

    @classmethod
    def from_multiple(cls, device_map: DeviceMap) -> "ComponentDevice":
        """
        Create a component device representation from a device map.

        :param device_map:
            The device map.
        :returns:
            The component device representation.
        """
        return cls(_multiple_devices=device_map)

    def _validate(self):
        """
        Validate the component device representation.
        """
        if not (self._single_device is not None) ^ (self._multiple_devices is not None):
            raise ValueError(
                "The component device can neither be empty nor contain both a single device and a device map"
            )

    def to_torch(self) -> "torch.device":
        """
        Convert the component device representation to PyTorch format.

        Device maps are not supported.

        :returns:
            The PyTorch device representation.
        """
        self._validate()

        if self._single_device is None:
            raise ValueError("Only single devices can be converted to PyTorch format")

        torch_import.check()
        assert self._single_device is not None
        return torch.device(str(self._single_device))

    def to_torch_str(self) -> str:
        """
        Convert the component device representation to PyTorch string format.

        Device maps are not supported.

        :returns:
            The PyTorch device string representation.
        """
        self._validate()

        if self._single_device is None:
            raise ValueError("Only single devices can be converted to PyTorch format")

        assert self._single_device is not None
        return str(self._single_device)

    def to_spacy(self) -> int:
        """
        Convert the component device representation to spaCy format.

        Device maps are not supported.

        :returns:
            The spaCy device representation.
        """
        self._validate()

        if self._single_device is None:
            raise ValueError("Only single devices can be converted to spaCy format")

        assert self._single_device is not None
        if self._single_device.type == DeviceType.GPU:
            assert self._single_device.id is not None
            return self._single_device.id
        else:
            return -1

    def to_hf(self) -> Union[Union[int, str], Dict[str, Union[int, str]]]:
        """
        Convert the component device representation to HuggingFace format.

        :returns:
            The HuggingFace device representation.
        """
        self._validate()

        def convert_device(device: Device, *, gpu_id_only: bool = False) -> Union[int, str]:
            if gpu_id_only and device.type == DeviceType.GPU:
                assert device.id is not None
                return device.id
            else:
                return str(device)

        if self._single_device is not None:
            return convert_device(self._single_device)

        assert self._multiple_devices is not None
        return {key: convert_device(device, gpu_id_only=True) for key, device in self._multiple_devices.mapping.items()}

    def update_hf_kwargs(self, hf_kwargs: Dict[str, Any], *, overwrite: bool) -> Dict[str, Any]:
        """
        Convert the component device representation to HuggingFace format.

        Add them as canonical keyword arguments to the keyword arguments dictionary.

        :param hf_kwargs:
            The HuggingFace keyword arguments dictionary.
        :param overwrite:
            Whether to overwrite existing device arguments.
        :returns:
            The HuggingFace keyword arguments dictionary.
        """
        self._validate()

        if not overwrite and any(x in hf_kwargs for x in ("device", "device_map")):
            return hf_kwargs

        converted = self.to_hf()
        key = "device_map" if self.has_multiple_devices else "device"
        hf_kwargs[key] = converted
        return hf_kwargs

    @property
    def has_multiple_devices(self) -> bool:
        """
        Whether this component device representation contains multiple devices.
        """
        self._validate()

        return self._multiple_devices is not None

    @property
    def first_device(self) -> Optional["ComponentDevice"]:
        """
        Return either the single device or the first device in the device map, if any.

        :returns:
            The first device.
        """
        self._validate()

        if self._single_device is not None:
            return self.from_single(self._single_device)

        assert self._multiple_devices is not None
        assert self._multiple_devices.first_device is not None
        return self.from_single(self._multiple_devices.first_device)

    @staticmethod
    def resolve_device(device: Optional["ComponentDevice"] = None) -> "ComponentDevice":
        """
        Select a device for a component. If a device is specified, it's used. Otherwise, the default device is used.

        :param device:
            The provided device, if any.
        :returns:
            The resolved device.
        """
        if not isinstance(device, ComponentDevice) and device is not None:
            raise ValueError(
                f"Invalid component device type '{type(device).__name__}'. Must either be None or ComponentDevice."
            )

        if device is None:
            device = ComponentDevice.from_single(_get_default_device())

        return device

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the component device representation to a JSON-serializable dictionary.

        :returns:
            The dictionary representation.
        """
        if self._single_device is not None:
            return {"type": "single", "device": str(self._single_device)}
        elif self._multiple_devices is not None:
            return {"type": "multiple", "device_map": self._multiple_devices.to_dict()}
        else:
            # Unreachable
            assert False

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> "ComponentDevice":  # noqa:A002
        """
        Create a component device representation from a JSON-serialized dictionary.

        :param dict:
            The serialized representation.
        :returns:
            The deserialized component device.
        """
        if dict["type"] == "single":
            return cls.from_str(dict["device"])
        elif dict["type"] == "multiple":
            return cls.from_multiple(DeviceMap.from_dict(dict["device_map"]))
        else:
            raise ValueError(f"Unknown component device type '{dict['type']}' in serialized data")


def _get_default_device() -> Device:
    """
    Return the default device for Haystack.

    Precedence:
        GPU > MPS > CPU. If PyTorch is not installed, only CPU is available.

    :returns:
        The default device.
    """
    try:
        torch_import.check()

        has_mps = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and os.getenv("HAYSTACK_MPS_ENABLED", "true") != "false"
        )
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_mps = False
        has_cuda = False

    if has_cuda:
        return Device.gpu()
    elif has_mps:
        return Device.mps()
    else:
        return Device.cpu()


def _split_device_string(string: str) -> Tuple[str, Optional[int]]:
    """
    Split a device string into device type and device id.

    :param string:
        The device string to split.
    :returns:
        The device type and device id, if any.
    """
    if ":" in string:
        device_type, device_id_str = string.split(":")
        try:
            device_id = int(device_id_str)
        except ValueError:
            raise ValueError(f"Device id must be an integer, got {device_id_str}")
    else:
        device_type = string
        device_id = None
    return device_type, device_id
