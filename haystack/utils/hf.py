from typing import Any, Dict, Union
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, DeviceMap


with LazyImport(message="Run 'pip install transformers[torch]'") as torch_import:
    import torch


def serialize_hf_model_kwargs(kwargs: Dict[str, Any]):
    """
    Recursively serialize HuggingFace specific model keyword arguments
    in-place to make them JSON serializable.
    """
    torch_import.check()

    for k, v in kwargs.items():
        # torch.dtype
        if isinstance(v, torch.dtype):
            kwargs[k] = str(v)

        if isinstance(v, dict):
            serialize_hf_model_kwargs(v)


def deserialize_hf_model_kwargs(kwargs: Dict[str, Any]):
    """
    Recursively deserialize HuggingFace specific model keyword arguments
    in-place to make them JSON serializable.
    """
    torch_import.check()

    for k, v in kwargs.items():
        # torch.dtype
        if isinstance(v, str) and v.startswith("torch."):
            dtype_str = v.split(".")[1]
            dtype = getattr(torch, dtype_str, None)
            if dtype is not None and isinstance(dtype, torch.dtype):
                kwargs[k] = dtype

        if isinstance(v, dict):
            deserialize_hf_model_kwargs(v)


def resolve_hf_device_map(device_map: Union[str, dict]) -> ComponentDevice:
    """
    Convert a HuggingFace device_map into a ComponentDevice
    """
    # Resolve device if device_map is provided in model_kwargs
    if isinstance(device_map, str):
        component_device = ComponentDevice.from_str(device_map)
    else:
        assert isinstance(device_map, dict)
        component_device = ComponentDevice.from_multiple(DeviceMap.from_hf(device_map))
    return component_device
