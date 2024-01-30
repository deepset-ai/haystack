import copy
import logging
from typing import Any, Dict, Optional
from haystack.lazy_imports import LazyImport
from haystack.utils.device import ComponentDevice


with LazyImport(message="Run 'pip install transformers[torch]'") as torch_import:
    import torch


logger = logging.getLogger(__name__)


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


def resolve_hf_device_map(device: Optional[ComponentDevice], model_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Update `model_kwargs` to include the keyword argument `device_map` based on `device` if `device_map` is not
    already present in `model_kwargs`. This method is useful you want to force loading a transformers model when using
    `AutoModel.from_pretrained` to use `device_map`.

    We handle the edge case where `device` and `device_map` is specified by ignoring the `device` parameter and printing
    a warning.

    :param device: The device on which the model is loaded. If `None`, the default device is automatically
        selected.
    :param model_kwargs: Additional HF keyword arguments passed to `AutoModel.from_pretrained`.
        For details on what kwargs you can pass, see the model's documentation.
    """
    model_kwargs = copy.copy(model_kwargs) or {}
    if model_kwargs.get("device_map"):
        if device is not None:
            logger.warning(
                "The parameters `device` and `device_map` from `model_kwargs` are both provided. "
                "Ignoring `device` and using `device_map`."
            )
        # Resolve device if device_map is provided in model_kwargs
        device_map = model_kwargs["device_map"]
    else:
        device_map = ComponentDevice.resolve_device(device).to_hf()

    # Set up device_map which allows quantized loading and multi device inference
    # requires accelerate which is always installed when using `pip install transformers[torch]`
    model_kwargs["device_map"] = device_map

    return model_kwargs
