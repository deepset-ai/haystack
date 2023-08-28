from canals import component, Pipeline
from haystack.preview.dataclasses import *


# TODO use the Canals implementation once we're ready to upgrade to 0.7.0
# from canals.serialization import default_to_dict, default_from_dict

from typing import Type, Dict, Any
from canals.errors import DeserializationError


def default_to_dict(obj: Any, **init_parameters) -> Dict[str, Any]:
    return {"type": obj.__class__.__name__, "init_parameters": init_parameters}


def default_from_dict(cls: Type[object], data: Dict[str, Any]) -> Any:
    init_params = data.get("init_parameters", {})
    if "type" not in data:
        raise DeserializationError("Missing 'type' in serialization data")
    if data["type"] != cls.__name__:
        raise DeserializationError(f"Class '{data['type']}' can't be deserialized as '{cls.__name__}'")
    return cls(**init_params)
