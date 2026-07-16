# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, get_args, get_origin

from haystack.core.errors import DeserializationError
from haystack.core.serialization import component_from_dict, import_class_by_name
from haystack.utils.type_serialization import _is_union_type

if TYPE_CHECKING:
    from haystack.core.pipeline.base import PipelineBase


def deserialize_chatgenerator_inplace(data: dict[str, Any], key: str = "chat_generator") -> None:
    """
    Deserialize a ChatGenerator in a dictionary inplace.

    :param data:
        The dictionary with the serialized data.
    :param key:
        The key in the dictionary where the ChatGenerator is stored.

    :raises DeserializationError:
        If the key is missing in the serialized data, the value is not a dictionary,
        the type key is missing, the class cannot be imported, or the class lacks a 'from_dict' method.
    """
    deserialize_component_inplace(data, key=key)


def deserialize_component_inplace(data: dict[str, Any], key: str = "chat_generator") -> None:
    """
    Deserialize a Component in a dictionary inplace.

    :param data:
        The dictionary with the serialized data.
    :param key:
        The key in the dictionary where the Component is stored. Default is "chat_generator".

    :raises DeserializationError:
        If the key is missing in the serialized data, the value is not a dictionary,
        the type key is missing, the class cannot be imported, or the class lacks a 'from_dict' method.
    """
    if key not in data:
        raise DeserializationError(f"Missing '{key}' in serialization data")

    serialized_component = data[key]

    if not isinstance(serialized_component, dict):
        raise DeserializationError(f"The value of '{key}' is not a dictionary")

    if "type" not in serialized_component:
        raise DeserializationError(f"Missing 'type' in {key} serialization data")

    try:
        component_class = import_class_by_name(serialized_component["type"])
    except ImportError as e:
        raise DeserializationError(f"Class '{serialized_component['type']}' not correctly imported") from e

    data[key] = component_from_dict(cls=component_class, data=serialized_component, name=key)


def _resolve_from_dict_class(type_: Any) -> Any:
    """
    Resolve the class whose `from_dict` method can deserialize values of `type_`.

    Handles plain classes, `list[T]`, and optionals/unions of those. In unions, the first arm resolving to a
    `from_dict`-capable class wins.

    :param type_: The type to inspect.
    :returns: The resolved class, or None if `type_` involves no `from_dict`-capable class.
    """
    if _is_union_type(type_):
        for arm in get_args(type_):
            resolved = _resolve_from_dict_class(arm)
            if resolved is not None:
                return resolved
        return None
    if get_origin(type_) is list:
        args = get_args(type_)
        return _resolve_from_dict_class(args[0]) if args else None
    return type_ if hasattr(type_, "from_dict") else None


def _deserialize_from_dict(value: Any, target_class: Any) -> Any:
    """
    Deserialize `value` using `target_class.from_dict`.

    Dictionaries are deserialized directly, lists element-wise (leaving non-dictionary items untouched). Any other
    value is returned unchanged.

    :param value: The value to deserialize.
    :param target_class: The class providing the `from_dict` method.
    :returns: The deserialized value.
    """
    if isinstance(value, dict):
        return target_class.from_dict(value)
    if isinstance(value, list):
        return [target_class.from_dict(item) if isinstance(item, dict) else item for item in value]
    return value


def _coerce_input_value(value: Any, socket_types: list[Any]) -> Any:
    for type_ in socket_types:
        target_class = _resolve_from_dict_class(type_)
        if target_class is not None:
            return _deserialize_from_dict(value, target_class)
    return value


def coerce_pipeline_inputs(pipeline: "PipelineBase", data: dict[str, Any]) -> dict[str, Any]:
    """
    Deserialize serialized Haystack objects in pipeline input data, based on the pipeline's input socket types.

    For every provided value whose input socket type involves a class exposing a `from_dict` method
    (such as `ChatMessage` or `Document`), plain dictionaries are converted into instances of that class.
    Socket types of the form `T`, `list[T]`, and optionals/unions of those are supported. Values that are
    already deserialized, or whose socket types involve no `from_dict`-capable class, are returned unchanged.

    Like `Pipeline.run`, `data` accepts the nested format (`{"component_name": {"input_name": value}}`) and
    the flat format (`{"input_name": value}`). The same format detection rules apply and the returned
    dictionary keeps the format of `data`.

    Usage example:
    ```python
    serialized_inputs = {"prompt_builder": {"template": [message.to_dict() for message in messages]}}
    inputs = coerce_pipeline_inputs(pipeline, serialized_inputs)
    result = pipeline.run(inputs)
    ```

    :param pipeline: The pipeline whose input socket types drive the coercion.
    :param data: The pipeline input data, in nested or flat format.
    :returns: A new dictionary with the same structure as `data` and deserialized values.
    :raises Exception: Any error raised by a `from_dict` method if a dictionary does not match the expected format.
    """
    # mirrors the format detection in Pipeline.run: nested if all values are dictionaries
    if all(isinstance(value, dict) for value in data.values()):
        available_inputs = pipeline.inputs(include_components_with_connected_inputs=True)
        coerced: dict[str, Any] = {}
        for component_name, component_inputs in data.items():
            sockets = available_inputs.get(component_name, {})
            coerced[component_name] = {
                input_name: _coerce_input_value(value, [sockets[input_name]["type"]] if input_name in sockets else [])
                for input_name, value in component_inputs.items()
            }
        return coerced

    # flat format: input names are matched across components like in Pipeline.run
    available_inputs = pipeline.inputs()
    return {
        input_name: _coerce_input_value(
            value, [sockets[input_name]["type"] for sockets in available_inputs.values() if input_name in sockets]
        )
        for input_name, value in data.items()
    }
