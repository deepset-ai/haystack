# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.errors import DeserializationError
from haystack.core.serialization import component_from_dict, import_class_by_name


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


def _copy_serialized_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of serialized data that `from_dict` implementations can safely deserialize in place.

    Deserialization replaces serialized values with live objects, for example a callable path with the callable
    itself or a nested component dictionary with the component. Doing that on the dictionary passed by the caller
    leaves it holding objects that are no longer serializable and that a second `from_dict` call cannot read,
    which is why `Pipeline.from_dict` copies its input before deserializing it.

    :param data:
        The serialized data a `from_dict` implementation received.
    :returns:
        A copy of the data. Objects that cannot be deep-copied, such as components and tools, are kept as they are.
    """
    # Local import to avoid a circular import at module load time.
    from haystack.core.pipeline.utils import _deepcopy_with_exceptions

    return _deepcopy_with_exceptions(data)
