# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

from haystack import DeserializationError
from haystack.core.serialization import component_from_dict, default_from_dict, import_class_by_name


def deserialize_document_store_in_init_params_inplace(data: Dict[str, Any], key: str = "document_store"):
    """
    Deserializes a generic document store from the init_parameters of a serialized component in place.

    :param data:
        The dictionary to deserialize from.
    :param key:
        The key in the `data["init_parameters"]` dictionary where the document store is specified.
    :returns:
        The dictionary, with the document store deserialized.

    :raises DeserializationError:
        If the document store is not properly specified in the serialization data or its type cannot be imported.
    """
    init_params = data.get("init_parameters", {})
    if key not in init_params:
        raise DeserializationError(f"Missing '{key}' in serialization data")
    if "type" not in init_params[key]:
        raise DeserializationError(f"Missing 'type' in {key} serialization data")

    doc_store_data = data["init_parameters"][key]
    try:
        doc_store_class = import_class_by_name(doc_store_data["type"])
    except ImportError as e:
        raise DeserializationError(f"Class '{doc_store_data['type']}' not correctly imported") from e
    if hasattr(doc_store_class, "from_dict"):
        data["init_parameters"][key] = doc_store_class.from_dict(doc_store_data)
    else:
        data["init_parameters"][key] = default_from_dict(doc_store_class, doc_store_data)


def deserialize_chatgenerator_inplace(data: Dict[str, Any], key: str = "chat_generator"):
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
    if key not in data:
        raise DeserializationError(f"Missing '{key}' in serialization data")

    serialized_chat_generator = data[key]

    if not isinstance(serialized_chat_generator, dict):
        raise DeserializationError(f"The value of '{key}' is not a dictionary")

    if "type" not in serialized_chat_generator:
        raise DeserializationError(f"Missing 'type' in {key} serialization data")

    try:
        chat_generator_class = import_class_by_name(serialized_chat_generator["type"])
    except ImportError as e:
        raise DeserializationError(f"Class '{serialized_chat_generator['type']}' not correctly imported") from e

    data[key] = component_from_dict(cls=chat_generator_class, data=serialized_chat_generator, name="chat_generator")
