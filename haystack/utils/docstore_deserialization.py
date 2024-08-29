# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

from haystack import DeserializationError
from haystack.core.serialization import default_from_dict, import_class_by_name


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
