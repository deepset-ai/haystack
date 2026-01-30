# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from haystack import logging

logger = logging.getLogger(__name__)

PRIMITIVE_TYPES = (bool, str, int, float)


def coerce_tag_value(value: Any) -> bool | str | int | float:
    """
    Coerces span tag values to compatible types for the tracing backend.

    Most tracing libraries don't support sending complex types to the backend. Hence, we need to convert them to
    compatible types.

    :param value: an arbitrary value which should be coerced to a compatible type
    :return: the value coerced to a compatible type
    """
    if isinstance(value, PRIMITIVE_TYPES):
        return value

    if value is None:
        return ""

    try:
        # do that with-in try-except because who knows what kind of objects are being passed
        serializable = _serializable_value(value=value, use_placeholders=True)
        return json.dumps(serializable)
    except Exception as error:
        logger.debug("Failed to coerce tag value to string: {error}", error=error)

        # Our last resort is to convert the value to a string
        return str(value)


def _serializable_value(value: Any, use_placeholders: bool = True) -> Any:
    """
    Serializes a value into a format suitable for tracing.

    :param value:
        The value to serialize.
    :param use_placeholders:
        Whether to use string placeholders for large objects like ByteStream and ImageContent.
    :returns:
        The serialized value.
    """
    if isinstance(value, list):
        return [_serializable_value(value=v, use_placeholders=use_placeholders) for v in value]

    if isinstance(value, dict):
        return {k: _serializable_value(value=v, use_placeholders=use_placeholders) for k, v in value.items()}

    if use_placeholders and getattr(value, "_to_trace_dict", None):
        return _serializable_value(value._to_trace_dict())

    if getattr(value, "to_dict", None):
        return _serializable_value(value.to_dict())

    return value
