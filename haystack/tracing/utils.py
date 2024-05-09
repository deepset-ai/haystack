# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Union

from haystack import logging

logger = logging.getLogger(__name__)

PRIMITIVE_TYPES = (bool, str, int, float)


def coerce_tag_value(value: Any) -> Union[bool, str, int, float]:
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
        serializable = _serializable_value(value)
        return json.dumps(serializable)
    except Exception as error:
        logger.debug("Failed to coerce tag value to string: {error}", error=error)

        # Our last resort is to convert the value to a string
        return str(value)


def _serializable_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_serializable_value(v) for v in value]

    if isinstance(value, dict):
        return {k: _serializable_value(v) for k, v in value.items()}

    if getattr(value, "to_dict", None):
        return _serializable_value(value.to_dict())

    return value
