import json
import logging
from typing import Any, Union


logger = logging.getLogger(__name__)


def coerce_tag_value(value: Any) -> Union[bool, str, int, float]:
    """Coerces span tag values to compatible types for the tracing backend.

    Most tracing libraries don't support sending complex types to the backend. Hence, we need to convert them to
    compatible types.

    :param value: an arbitrary value which should be coerced to a compatible type
    :return: the value coerced to a compatible type
    """
    if isinstance(value, (bool, str, int, float)):
        return value

    if value is None:
        return ""

    try:
        return json.dumps(value)
    except Exception as error:
        logger.debug("Failed to coerce tag value to string: %s", error, exc_info=True)

        # Our last resort is to convert the value to a string
        return str(value)
