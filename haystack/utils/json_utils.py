# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from haystack import logging

logger = logging.getLogger(__name__)


def _parse_json_from_text(text: str, expected_keys: list[str] | None = None, raise_on_failure: bool = True) -> Any:
    """
    Parses a JSON string.

    :param text: The string to parse.
    :param expected_keys: A list of keys that must be present in the parsed JSON object (if it's a dict).
    :param raise_on_failure: If True, raises an exception on failure. If False, logs a warning and returns None.
    :return: The parsed JSON object, or None if parsing fails and raise_on_failure is False.
    :raises json.JSONDecodeError: If the text is not valid JSON and raise_on_failure is True.
    :raises ValueError: If `expected_keys` are provided and the parsed JSON is not a dict or is missing keys,
        and `raise_on_failure` is True.
    """
    try:
        # Create a clean text, though json.loads handles whitespace, strictly checking for empty string is useful
        cleaned_text = text.strip()

        if not cleaned_text:
            # Handle empty string or string with only whitespace
            raise json.JSONDecodeError("Expecting value", text, 0)

        parsed_json = json.loads(cleaned_text)

        if expected_keys:
            if not isinstance(parsed_json, dict):
                raise ValueError(f"Expected a JSON object (dict) but got {type(parsed_json).__name__}")

            missing_keys = [key for key in expected_keys if key not in parsed_json]
            if missing_keys:
                raise ValueError(f"Missing expected keys in JSON: {missing_keys}. Got keys: {list(parsed_json.keys())}")

        return parsed_json

    except (json.JSONDecodeError, ValueError) as e:
        if raise_on_failure:
            raise e
        logger.warning("Failed to parse JSON from text: {text}. Error: {error}", text=text, error=e)
        return None
