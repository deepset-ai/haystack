# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any


def _parse_json_from_text(text: str, expected_keys: list[str] | None = None) -> Any:
    """
    Parses a JSON string (or a string containing JSON in a markdown code block).

    :param text: The string to parse.
    :param expected_keys: A list of keys that must be present in the parsed JSON object (if it's a dict).
    :return: The parsed JSON object.
    :raises json.JSONDecodeError: If the text is not valid JSON.
    :raises ValueError: If `expected_keys` are provided and the parsed JSON is not a dict or is missing keys.
    """
    # Try to find JSON inside a markdown code block with "json" language specifier
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned_text = match.group(1).strip()
    else:
        # Fallback: Try to find JSON inside a generic markdown code block
        match = re.search(r"```(?!\w)\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            cleaned_text = match.group(1).strip()
        else:
            # If no markdown code block, return the original text stripped of whitespace
            cleaned_text = text.strip()

    if not cleaned_text:
        # Handle empty string or string with only whitespace
        raise json.JSONDecodeError("Expecting value", cleaned_text, 0)

    parsed_json = json.loads(cleaned_text)

    if expected_keys:
        if not isinstance(parsed_json, dict):
            raise ValueError(f"Expected a JSON object (dict) but got {type(parsed_json).__name__}")

        missing_keys = [key for key in expected_keys if key not in parsed_json]
        if missing_keys:
            raise ValueError(f"Missing expected keys in JSON: {missing_keys}. Got keys: {list(parsed_json.keys())}")

    return parsed_json
