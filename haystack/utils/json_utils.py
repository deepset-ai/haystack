# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any


def extract_json_from_text(text: str) -> str:
    """
    Extracts JSON content from a string, handling markdown code blocks if present.

    :param text: The text containing JSON, possibly wrapped in markdown code blocks.
    :return: The extracted JSON string, or the original text if no markdown code blocks are found.
    """
    # Try to find JSON inside a markdown code block with "json" language specifier
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: Try to find JSON inside a generic markdown code block
    match = re.search(r"```(?!\w)\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no markdown code block, return the original text stripped of whitespace
    return text.strip()


def parse_json_from_text(text: str, expected_keys: list[str] | None = None) -> Any:
    """
    Parses a JSON string (or a string containing JSON in a markdown code block).

    :param text: The string to parse.
    :param expected_keys: A list of keys that must be present in the parsed JSON object (if it's a dict).
    :return: The parsed JSON object.
    :raises json.JSONDecodeError: If the text is not valid JSON.
    :raises ValueError: If `expected_keys` are provided and the parsed JSON is not a dict or is missing keys.
    """
    cleaned_text = extract_json_from_text(text)

    if not cleaned_text:
        # Handle empty string or string with only whitespace
        raise json.JSONDecodeError("Expecting value", cleaned_text, 0)

    try:
        parsed_json = json.loads(cleaned_text)
    except json.JSONDecodeError:
        # Re-raise the exception to let the caller handle it (or log it as they prefer)
        # We don't log here to avoid duplicate logging if the caller handles it.
        raise

    if expected_keys:
        if not isinstance(parsed_json, dict):
            raise ValueError(f"Expected a JSON object (dict) but got {type(parsed_json).__name__}")

        missing_keys = [key for key in expected_keys if key not in parsed_json]
        if missing_keys:
            raise ValueError(f"Missing expected keys in JSON: {missing_keys}. Got keys: {list(parsed_json.keys())}")

    return parsed_json
