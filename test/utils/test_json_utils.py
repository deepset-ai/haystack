# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from haystack.utils.json_utils import extract_json_from_text, parse_json_from_text


class TestJsonUtils:
    def test_extract_json_from_text_plain(self):
        text = '{"key": "value"}'
        assert extract_json_from_text(text) == text

    def test_extract_json_from_text_markdown(self):
        text = '```json\n{"key": "value"}\n```'
        assert extract_json_from_text(text) == '{"key": "value"}'
    
    def test_extract_json_from_text_markdown_with_text_around(self):
        text = 'Here is the JSON:\n```json\n{"key": "value"}\n```\nHope this helps.'
        assert extract_json_from_text(text) == '{"key": "value"}'

    def test_extract_json_from_text_case_insensitive(self):
        text = '```JSON\n{"key": "value"}\n```'
        assert extract_json_from_text(text) == '{"key": "value"}'

    def test_parse_json_from_text_valid(self):
        text = '{"key": "value"}'
        assert parse_json_from_text(text) == {"key": "value"}

    def test_parse_json_from_text_valid_markdown(self):
        text = '```json\n{"key": "value"}\n```'
        assert parse_json_from_text(text) == {"key": "value"}

    def test_parse_json_from_text_invalid(self):
        text = 'invalid json'
        with pytest.raises(json.JSONDecodeError):
            parse_json_from_text(text)

    def test_parse_json_from_text_empty(self):
        text = ''
        with pytest.raises(json.JSONDecodeError):
            parse_json_from_text(text)

    def test_parse_json_from_text_expected_keys(self):
        text = '{"key1": "value1", "key2": "value2"}'
        result = parse_json_from_text(text, expected_keys=["key1"])
        assert result == {"key1": "value1", "key2": "value2"}

    def test_parse_json_from_text_missing_keys(self):
        text = '{"key1": "value1"}'
        with pytest.raises(ValueError, match="Missing expected keys"):
            parse_json_from_text(text, expected_keys=["key1", "key2"])

    def test_parse_json_from_text_not_dict_with_expected_keys(self):
        text = '["item1", "item2"]'
        with pytest.raises(ValueError, match="Expected a JSON object"):
            parse_json_from_text(text, expected_keys=["key1"])
