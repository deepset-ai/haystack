# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from haystack.utils.json_utils import _parse_json_from_text, extract_json_from_text


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

    def test__parse_json_from_text_valid(self):
        text = '{"key": "value"}'
        assert _parse_json_from_text(text) == {"key": "value"}

    def test__parse_json_from_text_valid_markdown(self):
        text = '```json\n{"key": "value"}\n```'
        assert _parse_json_from_text(text) == {"key": "value"}

    def test__parse_json_from_text_invalid(self):
        text = "invalid json"
        with pytest.raises(json.JSONDecodeError):
            _parse_json_from_text(text)

    def test__parse_json_from_text_empty(self):
        text = ""
        with pytest.raises(json.JSONDecodeError):
            _parse_json_from_text(text)

    def test__parse_json_from_text_expected_keys(self):
        text = '{"key1": "value1", "key2": "value2"}'
        result = _parse_json_from_text(text, expected_keys=["key1"])
        assert result == {"key1": "value1", "key2": "value2"}

    def test__parse_json_from_text_missing_keys(self):
        text = '{"key1": "value1"}'
        with pytest.raises(ValueError, match="Missing expected keys"):
            _parse_json_from_text(text, expected_keys=["key1", "key2"])

    def test_parse_json_from_text_not_dict_with_expected_keys(self):
        text = '["item1", "item2"]'
        with pytest.raises(ValueError, match="Expected a JSON object"):
            _parse_json_from_text(text, expected_keys=["key1"])

    def test_extract_json_from_text_generic_block(self):
        text = '```\n{"key": "value"}\n```'
        assert extract_json_from_text(text) == '{"key": "value"}'

    def test_extract_json_from_text_multiple_blocks(self):
        text = '```json\n{"first": 1}\n```\nSome text\n```json\n{"second": 2}\n```'
        # Should return the first match
        assert extract_json_from_text(text) == '{"first": 1}'

    def test_extract_json_from_text_whitespace_only(self):
        text = "   \n\t  "
        assert extract_json_from_text(text) == ""

    def test__parse_json_from_text_generic_block(self):
        text = '```\n{"key": "value"}\n```'
        assert _parse_json_from_text(text) == {"key": "value"}

    def test__parse_json_from_text_empty_expected_keys(self):
        text = '{"key": "value"}'
        # Should pass without validation if expected_keys is empty list
        assert _parse_json_from_text(text, expected_keys=[]) == {"key": "value"}

    def test_extract_json_from_text_json_block_priority_over_generic(self):
        # Generic block appears first, but json-labeled block should be preferred
        text = '```\n{"generic": 0}\n```\n```json\n{"labeled": 1}\n```'
        assert extract_json_from_text(text) == '{"labeled": 1}'
