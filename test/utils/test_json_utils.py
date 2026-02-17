import json
import logging

import pytest

from haystack.utils.json_utils import _parse_json_from_text


class TestJsonUtils:
    def test__parse_json_from_text_valid(self):
        text = '{"key": "value"}'
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

    def test__parse_json_from_text_whitespace_only(self):
        text = "   \n\t  "
        with pytest.raises(json.JSONDecodeError):
            _parse_json_from_text(text)

    def test__parse_json_from_text_empty_expected_keys(self):
        text = '{"key": "value"}'
        # Should pass without validation if expected_keys is empty list
        assert _parse_json_from_text(text, expected_keys=[]) == {"key": "value"}

    def test__parse_json_from_text_invalid_raise_false(self, caplog):
        text = "invalid json"
        with caplog.at_level(logging.WARNING):
            result = _parse_json_from_text(text, raise_on_failure=False)
            assert result is None
            assert "Failed to parse JSON from text" in caplog.text

    def test__parse_json_from_text_missing_keys_raise_false(self, caplog):
        text = '{"key1": "value1"}'
        with caplog.at_level(logging.WARNING):
            result = _parse_json_from_text(text, expected_keys=["key1", "key2"], raise_on_failure=False)
            assert result is None
            assert "Failed to parse JSON from text" in caplog.text
