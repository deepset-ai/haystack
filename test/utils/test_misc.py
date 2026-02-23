# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

import pytest

from haystack import Document
from haystack.utils.misc import _deduplicate_documents, _parse_dict_from_json


class TestDeduplicateDocuments:
    def test_deduplicate_documents_keeps_highest_score(self):
        documents = [
            Document(id="duplicate", content="keep me", score=0.9),
            Document(id="duplicate", content="drop me", score=0.1),
            Document(id="unique", content="unique"),
        ]

        result = _deduplicate_documents(documents)

        assert len(result) == 2
        assert result[0].content == "keep me"
        assert result[1].content == "unique"

    def test_deduplicate_documents_keeps_first_when_scores_missing(self):
        documents = [Document(id="duplicate", content="first"), Document(id="duplicate", content="second")]

        result = _deduplicate_documents(documents)

        assert len(result) == 1
        assert result[0].content == "first"


class TestJsonParsing:
    @pytest.fixture
    def mock_logger(self):
        with patch("haystack.utils.misc.logger") as mock:
            yield mock

    def test_parse_dict_from_json_valid(self):
        text = '{"key": "value"}'
        assert _parse_dict_from_json(text) == {"key": "value"}

    def test_parse_dict_from_json_invalid(self):
        text = "invalid json"
        with pytest.raises(json.JSONDecodeError):
            _parse_dict_from_json(text)

    def test_parse_dict_from_json_empty(self):
        text = ""
        with pytest.raises(json.JSONDecodeError):
            _parse_dict_from_json(text)

    def test_parse_dict_from_json_expected_keys(self):
        text = '{"key1": "value1", "key2": "value2"}'
        result = _parse_dict_from_json(text, expected_keys=["key1"])
        assert result == {"key1": "value1", "key2": "value2"}

    def test_parse_dict_from_json_missing_keys(self):
        text = '{"key1": "value1"}'
        with pytest.raises(ValueError, match="Missing expected keys"):
            _parse_dict_from_json(text, expected_keys=["key1", "key2"])

    def test_parse_dict_from_json_not_dict(self):
        text = '["item1", "item2"]'
        with pytest.raises(ValueError, match="Expected a JSON object"):
            _parse_dict_from_json(text, expected_keys=["key1"])

    def test_parse_dict_from_json_not_dict_raise_false(self, mock_logger):
        text = '["item1", "item2"]'
        result = _parse_dict_from_json(text, expected_keys=["key1"], raise_on_failure=False)
        assert result is None
        mock_logger.warning.assert_called_once()
        args, kwargs = mock_logger.warning.call_args
        assert "Expected a JSON object containing a dictionary but got {type}" in args[0]
        assert kwargs["type"] == "list"

    def test_parse_dict_from_json_invalid_raise_false(self, mock_logger):
        text = "invalid json"
        result = _parse_dict_from_json(text, raise_on_failure=False)
        assert result is None
        mock_logger.warning.assert_called_once()
        args, kwargs = mock_logger.warning.call_args
        assert "Failed to parse JSON from text: {text}" in args[0]
        assert kwargs["text"] == text
        assert isinstance(kwargs["error"], json.JSONDecodeError)

    def test_parse_dict_from_json_missing_keys_raise_false(self, mock_logger):
        text = '{"key1": "value1"}'
        result = _parse_dict_from_json(text, expected_keys=["key1", "key2"], raise_on_failure=False)
        assert result is None
        mock_logger.warning.assert_called_once()
        args, kwargs = mock_logger.warning.call_args
        assert "Missing expected keys in JSON: {missing_keys}" in args[0]
        assert kwargs["missing_keys"] == ["key2"]
        assert kwargs["keys"] == ["key1"]
