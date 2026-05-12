# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

import pytest

from haystack import Document
from haystack.utils.misc import (
    _deduplicate_documents,
    _normalize_metadata_field_name,
    _parse_dict_from_json,
    _reciprocal_rank_fusion,
)


class TestNormalizeMetadataFieldName:
    def test_removes_meta_prefix(self):
        assert _normalize_metadata_field_name("meta.year") == "year"
        assert _normalize_metadata_field_name("meta.category") == "category"

    def test_returns_unchanged_when_no_prefix(self):
        assert _normalize_metadata_field_name("year") == "year"
        assert _normalize_metadata_field_name("category") == "category"

    def test_meta_prefix_only_returns_empty_string(self):
        assert _normalize_metadata_field_name("meta.") == ""

    def test_does_not_strip_meta_substring(self):
        assert _normalize_metadata_field_name("my_meta_field") == "my_meta_field"


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


class TestReciprocalRankFusion:
    def test_empty_input_returns_empty(self):
        assert _reciprocal_rank_fusion([]) == []

    def test_single_list_assigns_scores(self):
        docs = [Document(id="a"), Document(id="b"), Document(id="c")]
        result = _reciprocal_rank_fusion([docs])
        assert len(result) == 3
        assert all(doc.score is not None for doc in result)

    def test_scores_decrease_with_rank(self):
        docs = [Document(id="a"), Document(id="b"), Document(id="c")]
        result = _reciprocal_rank_fusion([docs])
        by_id = {doc.id: doc.score for doc in result}
        assert by_id["a"] > by_id["b"] > by_id["c"]

    def test_deduplicates_across_lists(self):
        docs_a = [Document(id="a"), Document(id="b")]
        docs_b = [Document(id="b"), Document(id="c")]
        result = _reciprocal_rank_fusion([docs_a, docs_b])
        assert len(result) == 3
        assert {doc.id for doc in result} == {"a", "b", "c"}

    def test_higher_ranked_doc_gets_higher_score(self):
        docs_a = [Document(id="a"), Document(id="b"), Document(id="c")]
        docs_b = [Document(id="c"), Document(id="a"), Document(id="d")]
        result = _reciprocal_rank_fusion([docs_a, docs_b])
        by_id = {doc.id: doc.score for doc in result}
        # "a" is ranked 1st and 2nd; "c" is ranked 3rd and 1st — "a" should win
        assert by_id["a"] > by_id["c"]

    def test_equal_weights_by_default(self):
        docs_a = [Document(id="x")]
        docs_b = [Document(id="x")]
        result_default = _reciprocal_rank_fusion([docs_a, docs_b])
        result_explicit = _reciprocal_rank_fusion([docs_a, docs_b], weights=[0.5, 0.5])
        assert result_default[0].score == pytest.approx(result_explicit[0].score)

    def test_weights_influence_scores(self):
        docs_a = [Document(id="a"), Document(id="b")]
        docs_b = [Document(id="b"), Document(id="a")]
        result_equal = _reciprocal_rank_fusion([docs_a, docs_b])
        result_weighted = _reciprocal_rank_fusion([docs_a, docs_b], weights=[0.9, 0.1])
        equal_by_id = {doc.id: doc.score for doc in result_equal}
        weighted_by_id = {doc.id: doc.score for doc in result_weighted}
        # with equal weights both docs score the same; with heavy weight on list_a, "a" should outscore "b"
        assert equal_by_id["a"] == pytest.approx(equal_by_id["b"])
        assert weighted_by_id["a"] > weighted_by_id["b"]


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
