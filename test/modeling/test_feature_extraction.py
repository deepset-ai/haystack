from typing import Tuple

import re

import pytest
import numpy as np
from unittest.mock import MagicMock

from tokenizers.pre_tokenizers import WhitespaceSplit

import haystack
from haystack.modeling.model.feature_extraction import FeatureExtractor


class MockedFromPretrained:
    mocker: MagicMock = MagicMock()

    def __getattr__(self, name: str):
        if name == "model_type":
            return "bert"
        return MagicMock()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.mocker.from_pretrained(*args, **kwargs)
        return cls()

    @classmethod
    def __call__(cls, *args, **kwargs):
        cls.mocker.__call__(*args, **kwargs)


class TestFeatureExtraction:
    @pytest.fixture(autouse=True)
    def mock_autotokenizer(self, request, monkeypatch):

        # Do not patch integration tests
        if "integration" in request.keywords:
            return

        monkeypatch.setattr(haystack.modeling.model.feature_extraction, "AutoTokenizer", MockedFromPretrained)
        monkeypatch.setattr(haystack.modeling.model.feature_extraction, "AutoConfig", MockedFromPretrained)

    def convert_offset_from_word_reference_to_text_reference(self, offsets, words, word_spans):
        """
        Token offsets are originally relative to the beginning of the word
        We make them relative to the beginning of the sentence.

        Not a fixture, just a utility.
        """
        token_offsets = []
        for ((start, end), word_index) in zip(offsets, words):
            word_start = word_spans[word_index][0]
            token_offsets.append((start + word_start, end + word_start))
        return token_offsets

    #
    # Unit tests
    #

    def test_init_str(self):
        tokenizer = FeatureExtractor(pretrained_model_name_or_path="test-model-name")
        tokenizer.mocker.from_pretrained.assert_called_with(
            pretrained_model_name_or_path="test-model-name", revision=None, use_fast=True, use_auth_token=None
        )

    def test_init_path(self, tmp_path):
        tokenizer = FeatureExtractor(pretrained_model_name_or_path=tmp_path / "test-path")
        tokenizer.mocker.from_pretrained.assert_called_with(
            pretrained_model_name_or_path=str(tmp_path / "test-path"), revision=None, use_fast=True, use_auth_token=None
        )

    #
    # Integration tests
    #

    FEATURE_EXTRACTORS = ["bert-base-cased"]

    @pytest.mark.integration
    @pytest.mark.parametrize("model_name", FEATURE_EXTRACTORS)
    def test_load_modify_save_load(self, tmp_path, model_name: str):

        # Load base tokenizer
        feature_extractor = FeatureExtractor(pretrained_model_name_or_path=model_name, do_lower_case=False)

        # Add new tokens
        feature_extractor.feature_extractor.add_tokens(new_tokens=["neverseentokens"])

        # Save modified tokenizer
        save_dir = tmp_path / "saved_tokenizer"
        feature_extractor.save_pretrained(save_dir)

        # Load modified tokenizer
        new_feature_extractor = FeatureExtractor(pretrained_model_name_or_path=save_dir)

        # Assert the new tokenizer still has the added tokens
        assert len(new_feature_extractor.feature_extractor) == len(feature_extractor.feature_extractor)
