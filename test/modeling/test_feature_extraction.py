import pytest
from unittest.mock import MagicMock, patch

import haystack
from haystack.modeling.model.feature_extraction import FeatureExtractor, FEATURE_EXTRACTORS


class MockedAutoTokenizer:
    mocker: MagicMock = MagicMock()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.mocker.from_pretrained(*args, **kwargs)
        return cls()


class MockedAutoConfig:
    mocker: MagicMock = MagicMock()
    model_type: str = "mocked"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.mocker.from_pretrained(*args, **kwargs)
        return cls()


@pytest.fixture(autouse=True)
def mock_autotokenizer(request, monkeypatch):
    # Do not patch integration tests
    if "integration" in request.keywords:
        return
    monkeypatch.setattr(
        haystack.modeling.model.feature_extraction, "FEATURE_EXTRACTORS", {"mocked": MockedAutoTokenizer}
    )
    monkeypatch.setattr(haystack.modeling.model.feature_extraction, "AutoConfig", MockedAutoConfig)
    monkeypatch.setattr(haystack.modeling.model.feature_extraction, "AutoTokenizer", MockedAutoTokenizer)


#
# Unit tests
#


def test_init_str():
    tokenizer = FeatureExtractor(pretrained_model_name_or_path="test-model-name")

    tokenizer.feature_extractor.mocker.from_pretrained.assert_called_with(
        pretrained_model_name_or_path="test-model-name", revision=None, use_fast=True, use_auth_token=None
    )


def test_init_path(tmp_path):
    tokenizer = FeatureExtractor(pretrained_model_name_or_path=tmp_path / "test-path")

    tokenizer.feature_extractor.mocker.from_pretrained.assert_called_with(
        pretrained_model_name_or_path=str(tmp_path / "test-path"), revision=None, use_fast=True, use_auth_token=None
    )


#
# Integration tests
#

FEATURE_EXTRACTORS_TO_TEST = ["bert-base-cased"]


@pytest.mark.integration
@pytest.mark.parametrize("model_name", FEATURE_EXTRACTORS_TO_TEST)
def test_load_modify_save_load(tmp_path, model_name: str):
    # Load base tokenizer
    feature_extractor = FeatureExtractor(pretrained_model_name_or_path=model_name, do_lower_case=False)

    # Add new tokens
    feature_extractor.feature_extractor.add_tokens(new_tokens=["neverseentokens"])

    # Save modified tokenizer
    save_dir = tmp_path / "saved_tokenizer"
    feature_extractor.feature_extractor.save_pretrained(save_dir)

    # Load modified tokenizer
    new_feature_extractor = FeatureExtractor(pretrained_model_name_or_path=save_dir)

    # Assert the new tokenizer still has the added tokens
    assert len(new_feature_extractor.feature_extractor) == len(feature_extractor.feature_extractor)
