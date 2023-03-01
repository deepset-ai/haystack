import pytest
from unittest.mock import MagicMock

import haystack
from haystack.modeling.model.feature_extraction import FeatureExtractor


class AutoTokenizer:
    mocker: MagicMock = MagicMock()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.mocker.from_pretrained(*args, **kwargs)
        return cls()


@pytest.fixture(autouse=True)
def mock_autotokenizer(request, monkeypatch):
    monkeypatch.setattr(haystack.modeling.model.tokenization, "AutoTokenizer", AutoTokenizer)


@pytest.mark.unit
def test_get_tokenizer_str():
    tokenizer = FeatureExtractor(pretrained_model_name_or_path="test-model-name")
    tokenizer.mocker.from_pretrained.assert_called_with(
        pretrained_model_name_or_path="test-model-name", revision=None, use_fast=True, use_auth_token=None
    )


@pytest.mark.unit
def test_get_tokenizer_path(tmp_path):
    tokenizer = FeatureExtractor(pretrained_model_name_or_path=tmp_path / "test-path")
    tokenizer.mocker.from_pretrained.assert_called_with(
        pretrained_model_name_or_path=str(tmp_path / "test-path"), revision=None, use_fast=True, use_auth_token=None
    )


@pytest.mark.unit
def test_get_tokenizer_keep_accents():
    tokenizer = FeatureExtractor(pretrained_model_name_or_path="test-model-name-albert")
    tokenizer.mocker.from_pretrained.assert_called_with(
        pretrained_model_name_or_path="test-model-name-albert",
        revision=None,
        use_fast=True,
        use_auth_token=None,
        keep_accents=True,
    )


@pytest.mark.unit
def test_get_tokenizer_mlm_warning(caplog):
    tokenizer = FeatureExtractor(pretrained_model_name_or_path="test-model-name-mlm")
    tokenizer.mocker.from_pretrained.assert_called_with(
        pretrained_model_name_or_path="test-model-name-mlm", revision=None, use_fast=True, use_auth_token=None
    )
    assert "MLM part of codebert is currently not supported in Haystack".lower() in caplog.text.lower()
