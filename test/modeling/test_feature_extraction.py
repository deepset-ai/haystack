import pytest
from unittest.mock import MagicMock
from unittest import mock
from pathlib import Path

import haystack
from haystack.errors import ModelingError
from haystack.modeling.model.feature_extraction import FeatureExtractor


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


@pytest.fixture()
def mock_autotokenizer(monkeypatch):
    monkeypatch.setattr(
        haystack.modeling.model.feature_extraction, "FEATURE_EXTRACTORS", {"mocked": MockedAutoTokenizer}
    )
    monkeypatch.setattr(haystack.modeling.model.feature_extraction, "AutoConfig", MockedAutoConfig)
    monkeypatch.setattr(haystack.modeling.model.feature_extraction, "AutoTokenizer", MockedAutoTokenizer)


@pytest.mark.unit
def test_get_tokenizer_from_HF():
    with mock.patch("haystack.modeling.model.feature_extraction.AutoConfig") as mocked_ac:
        from haystack.modeling.model.feature_extraction import FEATURE_EXTRACTORS

        FEATURE_EXTRACTORS["test"] = mock.MagicMock()
        FEATURE_EXTRACTORS["test"].__name__ = "Test"
        mocked_ac.from_pretrained.return_value.model_type = "test"
        FeatureExtractor(pretrained_model_name_or_path="test-model-name")
        FEATURE_EXTRACTORS["test"].from_pretrained.assert_called_with(
            pretrained_model_name_or_path="test-model-name", revision=None, use_fast=True, use_auth_token=None
        )
        # clean up
        FEATURE_EXTRACTORS.pop("test")


@pytest.mark.unit
def test_get_tokenizer_from_HF_not_found():
    with mock.patch("haystack.modeling.model.feature_extraction.AutoConfig") as mocked_ac:
        mocked_ac.from_pretrained.return_value.model_type = "does_not_exist"
        with pytest.raises(ModelingError):
            FeatureExtractor(pretrained_model_name_or_path="test-model-name")


@pytest.mark.unit
def test_get_tokenizer_from_path_fast():
    here = Path(__file__).resolve().parent
    mocked_model_folder = here / "samples/test_get_tokenizer_from_path"
    with mock.patch("haystack.modeling.model.feature_extraction.transformers") as mocked_tf:
        mocked_tf.TestTokenizerFast.__class__.__name__ = "Test Class"
        FeatureExtractor(pretrained_model_name_or_path=mocked_model_folder)
        mocked_tf.TestTokenizerFast.from_pretrained.assert_called_with(
            pretrained_model_name_or_path=str(mocked_model_folder), revision=None, use_fast=True, use_auth_token=None
        )


@pytest.mark.unit
def test_get_tokenizer_from_path():
    here = Path(__file__).resolve().parent
    mocked_model_folder = here / "samples/test_get_tokenizer_from_path"
    with mock.patch("haystack.modeling.model.feature_extraction.transformers") as mocked_tf:
        mocked_tf.TestTokenizer.__class__.__name__ = "Test Class"
        FeatureExtractor(pretrained_model_name_or_path=mocked_model_folder)
        mocked_tf.TestTokenizerFast.from_pretrained.assert_called_with(
            pretrained_model_name_or_path=str(mocked_model_folder), revision=None, use_fast=True, use_auth_token=None
        )


@pytest.mark.unit
def test_get_tokenizer_from_path_class_doesnt_exist():
    here = Path(__file__).resolve().parent
    mocked_model_folder = here / "samples/test_get_tokenizer_from_path"
    with pytest.raises(AttributeError, match="module transformers has no attribute TestTokenizer"):
        FeatureExtractor(pretrained_model_name_or_path=mocked_model_folder)


@pytest.mark.unit
def test_get_tokenizer_keep_accents():
    here = Path(__file__).resolve().parent
    mocked_model_folder = here / "samples/test_get_tokenizer_from_path"
    with mock.patch("haystack.modeling.model.feature_extraction.transformers") as mocked_tf:
        mocked_tf.TestTokenizer.__class__.__name__ = "Test Class"
        FeatureExtractor(pretrained_model_name_or_path=mocked_model_folder, keep_accents=True)
        mocked_tf.TestTokenizerFast.from_pretrained.assert_called_with(
            pretrained_model_name_or_path=str(mocked_model_folder),
            revision=None,
            use_fast=True,
            use_auth_token=None,
            keep_accents=True,
        )


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
