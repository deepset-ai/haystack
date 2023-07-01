import os
from unittest.mock import patch, MagicMock, Mock

import pytest

from haystack.lazy_imports import LazyImport

from haystack.errors import SageMakerConfigurationError
from haystack.nodes.prompt.invocation_layer import SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer
from haystack.nodes.prompt.invocation_layer.sagemaker_jumpstart_ai21 import (
    SageMakerJumpStartAi21J2CompleteInferenceInvocationLayer,
)

with LazyImport() as boto3_import:
    from botocore.exceptions import BotoCoreError


# create a fixture with mocked boto3 client and session
@pytest.fixture
def mock_boto3_session():
    with patch("boto3.Session") as mock_client:
        yield mock_client


@pytest.fixture
def mock_prompt_handler():
    with patch("haystack.nodes.prompt.invocation_layer.handlers.DefaultPromptHandler") as mock_prompt_handler:
        yield mock_prompt_handler


@pytest.mark.unit
def test_default_constructor(mock_auto_tokenizer, mock_boto3_session):
    """
    Test that the default constructor sets the correct values
    """

    layer = SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer(
        model_name_or_path="some_fake_model",
        max_length=99,
        aws_access_key_id="some_fake_id",
        aws_secret_access_key="some_fake_key",
        aws_session_token="some_fake_token",
        aws_profile_name="some_fake_profile",
        aws_region_name="fake_region",
    )

    assert layer.max_length == 99
    assert layer.model_name_or_path == "some_fake_model"

    # assert mocked boto3 client called exactly once
    mock_boto3_session.assert_called_once()

    # assert mocked boto3 client was called with the correct parameters
    mock_boto3_session.assert_called_with(
        aws_access_key_id="some_fake_id",
        aws_secret_access_key="some_fake_key",
        aws_session_token="some_fake_token",
        profile_name="some_fake_profile",
        region_name="fake_region",
    )


@pytest.mark.unit
def test_constructor_with_model_kwargs(mock_auto_tokenizer, mock_boto3_session):
    """
    Test that model_kwargs are correctly set in the constructor
    and that model_kwargs_rejected are correctly filtered out
    """
    model_kwargs = {
        "context": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        "question": "What is the meaning of life?",
    }
    model_kwargs_rejected = {"fake_param": 0.7, "another_fake_param": 1}

    layer = SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer(
        model_name_or_path="some_fake_model", **model_kwargs, **model_kwargs_rejected
    )
    assert "context" in layer.model_input_kwargs
    assert "question" in layer.model_input_kwargs
    assert "fake_param" not in layer.model_input_kwargs
    assert "another_fake_param" not in layer.model_input_kwargs


@pytest.mark.unit
def test_invoke_with_no_question(mock_auto_tokenizer, mock_boto3_session):
    """
    Test that invoke raises an error if no prompt is provided
    """
    layer = SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer(model_name_or_path="some_fake_model")
    with pytest.raises(ValueError) as e:
        layer.invoke(
            context="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        )
        assert e.match("No question provided.")


@pytest.mark.unit
def test_invoke_with_no_context(mock_auto_tokenizer, mock_boto3_session):
    """
    Test that invoke raises an error if no prompt is provided
    """
    layer = SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer(model_name_or_path="some_fake_model")
    with pytest.raises(ValueError) as e:
        layer.invoke(question="What is the meaning of life?")
        assert e.match("No context provided.")


@pytest.mark.unit
def test_empty_model_name():
    with pytest.raises(ValueError, match="cannot be None or empty string"):
        SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer(model_name_or_path="")


@pytest.mark.unit
def test_streaming_init_kwarg(mock_auto_tokenizer, mock_boto3_session):
    """
    Test stream parameter passed as init kwarg is correctly logged as not supported
    """
    layer = SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer(
        model_name_or_path="irrelevant", stream=True
    )

    with pytest.raises(SageMakerConfigurationError, match="SageMaker model response streaming is not supported yet"):
        layer.invoke(question="Tell me hello", context="World")


@pytest.mark.unit
def test_streaming_invoke_kwarg(mock_auto_tokenizer, mock_boto3_session):
    """
    Test stream parameter passed as invoke kwarg is correctly logged as not supported
    """
    layer = SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer(model_name_or_path="irrelevant")

    with pytest.raises(SageMakerConfigurationError, match="SageMaker model response streaming is not supported yet"):
        layer.invoke(question="Tell me hello", context="World", stream=True)


@pytest.mark.unit
def test_streaming_handler_init_kwarg(mock_auto_tokenizer, mock_boto3_session):
    """
    Test stream_handler parameter passed as init kwarg is correctly logged as not supported
    """
    layer = SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer(
        model_name_or_path="irrelevant", stream_handler=Mock()
    )

    with pytest.raises(SageMakerConfigurationError, match="SageMaker model response streaming is not supported yet"):
        layer.invoke(question="Tell me hello", context="World")


@pytest.mark.unit
def test_streaming_handler_invoke_kwarg(mock_auto_tokenizer, mock_boto3_session):
    """
    Test stream_handler parameter passed as invoke kwarg is correctly logged as not supported
    """
    layer = SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer(model_name_or_path="irrelevant")

    with pytest.raises(SageMakerConfigurationError, match="SageMaker model response streaming is not supported yet"):
        layer.invoke(question="Tell me hello", context="World", stream_handler=Mock())


@pytest.mark.unit
def test_supports_for_valid_aws_configuration():
    """
    Test that the SageMakerInvocationLayer identifies a valid SageMaker Inference endpoint via the supports() method
    """
    with patch("boto3.Session") as mock_boto3_session:
        mock_boto3_session.return_value.client.return_value.invoke_endpoint.return_value = True
        supported = SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer.supports(
            model_name_or_path="some_sagemaker_deployed_model", aws_profile_name="some_real_profile"
        )
    assert supported
    assert mock_boto3_session.called
    _, called_kwargs = mock_boto3_session.call_args
    assert called_kwargs["profile_name"] == "some_real_profile"


@pytest.mark.unit
def test_supports_not_on_invalid_aws_profile_name():
    """
    Test that the SageMakerInvocationLayer raises SageMakerConfigurationError when the profile name is invalid
    """

    with patch("boto3.Session") as mock_boto3_session:
        mock_boto3_session.side_effect = BotoCoreError()
        with pytest.raises(SageMakerConfigurationError) as exc_info:
            supported = SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer.supports(
                model_name_or_path="some_fake_model", aws_profile_name="some_fake_profile"
            )
            assert "Failed to initialize the session" in exc_info.value
            assert not supported


@pytest.mark.skipif(
    not os.environ.get("TEST_SAGEMAKER_MODEL_ENDPOINT", None), reason="Skipping because SageMaker not configured"
)
@pytest.mark.integration
def test_supports_triggered_for_valid_sagemaker_endpoint():
    """
    Test that the SageMakerInvocationLayer identifies a valid SageMaker Inference endpoint via the supports() method
    """
    model_name_or_path = os.environ.get("TEST_SAGEMAKER_MODEL_ENDPOINT")
    assert SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer.supports(
        model_name_or_path=model_name_or_path
    )


@pytest.mark.skipif(
    not os.environ.get("TEST_SAGEMAKER_MODEL_ENDPOINT", None), reason="Skipping because SageMaker not configured"
)
@pytest.mark.integration
def test_supports_not_triggered_for_invalid_iam_profile():
    """
    Test that the SageMakerInvocationLayer identifies an invalid SageMaker Inference endpoint
    (in this case because of an invalid IAM AWS Profile via the supports() method)
    """
    assert not SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer.supports(
        model_name_or_path="fake_endpoint"
    )
    assert not SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer.supports(
        model_name_or_path="fake_endpoint", aws_profile_name="invalid-profile"
    )


@pytest.mark.unit
def test_jurassic_2_complete_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    # this is AI21 Jurassic 2 Complete json response
    response = {
        "id": 1234,
        "prompt": {
            "text": "Write an engaging product description for a clothing eCommerce site. Make sure to include the following features in the description.\nProduct: Humor Men's Graphic T-Shirt.\nFeatures:\n- Soft cotton\n- Short sleeve\n- Have a print of Einstein's quote: \"artificial intelligence is no match for natural stupidity”\nDescription:\n",
            "tokens": [
                {
                    "generatedToken": {
                        "token": "▁Write",
                        "logprob": -8.940605163574219,
                        "raw_logprob": -8.940605163574219,
                    },
                    "topTokens": None,
                    "textRange": {"start": 0, "end": 5},
                },
                {
                    "generatedToken": {
                        "token": "▁an▁engaging",
                        "logprob": -9.080601692199707,
                        "raw_logprob": -9.080601692199707,
                    },
                    "topTokens": None,
                    "textRange": {"start": 5, "end": 17},
                },
                {
                    "generatedToken": {
                        "token": "▁product▁description",
                        "logprob": -5.254765510559082,
                        "raw_logprob": -5.254765510559082,
                    },
                    "topTokens": None,
                    "textRange": {"start": 17, "end": 37},
                },
                {
                    "generatedToken": {
                        "token": "▁for▁a",
                        "logprob": -8.486310005187988,
                        "raw_logprob": -8.486310005187988,
                    },
                    "topTokens": None,
                    "textRange": {"start": 37, "end": 43},
                },
                {
                    "generatedToken": {
                        "token": "▁clothing",
                        "logprob": -4.247067451477051,
                        "raw_logprob": -4.247067451477051,
                    },
                    "topTokens": None,
                    "textRange": {"start": 43, "end": 52},
                },
                {
                    "generatedToken": {
                        "token": "▁eCommerce",
                        "logprob": -11.935654640197754,
                        "raw_logprob": -11.935654640197754,
                    },
                    "topTokens": None,
                    "textRange": {"start": 52, "end": 62},
                },
                {
                    "generatedToken": {
                        "token": "▁site",
                        "logprob": -2.5150115489959717,
                        "raw_logprob": -2.5150115489959717,
                    },
                    "topTokens": None,
                    "textRange": {"start": 62, "end": 67},
                },
                {
                    "generatedToken": {
                        "token": ".",
                        "logprob": -13.890634536743164,
                        "raw_logprob": -13.890634536743164,
                    },
                    "topTokens": None,
                    "textRange": {"start": 67, "end": 68},
                },
                {
                    "generatedToken": {"token": "▁Make▁sure▁to", "logprob": -21.25, "raw_logprob": -21.25},
                    "topTokens": None,
                    "textRange": {"start": 68, "end": 81},
                },
                {
                    "generatedToken": {
                        "token": "▁include",
                        "logprob": -0.2799517512321472,
                        "raw_logprob": -0.2799517512321472,
                    },
                    "topTokens": None,
                    "textRange": {"start": 81, "end": 89},
                },
                {
                    "generatedToken": {
                        "token": "▁the▁following",
                        "logprob": -3.852478265762329,
                        "raw_logprob": -3.852478265762329,
                    },
                    "topTokens": None,
                    "textRange": {"start": 89, "end": 103},
                },
                {
                    "generatedToken": {
                        "token": "▁features",
                        "logprob": -2.2079620361328125,
                        "raw_logprob": -2.2079620361328125,
                    },
                    "topTokens": None,
                    "textRange": {"start": 103, "end": 112},
                },
                {
                    "generatedToken": {
                        "token": "▁in▁the",
                        "logprob": -9.997041702270508,
                        "raw_logprob": -9.997041702270508,
                    },
                    "topTokens": None,
                    "textRange": {"start": 112, "end": 119},
                },
                {
                    "generatedToken": {
                        "token": "▁description",
                        "logprob": -0.10494892299175262,
                        "raw_logprob": -0.10494892299175262,
                    },
                    "topTokens": None,
                    "textRange": {"start": 119, "end": 131},
                },
                {
                    "generatedToken": {"token": ".", "logprob": -6.164520740509033, "raw_logprob": -6.164520740509033},
                    "topTokens": None,
                    "textRange": {"start": 131, "end": 132},
                },
                {
                    "generatedToken": {
                        "token": "<|newline|>",
                        "logprob": -1.5497195136049413e-06,
                        "raw_logprob": -1.5497195136049413e-06,
                    },
                    "topTokens": None,
                    "textRange": {"start": 132, "end": 133},
                },
                {
                    "generatedToken": {
                        "token": "▁Product",
                        "logprob": -4.678016662597656,
                        "raw_logprob": -4.678016662597656,
                    },
                    "topTokens": None,
                    "textRange": {"start": 133, "end": 140},
                },
                {
                    "generatedToken": {"token": ":", "logprob": -3.368314266204834, "raw_logprob": -3.368314266204834},
                    "topTokens": None,
                    "textRange": {"start": 140, "end": 141},
                },
                {
                    "generatedToken": {
                        "token": "▁Humor",
                        "logprob": -18.138029098510742,
                        "raw_logprob": -18.138029098510742,
                    },
                    "topTokens": None,
                    "textRange": {"start": 141, "end": 147},
                },
                {
                    "generatedToken": {
                        "token": "▁Men's",
                        "logprob": -10.727008819580078,
                        "raw_logprob": -10.727008819580078,
                    },
                    "topTokens": None,
                    "textRange": {"start": 147, "end": 153},
                },
                {
                    "generatedToken": {
                        "token": "▁Graphic",
                        "logprob": -4.4896979331970215,
                        "raw_logprob": -4.4896979331970215,
                    },
                    "topTokens": None,
                    "textRange": {"start": 153, "end": 161},
                },
                {
                    "generatedToken": {
                        "token": "▁T-Shirt",
                        "logprob": -0.2305745631456375,
                        "raw_logprob": -0.2305745631456375,
                    },
                    "topTokens": None,
                    "textRange": {"start": 161, "end": 169},
                },
                {
                    "generatedToken": {"token": ".", "logprob": -6.631472587585449, "raw_logprob": -6.631472587585449},
                    "topTokens": None,
                    "textRange": {"start": 169, "end": 170},
                },
                {
                    "generatedToken": {
                        "token": "<|newline|>",
                        "logprob": -0.0011917401570826769,
                        "raw_logprob": -0.0011917401570826769,
                    },
                    "topTokens": None,
                    "textRange": {"start": 170, "end": 171},
                },
                {
                    "generatedToken": {
                        "token": "▁Features",
                        "logprob": -4.403321743011475,
                        "raw_logprob": -4.403321743011475,
                    },
                    "topTokens": None,
                    "textRange": {"start": 171, "end": 179},
                },
                {
                    "generatedToken": {
                        "token": ":",
                        "logprob": -0.0006685405969619751,
                        "raw_logprob": -0.0006685405969619751,
                    },
                    "topTokens": None,
                    "textRange": {"start": 179, "end": 180},
                },
                {
                    "generatedToken": {
                        "token": "<|newline|>",
                        "logprob": -0.005023120902478695,
                        "raw_logprob": -0.005023120902478695,
                    },
                    "topTokens": None,
                    "textRange": {"start": 180, "end": 181},
                },
                {
                    "generatedToken": {
                        "token": "▁-",
                        "logprob": -0.5957688093185425,
                        "raw_logprob": -0.5957688093185425,
                    },
                    "topTokens": None,
                    "textRange": {"start": 181, "end": 182},
                },
                {
                    "generatedToken": {
                        "token": "▁Soft",
                        "logprob": -3.3877720832824707,
                        "raw_logprob": -3.3877720832824707,
                    },
                    "topTokens": None,
                    "textRange": {"start": 182, "end": 187},
                },
                {
                    "generatedToken": {
                        "token": "▁cotton",
                        "logprob": -1.6477241516113281,
                        "raw_logprob": -1.6477241516113281,
                    },
                    "topTokens": None,
                    "textRange": {"start": 187, "end": 194},
                },
                {
                    "generatedToken": {
                        "token": "<|newline|>",
                        "logprob": -6.65597677230835,
                        "raw_logprob": -6.65597677230835,
                    },
                    "topTokens": None,
                    "textRange": {"start": 194, "end": 195},
                },
                {
                    "generatedToken": {
                        "token": "▁-",
                        "logprob": -8.725739462533966e-05,
                        "raw_logprob": -8.725739462533966e-05,
                    },
                    "topTokens": None,
                    "textRange": {"start": 195, "end": 196},
                },
                {
                    "generatedToken": {
                        "token": "▁Short",
                        "logprob": -3.7746338844299316,
                        "raw_logprob": -3.7746338844299316,
                    },
                    "topTokens": None,
                    "textRange": {"start": 196, "end": 202},
                },
                {
                    "generatedToken": {
                        "token": "▁sleeve",
                        "logprob": -2.0452399253845215,
                        "raw_logprob": -2.0452399253845215,
                    },
                    "topTokens": None,
                    "textRange": {"start": 202, "end": 209},
                },
                {
                    "generatedToken": {
                        "token": "<|newline|>",
                        "logprob": -0.007816560566425323,
                        "raw_logprob": -0.007816560566425323,
                    },
                    "topTokens": None,
                    "textRange": {"start": 209, "end": 210},
                },
                {
                    "generatedToken": {
                        "token": "▁-",
                        "logprob": -0.004475695546716452,
                        "raw_logprob": -0.004475695546716452,
                    },
                    "topTokens": None,
                    "textRange": {"start": 210, "end": 211},
                },
                {
                    "generatedToken": {
                        "token": "▁Have▁a",
                        "logprob": -17.35095977783203,
                        "raw_logprob": -17.35095977783203,
                    },
                    "topTokens": None,
                    "textRange": {"start": 211, "end": 218},
                },
                {
                    "generatedToken": {
                        "token": "▁print",
                        "logprob": -10.28792667388916,
                        "raw_logprob": -10.28792667388916,
                    },
                    "topTokens": None,
                    "textRange": {"start": 218, "end": 224},
                },
                {
                    "generatedToken": {
                        "token": "▁of",
                        "logprob": -2.436065196990967,
                        "raw_logprob": -2.436065196990967,
                    },
                    "topTokens": None,
                    "textRange": {"start": 224, "end": 227},
                },
                {
                    "generatedToken": {
                        "token": "▁Einstein",
                        "logprob": -7.763306617736816,
                        "raw_logprob": -7.763306617736816,
                    },
                    "topTokens": None,
                    "textRange": {"start": 227, "end": 236},
                },
                {
                    "generatedToken": {
                        "token": "'s",
                        "logprob": -2.1889724731445312,
                        "raw_logprob": -2.1889724731445312,
                    },
                    "topTokens": None,
                    "textRange": {"start": 236, "end": 238},
                },
                {
                    "generatedToken": {
                        "token": "▁quote",
                        "logprob": -4.9218339920043945,
                        "raw_logprob": -4.9218339920043945,
                    },
                    "topTokens": None,
                    "textRange": {"start": 238, "end": 244},
                },
                {
                    "generatedToken": {
                        "token": ":",
                        "logprob": -2.6913697719573975,
                        "raw_logprob": -2.6913697719573975,
                    },
                    "topTokens": None,
                    "textRange": {"start": 244, "end": 245},
                },
                {
                    "generatedToken": {
                        "token": '▁"',
                        "logprob": -0.06187326833605766,
                        "raw_logprob": -0.06187326833605766,
                    },
                    "topTokens": None,
                    "textRange": {"start": 245, "end": 247},
                },
                {
                    "generatedToken": {
                        "token": "artificial",
                        "logprob": -15.582560539245605,
                        "raw_logprob": -15.582560539245605,
                    },
                    "topTokens": None,
                    "textRange": {"start": 247, "end": 257},
                },
                {
                    "generatedToken": {
                        "token": "▁intelligence",
                        "logprob": -0.03362813591957092,
                        "raw_logprob": -0.03362813591957092,
                    },
                    "topTokens": None,
                    "textRange": {"start": 257, "end": 270},
                },
                {
                    "generatedToken": {
                        "token": "▁is",
                        "logprob": -0.1476401835680008,
                        "raw_logprob": -0.1476401835680008,
                    },
                    "topTokens": None,
                    "textRange": {"start": 270, "end": 273},
                },
                {
                    "generatedToken": {
                        "token": "▁no▁match▁for",
                        "logprob": -0.15623262524604797,
                        "raw_logprob": -0.15623262524604797,
                    },
                    "topTokens": None,
                    "textRange": {"start": 273, "end": 286},
                },
                {
                    "generatedToken": {
                        "token": "▁natural",
                        "logprob": -0.0831436812877655,
                        "raw_logprob": -0.0831436812877655,
                    },
                    "topTokens": None,
                    "textRange": {"start": 286, "end": 294},
                },
                {
                    "generatedToken": {
                        "token": "▁stupidity",
                        "logprob": -0.0007270314963534474,
                        "raw_logprob": -0.0007270314963534474,
                    },
                    "topTokens": None,
                    "textRange": {"start": 294, "end": 304},
                },
                {
                    "generatedToken": {
                        "token": "”",
                        "logprob": -11.650083541870117,
                        "raw_logprob": -11.650083541870117,
                    },
                    "topTokens": None,
                    "textRange": {"start": 304, "end": 305},
                },
                {
                    "generatedToken": {
                        "token": "<|newline|>",
                        "logprob": -0.03856721892952919,
                        "raw_logprob": -0.03856721892952919,
                    },
                    "topTokens": None,
                    "textRange": {"start": 305, "end": 306},
                },
                {
                    "generatedToken": {
                        "token": "▁Description",
                        "logprob": -7.664284706115723,
                        "raw_logprob": -7.664284706115723,
                    },
                    "topTokens": None,
                    "textRange": {"start": 306, "end": 317},
                },
                {
                    "generatedToken": {
                        "token": ":",
                        "logprob": -0.0003323002893012017,
                        "raw_logprob": -0.0003323002893012017,
                    },
                    "topTokens": None,
                    "textRange": {"start": 317, "end": 318},
                },
                {
                    "generatedToken": {
                        "token": "<|newline|>",
                        "logprob": -0.0007071378640830517,
                        "raw_logprob": -0.0007071378640830517,
                    },
                    "topTokens": None,
                    "textRange": {"start": 318, "end": 319},
                },
            ],
        },
        "completions": [
            {
                "data": {
                    "text": "This funny t-shirt is perfect for anyone who enjoys a good laugh. The print features Albert Einstein's famous quote about",
                    "tokens": [
                        {
                            "generatedToken": {
                                "token": "▁This",
                                "logprob": -0.026408543810248375,
                                "raw_logprob": -0.2072315514087677,
                            },
                            "topTokens": None,
                            "textRange": {"start": 0, "end": 4},
                        },
                        {
                            "generatedToken": {
                                "token": "▁funny",
                                "logprob": -0.4105142056941986,
                                "raw_logprob": -0.7374352812767029,
                            },
                            "topTokens": None,
                            "textRange": {"start": 4, "end": 10},
                        },
                        {
                            "generatedToken": {
                                "token": "▁t-shirt",
                                "logprob": -0.6925970911979675,
                                "raw_logprob": -0.9299893379211426,
                            },
                            "topTokens": None,
                            "textRange": {"start": 10, "end": 18},
                        },
                        {
                            "generatedToken": {
                                "token": "▁is▁perfect▁for",
                                "logprob": -0.012538178823888302,
                                "raw_logprob": -0.08875184506177902,
                            },
                            "topTokens": None,
                            "textRange": {"start": 18, "end": 33},
                        },
                        {
                            "generatedToken": {
                                "token": "▁anyone▁who",
                                "logprob": -0.8686554431915283,
                                "raw_logprob": -1.3148537874221802,
                            },
                            "topTokens": None,
                            "textRange": {"start": 33, "end": 44},
                        },
                        {
                            "generatedToken": {
                                "token": "▁enjoys",
                                "logprob": -0.6595482230186462,
                                "raw_logprob": -1.060908555984497,
                            },
                            "topTokens": None,
                            "textRange": {"start": 44, "end": 51},
                        },
                        {
                            "generatedToken": {
                                "token": "▁a▁good▁laugh",
                                "logprob": -1.0522539615631104,
                                "raw_logprob": -1.206431269645691,
                            },
                            "topTokens": None,
                            "textRange": {"start": 51, "end": 64},
                        },
                        {
                            "generatedToken": {
                                "token": ".",
                                "logprob": -0.006621918175369501,
                                "raw_logprob": -0.0369347482919693,
                            },
                            "topTokens": None,
                            "textRange": {"start": 64, "end": 65},
                        },
                        {
                            "generatedToken": {
                                "token": "▁The",
                                "logprob": -0.026142634451389313,
                                "raw_logprob": -0.12252024561166763,
                            },
                            "topTokens": None,
                            "textRange": {"start": 65, "end": 69},
                        },
                        {
                            "generatedToken": {
                                "token": "▁print",
                                "logprob": -0.5334348678588867,
                                "raw_logprob": -0.8310137391090393,
                            },
                            "topTokens": None,
                            "textRange": {"start": 69, "end": 75},
                        },
                        {
                            "generatedToken": {
                                "token": "▁features",
                                "logprob": -0.17499904334545135,
                                "raw_logprob": -0.41624805331230164,
                            },
                            "topTokens": None,
                            "textRange": {"start": 75, "end": 84},
                        },
                        {
                            "generatedToken": {
                                "token": "▁Albert▁Einstein",
                                "logprob": -0.5906462073326111,
                                "raw_logprob": -0.8075539469718933,
                            },
                            "topTokens": None,
                            "textRange": {"start": 84, "end": 100},
                        },
                        {
                            "generatedToken": {
                                "token": "'s",
                                "logprob": -0.00022218143567442894,
                                "raw_logprob": -0.004031626507639885,
                            },
                            "topTokens": None,
                            "textRange": {"start": 100, "end": 102},
                        },
                        {
                            "generatedToken": {
                                "token": "▁famous",
                                "logprob": -0.014182372018694878,
                                "raw_logprob": -0.07163470983505249,
                            },
                            "topTokens": None,
                            "textRange": {"start": 102, "end": 109},
                        },
                        {
                            "generatedToken": {
                                "token": "▁quote",
                                "logprob": -8.964136941358447e-05,
                                "raw_logprob": -0.002941450336948037,
                            },
                            "topTokens": None,
                            "textRange": {"start": 109, "end": 115},
                        },
                        {
                            "generatedToken": {
                                "token": "▁about",
                                "logprob": -0.2814832031726837,
                                "raw_logprob": -0.4626731872558594,
                            },
                            "topTokens": None,
                            "textRange": {"start": 115, "end": 121},
                        },
                    ],
                },
                "finishReason": {"reason": "length", "length": 16},
            }
        ],
    }
    layer = SageMakerJumpStartAi21J2CompleteInferenceInvocationLayer(model_name_or_path="irrelevant")
    assert layer._extract_response(response) == [
        "This funny t-shirt is perfect for anyone who enjoys a good laugh. The print features Albert Einstein's famous quote about"
    ]


@pytest.mark.unit
def test_jurassic_2_contextual_answer_response_parsing(mock_auto_tokenizer, mock_boto3_session):
    # this is AI21 Jurassic 2 Contextual Answer json response
    response = {"answer": "Berlin is the capital of Germany"}
    layer = SageMakerJumpStartAi21ContextualAnswersInferenceInvocationLayer(model_name_or_path="irrelevant")
    assert layer._extract_response(response) == ["Berlin is the capital of Germany"]
