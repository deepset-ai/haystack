from typing import List
from unittest.mock import MagicMock, patch, Mock
import logging

import pytest
import torch
from torch import device
from transformers import AutoTokenizer, BloomForCausalLM, StoppingCriteriaList, GenerationConfig

from haystack.nodes.prompt.invocation_layer import HFLocalInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import HFTokenStreamingHandler, DefaultTokenStreamingHandler
from haystack.nodes.prompt.invocation_layer.hugging_face import StopWordsCriteria


@pytest.fixture
def mock_pipeline():
    # mock transformers pipeline
    # model returning some mocked text for pipeline invocation
    with patch("haystack.nodes.prompt.invocation_layer.hugging_face.pipeline") as mocked_pipeline:
        pipeline_mock = Mock(**{"model_name_or_path": None, "tokenizer.model_max_length": 100})
        pipeline_mock.side_effect = lambda *args, **kwargs: [{"generated_text": "some mocked text"}]
        mocked_pipeline.return_value = pipeline_mock
        yield mocked_pipeline


@pytest.fixture
def mock_get_task():
    # mock get_task function
    with patch("haystack.nodes.prompt.invocation_layer.hugging_face.get_task") as mock_get_task:
        mock_get_task.return_value = "text2text-generation"
        yield mock_get_task


@pytest.mark.unit
def test_constructor_with_invalid_task_name(mock_pipeline, mock_get_task):
    """
    Test HFLocalInvocationLayer init with invalid task_name
    """
    with pytest.raises(ValueError, match="Task name custom-text2text-generation is not supported"):
        HFLocalInvocationLayer("google/flan-t5-base", task_name="custom-text2text-generation")


@pytest.mark.unit
def test_constructor_with_model_name_only(mock_pipeline, mock_get_task):
    """
    Test HFLocalInvocationLayer init with model_name_or_path only
    """
    HFLocalInvocationLayer("google/flan-t5-base")

    mock_pipeline.assert_called_once()

    _, kwargs = mock_pipeline.call_args

    # device is set to cpu by default and device_map is empty
    assert kwargs["device"] == device("cpu")
    assert not kwargs["device_map"]

    # correct task and model are set
    assert kwargs["task"] == "text2text-generation"
    assert kwargs["model"] == "google/flan-t5-base"

    # no matter what kwargs we pass or don't pass, there are always 14 predefined kwargs passed to the pipeline
    assert len(kwargs) == 14

    # and these kwargs are passed to the pipeline
    assert list(kwargs.keys()) == [
        "task",
        "model",
        "config",
        "tokenizer",
        "feature_extractor",
        "device_map",
        "device",
        "torch_dtype",
        "model_kwargs",
        "pipeline_class",
        "use_fast",
        "revision",
        "use_auth_token",
        "trust_remote_code",
    ]


@pytest.mark.unit
def test_constructor_with_model_name_and_device_map(mock_pipeline, mock_get_task):
    """
    Test HFLocalInvocationLayer init with model_name_or_path and device_map
    """

    layer = HFLocalInvocationLayer("google/flan-t5-base", device="cpu", device_map="auto")

    assert layer.pipe == mock_pipeline.return_value
    mock_pipeline.assert_called_once()
    mock_get_task.assert_called_once()

    _, kwargs = mock_pipeline.call_args

    # device is NOT set; device_map is auto because device_map takes precedence over device
    assert not kwargs["device"]
    assert kwargs["device_map"] and kwargs["device_map"] == "auto"

    # correct task and model are set as well
    assert kwargs["task"] == "text2text-generation"
    assert kwargs["model"] == "google/flan-t5-base"


@pytest.mark.unit
def test_constructor_with_torch_dtype(mock_pipeline, mock_get_task):
    """
    Test HFLocalInvocationLayer init with torch_dtype parameter using the actual torch object
    """

    layer = HFLocalInvocationLayer("google/flan-t5-base", torch_dtype=torch.float16)

    assert layer.pipe == mock_pipeline.return_value
    mock_pipeline.assert_called_once()
    mock_get_task.assert_called_once()

    _, kwargs = mock_pipeline.call_args
    assert kwargs["torch_dtype"] == torch.float16


@pytest.mark.unit
def test_constructor_with_torch_dtype_as_str(mock_pipeline, mock_get_task):
    """
    Test HFLocalInvocationLayer init with torch_dtype parameter using the string definition
    """

    layer = HFLocalInvocationLayer("google/flan-t5-base", torch_dtype="torch.float16")

    assert layer.pipe == mock_pipeline.return_value
    mock_pipeline.assert_called_once()
    mock_get_task.assert_called_once()

    _, kwargs = mock_pipeline.call_args
    assert kwargs["torch_dtype"] == torch.float16


@pytest.mark.unit
def test_constructor_with_torch_dtype_auto(mock_pipeline, mock_get_task):
    """
    Test HFLocalInvocationLayer init with torch_dtype parameter using the auto string definition
    """

    layer = HFLocalInvocationLayer("google/flan-t5-base", torch_dtype="auto")

    assert layer.pipe == mock_pipeline.return_value
    mock_pipeline.assert_called_once()
    mock_get_task.assert_called_once()

    _, kwargs = mock_pipeline.call_args
    assert kwargs["torch_dtype"] == "auto"


@pytest.mark.unit
def test_constructor_with_invalid_torch_dtype(mock_pipeline, mock_get_task):
    """
    Test HFLocalInvocationLayer init with invalid torch_dtype parameter
    """

    # we need to provide torch_dtype as a string but with torch. prefix
    # this should raise an error
    with pytest.raises(ValueError, match="torch_dtype should be a torch.dtype, a string with 'torch.' prefix"):
        HFLocalInvocationLayer("google/flan-t5-base", torch_dtype="float16")


@pytest.mark.unit
def test_constructor_with_invalid_torch_dtype_object(mock_pipeline, mock_get_task):
    """
    Test HFLocalInvocationLayer init with invalid parameter
    """

    # we need to provide torch_dtype as a string but with torch. prefix
    # this should raise an error
    with pytest.raises(ValueError, match="Invalid torch_dtype value {'invalid': 'object'}"):
        HFLocalInvocationLayer("google/flan-t5-base", torch_dtype={"invalid": "object"})


@pytest.mark.integration
def test_ensure_token_limit_positive():
    """
    Test that ensure_token_limit works as expected, short prompt text is not changed
    """
    prompt_text = "this is a short prompt"
    layer = HFLocalInvocationLayer("google/flan-t5-base", max_length=10, model_max_length=20)

    processed_prompt_text = layer._ensure_token_limit(prompt_text)
    assert prompt_text == processed_prompt_text


@pytest.mark.integration
def test_ensure_token_limit_negative(caplog):
    """
    Test that ensure_token_limit chops the prompt text if it's longer than the max length allowed by the model
    """
    prompt_text = "this is a prompt test that is longer than the max length allowed by the model"
    layer = HFLocalInvocationLayer("google/flan-t5-base", max_length=10, model_max_length=20)

    processed_prompt_text = layer._ensure_token_limit(prompt_text)
    assert prompt_text != processed_prompt_text
    assert len(processed_prompt_text.split()) <= len(prompt_text.split())
    expected_message = (
        "The prompt has been truncated from 17 tokens to 10 tokens so that the prompt length and "
        "answer length (10 tokens) fit within the max token limit (20 tokens). Shorten the prompt "
        "to prevent it from being cut off"
    )
    assert caplog.records[0].message == expected_message


@pytest.mark.unit
def test_constructor_with_custom_pretrained_model(mock_pipeline, mock_get_task):
    """
    Test that the constructor sets the pipeline with the pretrained model (if provided)
    """
    model = Mock()
    tokenizer = Mock()

    HFLocalInvocationLayer(
        model_name_or_path="irrelevant_when_model_is_provided",
        model=model,
        tokenizer=tokenizer,
        task_name="text2text-generation",
    )

    mock_pipeline.assert_called_once()
    # mock_get_task is not called as we provided task_name parameter
    mock_get_task.assert_not_called()

    _, kwargs = mock_pipeline.call_args

    # correct tokenizer and model are set as well
    assert kwargs["tokenizer"] == tokenizer
    assert kwargs["model"] == model


@pytest.mark.unit
def test_constructor_with_invalid_kwargs(mock_pipeline, mock_get_task):
    """
    Test HFLocalInvocationLayer init with invalid kwargs
    """

    HFLocalInvocationLayer("google/flan-t5-base", some_invalid_kwarg="invalid")

    mock_pipeline.assert_called_once()
    mock_get_task.assert_called_once()

    _, kwargs = mock_pipeline.call_args

    # invalid kwargs are ignored and not passed to the pipeline
    assert "some_invalid_kwarg" not in kwargs

    # still our 14 kwargs passed to the pipeline
    assert len(kwargs) == 14


@pytest.mark.unit
def test_constructor_with_various_kwargs(mock_pipeline, mock_get_task):
    """
    Test HFLocalInvocationLayer init with various kwargs, make sure all of them are passed to the pipeline
    except for the invalid ones
    """

    HFLocalInvocationLayer(
        "google/flan-t5-base",
        task_name="text2text-generation",
        tokenizer=Mock(),
        config=Mock(),
        revision="1.1",
        device="cpu",
        device_map="auto",
        first_invalid_kwarg="invalid",
        second_invalid_kwarg="invalid",
    )

    mock_pipeline.assert_called_once()
    # mock_get_task is not called as we provided task_name parameter
    mock_get_task.assert_not_called()

    _, kwargs = mock_pipeline.call_args

    # invalid kwargs are ignored and not passed to the pipeline
    assert "first_invalid_kwarg" not in kwargs
    assert "second_invalid_kwarg" not in kwargs

    # correct task and model are set as well
    assert kwargs["task"] == "text2text-generation"
    assert not kwargs["device"]
    assert kwargs["device_map"] and kwargs["device_map"] == "auto"
    assert kwargs["revision"] == "1.1"

    # still on 14 kwargs passed to the pipeline
    assert len(kwargs) == 14


@pytest.mark.integration
def test_text_generation_model():
    # test simple prompting with text generation model
    # by default, we force the model not return prompt text
    # Thus text-generation models can be used with PromptNode
    # just like text2text-generation models
    layer = HFLocalInvocationLayer("bigscience/bigscience-small-testing")
    r = layer.invoke(prompt="Hello big science!")
    assert len(r[0]) > 0

    # test prompting with parameter to return prompt text as well
    # users can use this param to get the prompt text and the generated text
    r = layer.invoke(prompt="Hello big science!", return_full_text=True)
    assert len(r[0]) > 0 and r[0].startswith("Hello big science!")


@pytest.mark.integration
def test_text_generation_model_via_custom_pretrained_model():
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bigscience-small-testing")
    model = BloomForCausalLM.from_pretrained("bigscience/bigscience-small-testing")
    layer = HFLocalInvocationLayer(
        "irrelevant_when_model_is_provided", model=model, tokenizer=tokenizer, task_name="text-generation"
    )
    r = layer.invoke(prompt="Hello big science")
    assert len(r[0]) > 0

    # test prompting with parameter to return prompt text as well
    # users can use this param to get the prompt text and the generated text
    r = layer.invoke(prompt="Hello big science", return_full_text=True)
    assert len(r[0]) > 0 and r[0].startswith("Hello big science")


@pytest.mark.unit
def test_streaming_stream_param_in_constructor(mock_pipeline, mock_get_task):
    """
    Test stream parameter is correctly passed to pipeline invocation via HF streamer parameter
    """
    layer = HFLocalInvocationLayer(stream=True)

    layer.invoke(prompt="Tell me hello")

    _, kwargs = layer.pipe.call_args
    assert "streamer" in kwargs and isinstance(kwargs["streamer"], HFTokenStreamingHandler)


@pytest.mark.unit
def test_streaming_stream_handler_param_in_constructor(mock_pipeline, mock_get_task):
    """
    Test stream parameter is correctly passed to pipeline invocation
    """
    dtsh = DefaultTokenStreamingHandler()
    layer = HFLocalInvocationLayer(stream_handler=dtsh)

    layer.invoke(prompt="Tell me hello")

    _, kwargs = layer.pipe.call_args
    assert "streamer" in kwargs
    hf_streamer = kwargs["streamer"]

    # we wrap our TokenStreamingHandler with HFTokenStreamingHandler
    assert isinstance(hf_streamer, HFTokenStreamingHandler)

    # but under the hood, the wrapped handler is DefaultTokenStreamingHandler we passed
    assert isinstance(hf_streamer.token_handler, DefaultTokenStreamingHandler)
    assert hf_streamer.token_handler == dtsh


@pytest.mark.unit
def test_supports(tmp_path, mock_get_task):
    """
    Test that supports returns True correctly for HFLocalInvocationLayer
    """

    assert HFLocalInvocationLayer.supports("google/flan-t5-base")
    assert HFLocalInvocationLayer.supports("mosaicml/mpt-7b")
    assert HFLocalInvocationLayer.supports("CarperAI/stable-vicuna-13b-delta")
    mock_get_task.side_effect = RuntimeError
    assert not HFLocalInvocationLayer.supports("google/flan-t5-base")
    assert mock_get_task.call_count == 4

    # some HF local model directory, let's use the one from test/prompt/invocation_layer
    assert HFLocalInvocationLayer.supports(str(tmp_path))

    # we can also specify the task name to override the default
    # short-circuit the get_task call
    assert HFLocalInvocationLayer.supports(
        "vblagoje/bert-english-uncased-finetuned-pos", task_name="text2text-generation"
    )


@pytest.mark.unit
def test_supports_not(mock_get_task):
    """
    Test that supports returns False correctly for HFLocalInvocationLayer
    """
    assert not HFLocalInvocationLayer.supports("google/flan-t5-base", api_key="some_key")

    # also not some non text2text-generation or non text-generation model
    # i.e image classification model
    mock_get_task = Mock(return_value="image-classification")
    with patch("haystack.nodes.prompt.invocation_layer.hugging_face.get_task", mock_get_task):
        assert not HFLocalInvocationLayer.supports("nateraw/vit-age-classifier")
        assert mock_get_task.call_count == 1

    # or some POS tagging model
    mock_get_task = Mock(return_value="pos-tagging")
    with patch("haystack.nodes.prompt.invocation_layer.hugging_face.get_task", mock_get_task):
        assert not HFLocalInvocationLayer.supports("vblagoje/bert-english-uncased-finetuned-pos")
        assert mock_get_task.call_count == 1


@pytest.mark.unit
def test_stop_words_criteria_set(mock_pipeline, mock_get_task):
    """
    Test that stop words criteria is correctly set in pipeline invocation
    """
    layer = HFLocalInvocationLayer(
        model_name_or_path="hf-internal-testing/tiny-random-t5", task_name="text2text-generation"
    )

    layer.invoke(prompt="Tell me hello", stop_words=["hello", "world"])

    _, kwargs = layer.pipe.call_args
    assert "stopping_criteria" in kwargs
    assert isinstance(kwargs["stopping_criteria"], StoppingCriteriaList)
    assert len(kwargs["stopping_criteria"]) == 1
    assert isinstance(kwargs["stopping_criteria"][0], StopWordsCriteria)


@pytest.mark.integration
@pytest.mark.parametrize("stop_words", [["good"], ["hello", "good"]])
def test_stop_words_single_token(stop_words: List[str]):
    """
    Test that stop words criteria is used and that it works with single token stop words
    """

    # simple test with words not broken down into multiple tokens
    default_model = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(default_model)
    for stop_word in stop_words:
        # confirm we are dealing with single-token words
        tokens = tokenizer.tokenize(stop_word)
        assert len(tokens) == 1

    layer = HFLocalInvocationLayer(model_name_or_path=default_model)
    result = layer.invoke(prompt="Generate a sentence `I wish you a good health`", stop_words=stop_words)
    assert len(result) > 0
    assert result[0].startswith("I wish you a")
    assert "good" not in result[0]
    assert "health" not in result[0]


@pytest.mark.integration
@pytest.mark.parametrize(
    "stop_words", [["unambiguously"], ["unambiguously", "unrelated"], ["unambiguously", "hearted"]]
)
def test_stop_words_multiple_token(stop_words: List[str]):
    """
    Test that stop words criteria is used and that it works for multi-token words
    """
    default_model = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(default_model)
    for stop_word in stop_words:
        # confirm we are dealing with multi-token words
        tokens = tokenizer.tokenize(stop_word)
        assert len(tokens) > 1

    layer = HFLocalInvocationLayer(model_name_or_path=default_model)
    result = layer.invoke(prompt="Generate a sentence `I wish you unambiguously good health`", stop_words=stop_words)
    # yet the stop word is correctly stopped on and removed
    assert len(result) > 0
    assert result[0].startswith("I wish you")
    assert "unambiguously" not in result[0]
    assert "good" not in result[0]
    assert "health" not in result[0]


@pytest.mark.unit
def test_stop_words_criteria():
    """
    Test that StopWordsCriteria will check stop word tokens in a continuous and sequential order
    """
    # input ids for "unambiguously"
    stop_words_id = torch.tensor([[73, 24621, 11937]])

    # input ids for "This is ambiguously, but is unrelated."
    input_ids1 = torch.tensor([[100, 19, 24621, 11937, 6, 68, 19, 73, 3897, 5]])
    # input ids for "This is unambiguously"
    input_ids2 = torch.tensor([[100, 19, 73, 24621, 11937]])

    # We used to implement stop words algorithm using the torch.isin function like this:
    # `all(torch.isin(stop_words_id, input_ids1)[0])`
    # However, this algorithm is not correct as it will return True for presence of "unambiguously" in input_ids1
    # and True for presence of "unambiguously" in input_ids2. This is because the algorithm will check
    # if the stop word tokens are present in the input_ids, but it does not check if the stop word tokens are
    # present in a continuous/sequential order.

    # In "This is ambiguously, but is unrelated." sentence the "un" token comes from "unrelated" and the
    # "ambiguously" token comes from "ambiguously". The algorithm will return True for presence of
    # "unambiguously" in input_ids1 which is not correct.

    stop_words_criteria = StopWordsCriteria(tokenizer=Mock(), stop_words=["mock data"])
    # because we are mocking the tokenizer, we need to set the stop words manually
    stop_words_criteria.stop_words = stop_words_id

    # this is the correct algorithm to check if the stop word tokens are present in a continuous and sequential order
    # For the input_ids1, the stop word tokens are present BUT not in a continuous order
    present_and_continuous = stop_words_criteria(input_ids1, scores=None)
    assert not present_and_continuous

    # For the input_ids2, the stop word tokens are both present and in a continuous order
    present_and_continuous = stop_words_criteria(input_ids2, scores=None)
    assert present_and_continuous


@pytest.mark.integration
@pytest.mark.parametrize("stop_words", [["Berlin"], ["Berlin", "Brandenburg"], ["Berlin", "Brandenburg", "Germany"]])
def test_stop_words_not_being_found(stop_words: List[str]):
    """
    Test that stop works on tokens that are not found in the generated text, stop words are not found
    """
    layer = HFLocalInvocationLayer()
    result = layer.invoke(prompt="Generate a sentence `I wish you a good health`", stop_words=stop_words)
    assert len(result) > 0
    for word in "I wish you a good health".split():
        assert word in result[0]


@pytest.mark.unit
def test_generation_kwargs_from_constructor(mock_auto_tokenizer, mock_pipeline, mock_get_task):
    """
    Test that generation_kwargs are correctly passed to pipeline invocation from constructor
    """
    query = "What does 42 mean?"
    # test that generation_kwargs are passed to the underlying HF model
    layer = HFLocalInvocationLayer(generation_kwargs={"do_sample": True})
    layer.invoke(prompt=query)
    assert any(
        (call.kwargs == {"do_sample": True, "max_length": 100}) and (query in call.args)
        for call in mock_pipeline.mock_calls
    )

    # test that generation_kwargs in the form of GenerationConfig are passed to the underlying HF model
    layer = HFLocalInvocationLayer(generation_kwargs=GenerationConfig(do_sample=True, top_p=0.9))
    layer.invoke(prompt=query)
    assert any(
        (call.kwargs == {"do_sample": True, "max_length": 100, "top_p": 0.9}) and (query in call.args)
        for call in mock_pipeline.mock_calls
    )


@pytest.mark.unit
def test_generation_kwargs_from_invoke(mock_auto_tokenizer, mock_pipeline, mock_get_task):
    """
    Test that generation_kwargs passed to invoke are passed to the underlying HF model
    """
    query = "What does 42 mean?"
    # test that generation_kwargs are passed to the underlying HF model
    layer = HFLocalInvocationLayer()
    layer.invoke(prompt=query, generation_kwargs={"do_sample": True})
    assert any(
        (call.kwargs == {"do_sample": True, "max_length": 100}) and (query in call.args)
        for call in mock_pipeline.mock_calls
    )

    layer = HFLocalInvocationLayer()
    layer.invoke(prompt=query, generation_kwargs=GenerationConfig(do_sample=True, top_p=0.9))
    assert any(
        (call.kwargs == {"do_sample": True, "max_length": 100, "top_p": 0.9}) and (query in call.args)
        for call in mock_pipeline.mock_calls
    )


@pytest.mark.unit
def test_max_length_from_invoke(mock_auto_tokenizer, mock_pipeline, mock_get_task):
    """
    Test that max_length passed to invoke are passed to the underlying HF model
    """
    query = "What does 42 mean?"
    # test that generation_kwargs are passed to the underlying HF model
    layer = HFLocalInvocationLayer()
    layer.invoke(prompt=query, generation_kwargs={"max_length": 200})
    # find the call to pipeline invocation, and check that the kwargs are correct
    assert any((call.kwargs == {"max_length": 200}) and (query in call.args) for call in mock_pipeline.mock_calls)

    layer = HFLocalInvocationLayer()
    layer.invoke(prompt=query, generation_kwargs=GenerationConfig(max_length=235))
    assert any((call.kwargs == {"max_length": 235}) and (query in call.args) for call in mock_pipeline.mock_calls)


@pytest.mark.unit
def test_ensure_token_limit_positive_mock(mock_pipeline, mock_get_task, mock_auto_tokenizer):
    # prompt of length 5 + max_length of 3 = 8, which is less than model_max_length of 10, so no resize
    mock_tokens = ["I", "am", "a", "tokenized", "prompt"]
    mock_prompt = "I am a tokenized prompt"

    mock_auto_tokenizer.tokenize = Mock(return_value=mock_tokens)
    mock_auto_tokenizer.convert_tokens_to_string = Mock(return_value=mock_prompt)
    mock_pipeline.return_value.tokenizer = mock_auto_tokenizer

    layer = HFLocalInvocationLayer("google/flan-t5-base", max_length=3, model_max_length=10)
    result = layer._ensure_token_limit(mock_prompt)

    assert result == mock_prompt


@pytest.mark.unit
def test_ensure_token_limit_negative_mock(mock_pipeline, mock_get_task, mock_auto_tokenizer):
    # prompt of length 8 + max_length of 3 = 11, which is more than model_max_length of 10, so we resize to 7
    mock_tokens = ["I", "am", "a", "tokenized", "prompt", "of", "length", "eight"]
    correct_result = "I am a tokenized prompt of length"

    mock_auto_tokenizer.tokenize = Mock(return_value=mock_tokens)
    mock_auto_tokenizer.convert_tokens_to_string = Mock(return_value=correct_result)
    mock_pipeline.return_value.tokenizer = mock_auto_tokenizer

    layer = HFLocalInvocationLayer("google/flan-t5-base", max_length=3, model_max_length=10)
    result = layer._ensure_token_limit("I am a tokenized prompt of length eight")

    assert result == correct_result


@pytest.mark.unit
@patch("haystack.nodes.prompt.invocation_layer.hugging_face.AutoConfig.from_pretrained")
@patch("haystack.nodes.prompt.invocation_layer.hugging_face.AutoTokenizer.from_pretrained")
def test_tokenizer_loading_unsupported_model(mock_tokenizer, mock_config, mock_pipeline, mock_get_task, caplog):
    """
    Test loading of tokenizers for models that are not natively supported by the transformers library.
    """
    mock_config.return_value = Mock(tokenizer_class=None)

    with caplog.at_level(logging.WARNING):
        invocation_layer = HFLocalInvocationLayer("unsupported_model", trust_remote_code=True)
        assert (
            "The transformers library doesn't know which tokenizer class should be "
            "loaded for the model unsupported_model. Therefore, the tokenizer will be loaded in Haystack's "
            "invocation layer and then passed to the underlying pipeline. Alternatively, you could "
            "pass `tokenizer_class` to `model_kwargs` to workaround this, if your tokenizer is supported "
            "by the transformers library."
        ) in caplog.text
        assert mock_tokenizer.called


@pytest.mark.unit
@patch("haystack.nodes.prompt.invocation_layer.hugging_face.AutoTokenizer.from_pretrained")
def test_tokenizer_loading_unsupported_model_with_initialized_model(
    mock_tokenizer, mock_pipeline, mock_get_task, caplog
):
    """
    Test loading of tokenizers for models that are not natively supported by the transformers library. In this case,
    the model is already initialized and the model config is loaded from the model.
    """
    model = Mock()
    model.config = Mock(tokenizer_class=None, _name_or_path="unsupported_model")

    with caplog.at_level(logging.WARNING):
        invocation_layer = HFLocalInvocationLayer(model_name_or_path="unsupported", model=model, trust_remote_code=True)
        assert (
            "The transformers library doesn't know which tokenizer class should be "
            "loaded for the model unsupported_model. Therefore, the tokenizer will be loaded in Haystack's "
            "invocation layer and then passed to the underlying pipeline. Alternatively, you could "
            "pass `tokenizer_class` to `model_kwargs` to workaround this, if your tokenizer is supported "
            "by the transformers library."
        ) in caplog.text
        assert mock_tokenizer.called


@pytest.mark.unit
@patch("haystack.nodes.prompt.invocation_layer.hugging_face.AutoConfig.from_pretrained")
@patch("haystack.nodes.prompt.invocation_layer.hugging_face.AutoTokenizer.from_pretrained")
def test_tokenizer_loading_unsupported_model_with_tokenizer_class_in_config(
    mock_tokenizer, mock_config, mock_pipeline, mock_get_task, caplog
):
    """
    Test that tokenizer is not loaded if tokenizer_class is set in model config.
    """
    mock_config.return_value = Mock(tokenizer_class="Some-Supported-Tokenizer")

    with caplog.at_level(logging.WARNING):
        invocation_layer = HFLocalInvocationLayer(model_name_or_path="unsupported_model", trust_remote_code=True)
        assert not mock_tokenizer.called
        assert not caplog.text


@pytest.mark.unit
def test_skip_prompt_is_set_in_hf_text_streamer(mock_pipeline, mock_get_task):
    """
    Test that skip_prompt is set in HFTextStreamingHandler. Otherwise, we will output prompt text.
    """
    layer = HFLocalInvocationLayer(stream=True)

    layer.invoke(prompt="Tell me hello")

    _, kwargs = layer.pipe.call_args
    assert "streamer" in kwargs and isinstance(kwargs["streamer"], HFTokenStreamingHandler)
    assert kwargs["streamer"].skip_prompt
