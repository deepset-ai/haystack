from unittest.mock import MagicMock, patch, Mock

import pytest
import torch
from torch import device
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BloomForCausalLM, StoppingCriteriaList, GenerationConfig

from haystack.nodes.prompt.invocation_layer import HFLocalInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import HFTokenStreamingHandler, DefaultTokenStreamingHandler
from haystack.nodes.prompt.invocation_layer.hugging_face import StopWordsCriteria


@pytest.fixture
def mock_pipeline():
    # mock transformers pipeline
    with patch("haystack.nodes.prompt.invocation_layer.hugging_face.pipeline") as mocked_pipeline:
        mocked_pipeline.return_value = Mock(**{"model_name_or_path": None, "tokenizer.model_max_length": 100})
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

    args, kwargs = mock_pipeline.call_args

    # device is set to cpu by default and device_map is empty
    assert kwargs["device"] == device("cpu")
    assert not kwargs["device_map"]

    # correct task and model are set
    assert kwargs["task"] == "text2text-generation"
    assert kwargs["model"] == "google/flan-t5-base"

    # no matter what kwargs we pass or don't pass, there are always 13 predefined kwargs passed to the pipeline
    assert len(kwargs) == 13

    # and these kwargs are passed to the pipeline
    assert list(kwargs.keys()) == [
        "task",
        "model",
        "config",
        "tokenizer",
        "feature_extractor",
        "revision",
        "use_auth_token",
        "device_map",
        "device",
        "torch_dtype",
        "trust_remote_code",
        "model_kwargs",
        "pipeline_class",
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

    args, kwargs = mock_pipeline.call_args

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

    args, kwargs = mock_pipeline.call_args
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

    args, kwargs = mock_pipeline.call_args
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

    args, kwargs = mock_pipeline.call_args
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
    # actual model and tokenizer passed to the pipeline
    model = AutoModelForSeq2SeqLM.from_pretrained("hf-internal-testing/tiny-random-t5")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

    HFLocalInvocationLayer(
        model_name_or_path="irrelevant_when_model_is_provided",
        model=model,
        tokenizer=tokenizer,
        task_name="text2text-generation",
    )

    mock_pipeline.assert_called_once()
    # mock_get_task is not called as we provided task_name parameter
    mock_get_task.assert_not_called()

    args, kwargs = mock_pipeline.call_args

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

    args, kwargs = mock_pipeline.call_args

    # invalid kwargs are ignored and not passed to the pipeline
    assert "some_invalid_kwarg" not in kwargs

    # still our 13 kwargs passed to the pipeline
    assert len(kwargs) == 13


@pytest.mark.unit
def test_constructor_with_various_kwargs(mock_pipeline, mock_get_task):
    """
    Test HFLocalInvocationLayer init with various kwargs, make sure all of them are passed to the pipeline
    except for the invalid ones
    """

    HFLocalInvocationLayer(
        "google/flan-t5-base",
        task_name="text2text-generation",
        tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-base"),
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

    args, kwargs = mock_pipeline.call_args

    # invalid kwargs are ignored and not passed to the pipeline
    assert "first_invalid_kwarg" not in kwargs
    assert "second_invalid_kwarg" not in kwargs

    # correct task and model are set as well
    assert kwargs["task"] == "text2text-generation"
    assert not kwargs["device"]
    assert kwargs["device_map"] and kwargs["device_map"] == "auto"
    assert kwargs["revision"] == "1.1"

    # still on 13 kwargs passed to the pipeline
    assert len(kwargs) == 13


@pytest.mark.unit
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


@pytest.mark.unit
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
def test_streaming_stream_param_in_constructor():
    """
    Test stream parameter is correctly passed to pipeline invocation via HF streamer parameter
    """
    layer = HFLocalInvocationLayer(stream=True)
    layer.pipe = MagicMock()

    layer.invoke(prompt="Tell me hello")

    args, kwargs = layer.pipe.call_args
    assert "streamer" in kwargs and isinstance(kwargs["streamer"], HFTokenStreamingHandler)


@pytest.mark.unit
def test_streaming_stream_handler_param_in_constructor():
    """
    Test stream parameter is correctly passed to pipeline invocation
    """
    dtsh = DefaultTokenStreamingHandler()
    layer = HFLocalInvocationLayer(stream_handler=dtsh)
    layer.pipe = MagicMock()

    layer.invoke(prompt="Tell me hello")

    args, kwargs = layer.pipe.call_args
    assert "streamer" in kwargs
    hf_streamer = kwargs["streamer"]

    # we wrap our TokenStreamingHandler with HFTokenStreamingHandler
    assert isinstance(hf_streamer, HFTokenStreamingHandler)

    # but under the hood, the wrapped handler is DefaultTokenStreamingHandler we passed
    assert isinstance(hf_streamer.token_handler, DefaultTokenStreamingHandler)
    assert hf_streamer.token_handler == dtsh


@pytest.mark.unit
def test_supports(tmp_path):
    """
    Test that supports returns True correctly for HFLocalInvocationLayer
    """
    # mock get_task to avoid remote calls to HF hub
    mock_get_task = Mock(return_value="text2text-generation")

    with patch("haystack.nodes.prompt.invocation_layer.hugging_face.get_task", mock_get_task):
        assert HFLocalInvocationLayer.supports("google/flan-t5-base")
        assert HFLocalInvocationLayer.supports("mosaicml/mpt-7b")
        assert HFLocalInvocationLayer.supports("CarperAI/stable-vicuna-13b-delta")
        assert mock_get_task.call_count == 3

    # some HF local model directory, let's use the one from test/prompt/invocation_layer
    assert HFLocalInvocationLayer.supports(str(tmp_path))

    # but not some non text2text-generation or non text-generation model
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

    # unless we specify the task name to override the default
    # short-circuit the get_task call
    assert HFLocalInvocationLayer.supports(
        "vblagoje/bert-english-uncased-finetuned-pos", task_name="text2text-generation"
    )


@pytest.mark.unit
def test_stop_words_criteria_set():
    """
    Test that stop words criteria is correctly set in pipeline invocation
    """
    layer = HFLocalInvocationLayer(
        model_name_or_path="hf-internal-testing/tiny-random-t5", task_name="text2text-generation"
    )
    layer.pipe = MagicMock()

    layer.invoke(prompt="Tell me hello", stop_words=["hello", "world"])

    args, kwargs = layer.pipe.call_args
    assert "stopping_criteria" in kwargs
    assert isinstance(kwargs["stopping_criteria"], StoppingCriteriaList)
    assert len(kwargs["stopping_criteria"]) == 1
    assert isinstance(kwargs["stopping_criteria"][0], StopWordsCriteria)


@pytest.mark.integration
@pytest.mark.parametrize("stop_words", [["good"], ["hello", "good"], ["hello", "good", "health"]])
def test_stop_words_single_token(stop_words):
    """
    Test that stop words criteria is used and that it works with single token stop words
    """

    # simple test with words not broken down into multiple tokens
    default_model = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(default_model)
    # each word is broken down into a single token
    tokens = tokenizer.tokenize("good health wish")
    assert len(tokens) == 3

    layer = HFLocalInvocationLayer(model_name_or_path=default_model)
    result = layer.invoke(prompt="Generate a sentence `I wish you a good health`", stop_words=stop_words)
    assert len(result) > 0
    assert result[0].startswith("I wish you a")
    assert "good" not in result[0]
    assert "health" not in result[0]


@pytest.mark.integration
def test_stop_words_multiple_token():
    """
    Test that stop words criteria is used and that it works for multi-token words
    """
    # complex test with words broken down into multiple tokens
    default_model = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(default_model)
    # single word unambiguously is broken down into 3 tokens
    tokens = tokenizer.tokenize("unambiguously")
    assert len(tokens) == 3

    layer = HFLocalInvocationLayer(model_name_or_path=default_model)
    result = layer.invoke(
        prompt="Generate a sentence `I wish you unambiguously good health`", stop_words=["unambiguously"]
    )
    # yet the stop word is correctly stopped on and removed
    assert len(result) > 0
    assert result[0].startswith("I wish you")
    assert "unambiguously" not in result[0]
    assert "good" not in result[0]
    assert "health" not in result[0]


@pytest.mark.integration
def test_stop_words_not_being_found():
    # simple test with words not broken down into multiple tokens
    layer = HFLocalInvocationLayer()
    result = layer.invoke(prompt="Generate a sentence `I wish you a good health`", stop_words=["Berlin"])
    assert len(result) > 0
    for word in "I wish you a good health".split():
        assert word in result[0]


@pytest.mark.unit
def test_generation_kwargs_from_constructor():
    """
    Test that generation_kwargs are correctly passed to pipeline invocation from constructor
    """
    the_question = "What does 42 mean?"
    # test that generation_kwargs are passed to the underlying HF model
    layer = HFLocalInvocationLayer(generation_kwargs={"do_sample": True})
    with patch.object(layer.pipe, "run_single", MagicMock()) as mock_call:
        layer.invoke(prompt=the_question)

    mock_call.assert_called_with(the_question, {}, {"do_sample": True, "max_length": 100}, {})

    # test that generation_kwargs in the form of GenerationConfig are passed to the underlying HF model
    layer = HFLocalInvocationLayer(generation_kwargs=GenerationConfig(do_sample=True, top_p=0.9))
    with patch.object(layer.pipe, "run_single", MagicMock()) as mock_call:
        layer.invoke(prompt=the_question)

    mock_call.assert_called_with(the_question, {}, {"do_sample": True, "top_p": 0.9, "max_length": 100}, {})


@pytest.mark.unit
def test_generation_kwargs_from_invoke():
    """
    Test that generation_kwargs passed to invoke are passed to the underlying HF model
    """
    the_question = "What does 42 mean?"
    # test that generation_kwargs are passed to the underlying HF model
    layer = HFLocalInvocationLayer()
    with patch.object(layer.pipe, "run_single", MagicMock()) as mock_call:
        layer.invoke(prompt=the_question, generation_kwargs={"do_sample": True})

    mock_call.assert_called_with(the_question, {}, {"do_sample": True, "max_length": 100}, {})

    # test that generation_kwargs in the form of GenerationConfig are passed to the underlying HF model
    layer = HFLocalInvocationLayer()
    with patch.object(layer.pipe, "run_single", MagicMock()) as mock_call:
        layer.invoke(prompt=the_question, generation_kwargs=GenerationConfig(do_sample=True, top_p=0.9))

    mock_call.assert_called_with(the_question, {}, {"do_sample": True, "top_p": 0.9, "max_length": 100}, {})
