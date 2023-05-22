from unittest.mock import MagicMock, patch

import pytest
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BloomForCausalLM,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    StoppingCriteriaList,
    GenerationConfig,
)

from haystack.nodes.prompt.invocation_layer import HFLocalInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import HFTokenStreamingHandler, DefaultTokenStreamingHandler
from haystack.nodes.prompt.invocation_layer.hugging_face import StopWordsCriteria


@pytest.mark.unit
def test_constructor_with_model_name():
    """
    Test that the constructor sets the pipeline with the model name (if provided)
    """
    layer = HFLocalInvocationLayer("google/flan-t5-base")

    assert isinstance(layer.pipe.model, T5ForConditionalGeneration)
    assert isinstance(layer.pipe.tokenizer, T5TokenizerFast)


@pytest.mark.unit
def test_constructor_with_model_name_and_device_map():
    """
    Test that the constructor with device_map works in cases where it is provided alone or with devices
    """

    layer = HFLocalInvocationLayer(device_map="auto")

    assert isinstance(layer.pipe.model, T5ForConditionalGeneration)
    assert isinstance(layer.pipe.tokenizer, T5TokenizerFast)

    layer = HFLocalInvocationLayer(device_map="auto", devices=["cpu"])


@pytest.mark.integration
def test_constructor_with_custom_pretrained_model():
    """
    Test that the constructor sets the pipeline with the pretrained model (if provided)
    """
    model = AutoModelForSeq2SeqLM.from_pretrained("hf-internal-testing/tiny-random-t5")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

    layer = HFLocalInvocationLayer(
        model_name_or_path="irrelevant_when_model_is_provided",
        model=model,
        tokenizer=tokenizer,
        task_name="text2text-generation",
    )

    assert layer.pipe.model == model
    assert layer.pipe.tokenizer == tokenizer


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


@pytest.mark.integration
def test_supports(tmp_path):
    """
    Test that supports returns True correctly for HFLocalInvocationLayer
    """
    # some HF hub hosted models
    assert HFLocalInvocationLayer.supports("google/flan-t5-base")
    assert HFLocalInvocationLayer.supports("mosaicml/mpt-7b")
    assert HFLocalInvocationLayer.supports("CarperAI/stable-vicuna-13b-delta")

    # some HF local model directory, let's use the one from test/prompt/invocation_layer
    assert HFLocalInvocationLayer.supports(str(tmp_path))

    # but not some non text2text-generation or non text-generation model
    # i.e image classification model
    assert not HFLocalInvocationLayer.supports("nateraw/vit-age-classifier")

    # or some POS tagging model
    assert not HFLocalInvocationLayer.supports("vblagoje/bert-english-uncased-finetuned-pos")


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


@pytest.mark.unit
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
