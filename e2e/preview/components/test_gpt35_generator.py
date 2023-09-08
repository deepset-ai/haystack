import os
import pytest
import openai
from haystack.preview.components.generators.openai.gpt35 import GPT35Generator
from haystack.preview.components.generators.openai.gpt4 import GPT4Generator


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.parametrize("generator_class,model_name", [(GPT35Generator, "gpt-3.5"), (GPT4Generator, "gpt-4")])
def test_gpt35_generator_run(generator_class, model_name):
    component = generator_class(api_key=os.environ.get("OPENAI_API_KEY"), n=1)
    results = component.run(prompts=["What's the capital of France?", "What's the capital of Germany?"])

    assert len(results["replies"]) == 2
    assert len(results["replies"][0]) == 1
    assert "Paris" in results["replies"][0][0]
    assert len(results["replies"][1]) == 1
    assert "Berlin" in results["replies"][1][0]

    assert len(results["metadata"]) == 2
    assert len(results["metadata"][0]) == 1
    assert model_name in results["metadata"][0][0]["model"]
    assert "stop" == results["metadata"][0][0]["finish_reason"]
    assert len(results["metadata"][1]) == 1
    assert model_name in results["metadata"][1][0]["model"]
    assert "stop" == results["metadata"][1][0]["finish_reason"]


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.parametrize("generator_class", [GPT35Generator, GPT4Generator])
def test_gpt35_generator_run_wrong_model_name(generator_class):
    component = generator_class(model_name="something-obviously-wrong", api_key=os.environ.get("OPENAI_API_KEY"), n=1)
    with pytest.raises(openai.InvalidRequestError, match="The model `something-obviously-wrong` does not exist"):
        component.run(prompts=["What's the capital of France?"])


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.parametrize("generator_class", [GPT35Generator, GPT4Generator])
def test_gpt35_generator_run_above_context_length(generator_class):
    component = generator_class(api_key=os.environ.get("OPENAI_API_KEY"), n=1)
    with pytest.raises(
        openai.InvalidRequestError,
        match="However, your messages resulted in 35008 tokens. Please reduce the length of the messages.",
    ):
        component.run(prompts=["What's the capital of France? " * 5_000])


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.parametrize("generator_class,model_name", [(GPT35Generator, "gpt-3.5"), (GPT4Generator, "gpt-4")])
def test_gpt35_generator_run_streaming(generator_class, model_name):
    class Callback:
        def __init__(self):
            self.responses = ""

        def __call__(self, chunk):
            self.responses += chunk.choices[0].delta.content if chunk.choices[0].delta else ""
            return chunk

    callback = Callback()
    component = generator_class(os.environ.get("OPENAI_API_KEY"), streaming_callback=callback, n=1)
    results = component.run(prompts=["What's the capital of France?", "What's the capital of Germany?"])

    assert len(results["replies"]) == 2
    assert len(results["replies"][0]) == 1
    assert "Paris" in results["replies"][0][0]
    assert len(results["replies"][1]) == 1
    assert "Berlin" in results["replies"][1][0]

    assert callback.responses == results["replies"][0][0] + results["replies"][1][0]

    assert len(results["metadata"]) == 2
    assert len(results["metadata"][0]) == 1

    assert model_name in results["metadata"][0][0]["model"]
    assert "stop" == results["metadata"][0][0]["finish_reason"]
    assert len(results["metadata"][1]) == 1
    assert model_name in results["metadata"][1][0]["model"]
    assert "stop" == results["metadata"][1][0]["finish_reason"]
