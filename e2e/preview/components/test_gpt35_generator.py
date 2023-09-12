import os
import pytest
import openai
from haystack.preview.components.generators.openai.gpt35 import GPT35Generator


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_gpt35_generator_run():
    component = GPT35Generator(api_key=os.environ.get("OPENAI_API_KEY"), n=1)
    results = component.run(prompt="What's the capital of France?")

    assert len(results["replies"]) == 1
    assert "Paris" in results["replies"][0]

    assert len(results["metadata"]) == 1
    assert "gpt-3.5-turbo" in results["metadata"][0]["model"]
    assert "stop" == results["metadata"][0]["finish_reason"]


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_gpt35_generator_run_wrong_model_name():
    component = GPT35Generator(model_name="something-obviously-wrong", api_key=os.environ.get("OPENAI_API_KEY"), n=1)
    with pytest.raises(openai.InvalidRequestError, match="The model `something-obviously-wrong` does not exist"):
        component.run(prompt="What's the capital of France?")


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_gpt35_generator_run_streaming():
    class Callback:
        def __init__(self):
            self.responses = ""

        def __call__(self, chunk):
            self.responses += chunk.choices[0].delta.content if chunk.choices[0].delta else ""
            return chunk

    callback = Callback()
    component = GPT35Generator(os.environ.get("OPENAI_API_KEY"), streaming_callback=callback, n=1)
    results = component.run(prompt="What's the capital of France?")

    assert len(results["replies"]) == 1
    assert "Paris" in results["replies"][0]

    assert len(results["metadata"]) == 1
    assert "gpt-3.5-turbo" in results["metadata"][0]["model"]
    assert "stop" == results["metadata"][0]["finish_reason"]

    assert callback.responses == results["replies"][0]
