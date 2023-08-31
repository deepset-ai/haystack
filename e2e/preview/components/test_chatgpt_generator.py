import os
import pytest
from haystack.preview.components.generators.openai.chatgpt import ChatGPTGenerator


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_chatgpt_generator_run():
    component = ChatGPTGenerator(api_key=os.environ.get("OPENAI_API_KEY"))
    results = component.run(prompts=["What's the capital of France?", "What's the capital of Germany?"], n=1)

    assert len(results["replies"]) == 2
    assert len(results["replies"][0]) == 1
    assert "Paris" in results["replies"][0][0]
    assert len(results["replies"][1]) == 1
    assert "Berlin" in results["replies"][1][0]


# @pytest.mark.skipif(
#     not os.environ.get("OPENAI_API_KEY", None),
#     reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
# )
# def test_chatgpt_generator_run_streaming():

#     class Callback:
#         def __init__(self):
#             self.response = ""

#         def __call__(self, token):
#             self.responses += token
#             return token

#     callback = Callback()

#     component = ChatGPTGenerator(
#         os.environ.get("OPENAI_API_KEY"), stream=True, streaming_callback=callback
#     )
#     results = component.run(prompts=["test-prompt-1", "test-prompt-2"])

#     assert results == {"replies": [["test-response-a"], ["test-response-b"]]}

#     assert callback.responses == "test-response-a\ntest-response-b\n"
