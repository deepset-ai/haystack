import os

import pytest

from haystack.preview.components.prompt import Prompt


# FIXME hangs locally!
# def test_prompt_huggingface_remote_provider():
#     pn = Prompt(
#         template="question-answering",
#         model_name_or_path="https://api-inference.huggingface.co/models/google/flan-t5-base",
#     )
#     output = pn.prompt(question="What's the capital of France?", documents=["The capital of France is Paris."])
#     assert "Paris" in output[0]


def test_prompt_huggingface_local_provider():
    pn = Prompt(template="question-answering", model_name_or_path="google/flan-t5-base")
    output = pn.prompt(question="What's the capital of France?", documents=["The capital of France is Paris."])
    assert "Paris" in output[0]


@pytest.mark.parametrize("model", ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"])
def test_prompt_openai_provider(model):
    pn = Prompt(
        template="question-answering",
        model_name_or_path=model,
        model_params={"api_key": os.environ.get("OPENAI_API_KEY", None)},
    )
    output = pn.prompt(question="What's the capital of France?", documents=["The capital of France is Paris."])
    assert "Paris" in output[0]
