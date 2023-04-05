import os

import pytest

from haystack.preview.nodes.prompt import PromptNode


@pytest.mark.integration
def test_prompt_node_huggingface_provider():
    pn = PromptNode(template="question-answering", model_name_or_path="google/flan-t5-base")
    output = pn.prompt(question="What's the capital of France?", documents=["The capital of France is Paris."])
    assert "Paris" in output[0]


@pytest.mark.parametrize("model", ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"])
def test_prompt_node_openai_provider(model):
    pn = PromptNode(
        template="question-answering",
        model_name_or_path=model,
        model_kwargs={"api_key": os.environ.get("OPENAI_API_KEY", None)},
    )
    output = pn.prompt(question="What's the capital of France?", documents=["The capital of France is Paris."])
    assert "Paris" in output[0]
