import os

import pytest

from haystack.preview.components.prompt import Prompt


@pytest.mark.parametrize("model", ["text-davinci-003"])
def test_prompt_openai_provider(model):
    pn = Prompt(
        template="question-answering",
        model_name_or_path=model,
        model_params={"api_key": os.environ.get("OPENAI_API_KEY", None)},
    )
    output = pn.prompt(question="What's the capital of France?", documents=["The capital of France is Paris."])
    assert "Paris" in output[0]
