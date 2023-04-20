import pytest

import haystack
from haystack.preview.components.prompt.prompt import Prompt

from haystack.preview.components.prompt.models.base import prompt_model


@prompt_model
class MockImplementation:
    def __init__(self, model_name_or_path: str, **kwargs):
        pass

    def invoke(self, *args, **kwargs):
        return ["test response"]

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        return model_name_or_path == "mock_model"


@pytest.fixture()
def mock_implementation(monkeypatch):
    monkeypatch.setattr(haystack.preview.components.prompt.prompt, "IMPLEMENTATION_MODULES", [__name__])


@pytest.mark.unit
def test_prompt_custom_template_no_inputs(mock_implementation):
    pn = Prompt(custom_template="This is my own template", model_name_or_path="mock_model")
    output = pn.prompt()
    assert output == ["test response"]


@pytest.mark.unit
def test_prompt_custom_template_one_input_correct(mock_implementation):
    pn = Prompt(custom_template="This is my own template: {{ query }}", model_name_or_path="mock_model")
    output = pn.prompt(query="My question")
    assert output == ["test response"]


@pytest.mark.unit
def test_prompt_custom_template_one_input_missing(mock_implementation):
    pn = Prompt(custom_template="This is my own template: {{ query }}", model_name_or_path="mock_model")
    with pytest.raises(ValueError, match="query"):
        pn.prompt()


@pytest.mark.unit
def test_prompt_custom_template_one_input_wrong(mock_implementation):
    pn = Prompt(custom_template="This is my own template: {{ query }}", model_name_or_path="mock_model")
    with pytest.raises(ValueError, match="query"):
        pn.prompt(documents=["test!"])


@pytest.mark.unit
def test_prompt_custom_template_one_input_extra(mock_implementation):
    pn = Prompt(custom_template="This is my own template: {{ query }}", model_name_or_path="mock_model")
    with pytest.raises(ValueError, match="documents"):
        pn.prompt(query="test query", documents=["test!"])
