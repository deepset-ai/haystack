import pytest

import haystack
from haystack.preview.nodes.prompt import PromptNode

from haystack.preview.nodes.prompt.providers.base import prompt_model_provider


@prompt_model_provider
class MockProvider:
    def __init__(self, model_name_or_path: str, **kwargs):
        pass

    def invoke(self, *args, **kwargs):
        return ["test response"]

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        return model_name_or_path == "mock_model"


@pytest.fixture()
def mock_provider(monkeypatch):
    monkeypatch.setattr(haystack.preview.nodes.prompt.prompt_node, "PROVIDER_MODULES", [__name__])


@pytest.mark.unit
def test_prompt_node_custom_template_no_inputs(mock_provider):
    pn = PromptNode(custom_template="This is my own template", model_name_or_path="mock_model")
    output = pn.prompt()
    assert output == ["test response"]


@pytest.mark.unit
def test_prompt_node_custom_template_one_input_correct(mock_provider):
    pn = PromptNode(custom_template="This is my own template: {{ query }}", model_name_or_path="mock_model")
    output = pn.prompt(query="My question")
    assert output == ["test response"]


@pytest.mark.unit
def test_prompt_node_custom_template_one_input_missing(mock_provider):
    pn = PromptNode(custom_template="This is my own template: {{ query }}", model_name_or_path="mock_model")
    with pytest.raises(ValueError, match="query"):
        pn.prompt()


@pytest.mark.unit
def test_prompt_node_custom_template_one_input_wrong(mock_provider):
    pn = PromptNode(custom_template="This is my own template: {{ query }}", model_name_or_path="mock_model")
    with pytest.raises(ValueError, match="query"):
        pn.prompt(documents=["test!"])


@pytest.mark.unit
def test_prompt_node_custom_template_one_input_extra(mock_provider):
    pn = PromptNode(custom_template="This is my own template: {{ query }}", model_name_or_path="mock_model")
    with pytest.raises(ValueError, match="documents"):
        pn.prompt(query="test query", documents=["test!"])


@pytest.mark.integration
def test_prompt_node():
    pn = PromptNode(template="question-answering", model_name_or_path="google/flan-t5-base")
    output = pn.prompt(question="What's the capital of France?", documents=["The capital of France is Paris."])
    assert "Paris" in output[0]
