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


@pytest.fixture(autouse=True)
def mock_provider_modules(monkeypatch):
    monkeypatch.setattr(haystack.preview.nodes.prompt.prompt_node, "PROVIDER_MODULES", [__name__])


@pytest.mark.unit
def test_prompt_node_custom_template_no_inputs():
    pn = PromptNode(custom_template="This is my own template", model_name_or_path="mock_model")
    output = pn.prompt()
    assert output == ["test response"]


@pytest.mark.unit
def test_prompt_node_custom_template_one_input_discovered_and_provided():
    pn = PromptNode(custom_template="This is my own template: $query", model_name_or_path="mock_model")
    output = pn.prompt(query="My question")
    assert output == ["test response"]


@pytest.mark.unit
def test_prompt_node_custom_template_parsing_variables_corner_cases():
    pn = PromptNode(
        custom_template="$startofstring: This is $my$own$template $split-here. $included%not)included $this_has_no_comma, $Numb3rsARE30k something_wrong $endofstring",
        model_name_or_path="mock_model",
    )
    output = pn.prompt(my=True, own=10, template="test", query="My question")
    assert output == ["test response"]

    # a1 = re.split(r"(\$|[^\w])", a)
    # [word for word in a1 if re.match(r"^\$[A-Za-z0-9_]+$", word)]


@pytest.mark.unit
def test_prompt_node_custom_template_one_input_given_and_provided():
    pn = PromptNode(
        inputs=["query"], custom_template="This is my own template: $query", model_name_or_path="mock_model"
    )
    output = pn.prompt(query="My question")
    assert output == ["test response"]


@pytest.mark.unit
def test_prompt_node_custom_template_one_input_needed_none_given():
    pn = PromptNode(custom_template="This is my own template: $query", model_name_or_path="mock_model")
    with pytest.raises(ValueError, match="query"):
        pn.prompt()


@pytest.mark.unit
def test_prompt_node_custom_template_one_input_needed_wrong_one_given():
    pn = PromptNode(custom_template="This is my own template: $query", model_name_or_path="mock_model")
    with pytest.raises(ValueError, match="query"):
        pn.prompt(documents=["test!"])


@pytest.mark.unit
def test_prompt_node_custom_template_one_input_needed_additional_one_given():
    pn = PromptNode(custom_template="This is my own template: $query", model_name_or_path="mock_model")
    with pytest.raises(ValueError, match="documents"):
        pn.prompt(query="test query", documents=["test!"])
