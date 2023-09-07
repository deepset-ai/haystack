import pytest

from haystack.preview.components.builders.prompt_builder import PromptBuilder


@pytest.mark.unit
def test_init():
    builder = PromptBuilder(template="This is a {{ variable }}")
    assert builder._template_string == "This is a {{ variable }}"


@pytest.mark.unit
def test_to_dict():
    builder = PromptBuilder(template="This is a {{ variable }}")
    res = builder.to_dict()
    assert res == {"type": "PromptBuilder", "init_parameters": {"template": "This is a {{ variable }}"}}


@pytest.mark.unit
def test_from_dict():
    data = {"type": "PromptBuilder", "init_parameters": {"template": "This is a {{ variable }}"}}
    builder = PromptBuilder.from_dict(data)
    builder._template_string == "This is a {{ variable }}"


@pytest.mark.unit
def test_run():
    builder = PromptBuilder(template="This is a {{ variable }}")
    res = builder.run(variable="test")
    assert res == {"prompt": "This is a test"}


@pytest.mark.unit
def test_run_without_input():
    builder = PromptBuilder(template="This is a template without input")
    res = builder.run()
    assert res == {"prompt": "This is a template without input"}


@pytest.mark.unit
def test_run_with_missing_input():
    builder = PromptBuilder(template="This is a {{ variable }}")
    res = builder.run()
    assert res == {"prompt": "This is a "}
