import pytest

from haystack.components.builders.prompt_builder import PromptBuilder


def test_init():
    builder = PromptBuilder(template="This is a {{ variable }}")
    assert builder._template_string == "This is a {{ variable }}"


def test_to_dict():
    builder = PromptBuilder(template="This is a {{ variable }}")
    res = builder.to_dict()
    assert res == {
        "type": "haystack.components.builders.prompt_builder.PromptBuilder",
        "init_parameters": {"template": "This is a {{ variable }}"},
    }


def test_run():
    builder = PromptBuilder(template="This is a {{ variable }}")
    res = builder.run(variable="test")
    assert res == {"prompt": "This is a test"}


def test_run_without_input():
    builder = PromptBuilder(template="This is a template without input")
    res = builder.run()
    assert res == {"prompt": "This is a template without input"}


def test_run_with_missing_input():
    builder = PromptBuilder(template="This is a {{ variable }}")
    res = builder.run()
    assert res == {"prompt": "This is a "}
