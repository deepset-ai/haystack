import pytest

from haystack.nodes.prompt.prompt_template import PromptTemplate


@pytest.mark.unit
def test_prompt_templates():
    p = PromptTemplate("t1", "Here is some fake template with variable {foo}")
    assert set(p.prompt_params) == {"foo"}

    p = PromptTemplate("t3", "Here is some fake template with variable {foo} and {bar}")
    assert set(p.prompt_params) == {"foo", "bar"}

    p = PromptTemplate("t4", "Here is some fake template with variable {foo1} and {bar2}")
    assert set(p.prompt_params) == {"foo1", "bar2"}

    p = PromptTemplate("t4", "Here is some fake template with variable {foo_1} and {bar_2}")
    assert set(p.prompt_params) == {"foo_1", "bar_2"}

    p = PromptTemplate("t4", "Here is some fake template with variable {Foo_1} and {Bar_2}")
    assert set(p.prompt_params) == {"Foo_1", "Bar_2"}

    p = PromptTemplate("t4", "'Here is some fake template with variable {baz}'")
    assert set(p.prompt_params) == {"baz"}
    # strip single quotes, happens in YAML as we need to use single quotes for the template string
    assert p.prompt_text == "Here is some fake template with variable {baz}"

    p = PromptTemplate("t4", '"Here is some fake template with variable {baz}"')
    assert set(p.prompt_params) == {"baz"}
    # strip double quotes, happens in YAML as we need to use single quotes for the template string
    assert p.prompt_text == "Here is some fake template with variable {baz}"


@pytest.mark.unit
def test_prompt_template_repr():
    p = PromptTemplate("t", "Here is variable {baz}")
    desired_repr = "PromptTemplate(name=t, prompt_text=Here is variable {baz}, prompt_params=['baz'])"
    assert repr(p) == desired_repr
    assert str(p) == desired_repr
