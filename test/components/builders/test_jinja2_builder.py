from haystack.components.builders.jinja2_builder import Jinja2Builder


def test_init():
    builder = Jinja2Builder(template="This is a {{ variable }}")
    assert builder._template_string == "This is a {{ variable }}"


def test_to_dict():
    builder = Jinja2Builder(template="This is a {{ variable }}")
    res = builder.to_dict()
    assert res == {
        "type": "haystack.components.builders.jinja2_builder.Jinja2Builder",
        "init_parameters": {"template": "This is a {{ variable }}"},
    }


def test_run():
    builder = Jinja2Builder(template="This is a {{ variable }}")
    res = builder.run(variable="test")
    assert res == {"string": "This is a test"}


def test_run_without_input():
    builder = Jinja2Builder(template="This is a template without input")
    res = builder.run()
    assert res == {"string": "This is a template without input"}


def test_run_with_missing_input():
    builder = Jinja2Builder(template="This is a {{ variable }}")
    res = builder.run()
    assert res == {"string": "This is a "}
