import pytest
from haystack.testing.sample_components import FString


def test_fstring_with_one_var():
    fstring = FString(template="Hello, {name}!", variables=["name"])
    output = fstring.run(name="Alice")
    assert output == {"string": "Hello, Alice!"}


def test_fstring_with_no_vars():
    fstring = FString(template="No variables in this template.", variables=[])
    output = fstring.run()
    assert output == {"string": "No variables in this template."}


def test_fstring_with_template_at_runtime():
    fstring = FString(template="Hello {name}", variables=["name"])
    output = fstring.run(template="Goodbye {name}!", name="Alice")
    assert output == {"string": "Goodbye Alice!"}


def test_fstring_with_vars_mismatch():
    fstring = FString(template="Hello {name}", variables=["name"])
    with pytest.raises(KeyError):
        fstring.run(template="Goodbye {person}!", name="Alice")


def test_fstring_with_vars_in_excess():
    fstring = FString(template="Hello {name}", variables=["name"])
    output = fstring.run(template="Goodbye!", name="Alice")
    assert output == {"string": "Goodbye!"}


def test_fstring_with_vars_missing():
    fstring = FString(template="{greeting}, {name}!", variables=["name"])
    with pytest.raises(KeyError):
        fstring.run(greeting="Hello")
