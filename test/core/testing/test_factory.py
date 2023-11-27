import pytest

from haystack.core.component import component
from haystack.core.errors import ComponentError
from haystack.testing import factory


def test_component_class_default():
    MyComponent = factory.component_class("MyComponent")
    comp = MyComponent()
    res = comp.run(value=1)
    assert res == {"value": None}

    res = comp.run(value="something")
    assert res == {"value": None}

    res = comp.run(non_existing_input=1)
    assert res == {"value": None}


def test_component_class_is_registered():
    MyComponent = factory.component_class("MyComponent")
    assert component.registry["haystack.testing.factory.MyComponent"] == MyComponent


def test_component_class_with_input_types():
    MyComponent = factory.component_class("MyComponent", input_types={"value": int})
    comp = MyComponent()
    res = comp.run(value=1)
    assert res == {"value": None}

    res = comp.run(value="something")
    assert res == {"value": None}


def test_component_class_with_output_types():
    MyComponent = factory.component_class("MyComponent", output_types={"value": int})
    comp = MyComponent()

    res = comp.run(value=1)
    assert res == {"value": None}


def test_component_class_with_output():
    MyComponent = factory.component_class("MyComponent", output={"value": 100})
    comp = MyComponent()
    res = comp.run(value=1)
    assert res == {"value": 100}


def test_component_class_with_output_and_output_types():
    MyComponent = factory.component_class("MyComponent", output_types={"value": str}, output={"value": 100})
    comp = MyComponent()

    res = comp.run(value=1)
    assert res == {"value": 100}


def test_component_class_with_bases():
    MyComponent = factory.component_class("MyComponent", bases=(Exception,))
    comp = MyComponent()
    assert isinstance(comp, Exception)


def test_component_class_with_extra_fields():
    MyComponent = factory.component_class("MyComponent", extra_fields={"my_field": 10})
    comp = MyComponent()
    assert comp.my_field == 10
