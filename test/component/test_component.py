from typing import Any

import pytest

from canals import component
from canals.component.component import _is_valid_socket_name
from canals.errors import ComponentError


def test_correct_declaration():
    @component
    class MockComponent:
        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies also instantiation works with no issues
    assert MockComponent()
    assert component.registry["MockComponent"] == MockComponent


def test_correct_declaration_with_additional_readonly_property():
    @component
    class MockComponent:
        @property
        def store(self):
            return "test_store"

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies that instantiation works with no issues
    assert MockComponent()
    assert component.registry["MockComponent"] == MockComponent
    assert MockComponent().store == "test_store"


def test_correct_declaration_with_additional_writable_property():
    @component
    class MockComponent:
        @property
        def store(self):
            return "test_store"

        @store.setter
        def store(self, value):
            self._store = value

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies that instantiation works with no issues
    assert component.registry["MockComponent"] == MockComponent
    comp = MockComponent()
    comp.store = "test_store"
    assert comp.store == "test_store"


def test_missing_run():
    with pytest.raises(ComponentError, match="must have a 'run\(\)' method"):

        @component
        class MockComponent:
            def another_method(self, input_value: int):
                return {"output_value": input_value}


def test_set_input_types():
    class MockComponent:
        def __init__(self):
            component.set_input_types(self, value=Any)

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(value=int)
        def run(self, **kwargs):
            return {"value": 1}

    comp = MockComponent()
    assert comp.run.__canals_input__ == {
        "value": {
            "name": "value",
            "type": Any,
            "is_optional": False,
        }
    }
    assert comp.run() == {"value": 1}


def test_set_input_types_with_invalid_socket_name():
    class MockComponent:
        def __init__(self):
            component.set_input_types(self, **{"non valid": Any})

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(value=int)
        def run(self, **kwargs):
            return {"value": 1}

    with pytest.raises(ComponentError) as err:
        MockComponent()

    err.match("Invalid socket name 'non valid'")


def test_set_output_types():
    @component
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, value=int)

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        def run(self, value: int):
            return {"value": 1}

    comp = MockComponent()
    assert comp.run.__canals_output__ == {
        "value": {
            "name": "value",
            "type": int,
        }
    }


def test_set_output_types_with_invalid_socket_name():
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, **{"non valid": Any})

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        def run(self, value: int):
            return {"non valid": 1}

    with pytest.raises(ComponentError) as err:
        MockComponent()

    err.match("Invalid socket name 'non valid'")


def test_output_types_decorator_with_compatible_type():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

    comp = MockComponent()
    assert comp.run.__canals_output__ == {
        "value": {
            "name": "value",
            "type": int,
        }
    }


def test_output_types_decorator_with_invalid_socket_name():
    with pytest.raises(ComponentError) as err:

        @component
        class MockComponent:
            @component.output_types(**{"non valid": int})
            def run(self, value: int):
                return {"non valid": 1}

            def to_dict(self):
                return {}

            @classmethod
            def from_dict(cls, data):
                return cls()

    err.match("Invalid socket name 'non valid'")


def test_component_decorator_set_it_as_component():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

    comp = MockComponent()
    assert comp.__canals_component__


def test_is_valid_socket_name():
    assert _is_valid_socket_name("socket_name")
    assert _is_valid_socket_name("with_underscore")
    assert _is_valid_socket_name("value1")

    assert not _is_valid_socket_name("1")
    assert not _is_valid_socket_name(" ")
    assert not _is_valid_socket_name(" name")
    assert not _is_valid_socket_name("if")
    assert not _is_valid_socket_name("with space")
    assert not _is_valid_socket_name("with-hyphen")
    assert not _is_valid_socket_name("with.dot")
    assert not _is_valid_socket_name("1value")
    assert not _is_valid_socket_name("*value")
