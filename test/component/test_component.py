import pytest

from canals import component
from canals.errors import ComponentError


def test_correct_declaration():
    @component
    class MockComponent:
        @component.return_types(output_value=int)
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

        @component.return_types(output_value=int)
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

        @component.return_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies that instantiation works with no issues
    assert component.registry["MockComponent"] == MockComponent
    comp = MockComponent()
    comp.store = "test_store"
    assert comp.store == "test_store"


def test_missing_decorator():
    with pytest.raises(ComponentError, match="must have a @return_types decorator"):

        @component
        class MockComponent:
            def run(self, input_value: int):
                return {"output_value": input_value}


def test_missing_run():
    with pytest.raises(ComponentError, match="must have a 'run\(\)' method"):

        @component
        class MockComponent:
            def another_method(self, input_value: int):
                return {"output_value": input_value}


def test_run_must_have_types():
    with pytest.raises(
        ComponentError,
        match="must declare types for all its parameters, but these parameters are not typed: input_value",
    ):

        @component
        class MockComponent:
            def run(self, input_value):
                return {"output_value": input_value}
