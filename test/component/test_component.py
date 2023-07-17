import pytest

from canals.component import component
from canals.errors import ComponentError


def test_correct_declaration():
    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                input_value: int

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=1)

    # Verifies also instantiation works with no issues
    assert MockComponent()
    assert component.registry["MockComponent"] == MockComponent


def test_correct_declaration_with_additional_readonly_property():
    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                input_value: int

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        @property
        def store(self):
            return "test_store"

        def run(self, data):
            return self.output(output_value=1)

    # Verifies that instantiation works with no issues
    assert MockComponent()
    assert component.registry["MockComponent"] == MockComponent
    assert MockComponent().store == "test_store"


def test_correct_declaration_with_additional_writable_property():
    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                input_value: int

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        @property
        def store(self):
            return self._store

        @store.setter
        def store(self, value):
            self._store = value

        def run(self, data):
            return self.output(output_value=1)

    # Verifies that instantiation works with no issues
    assert component.registry["MockComponent"] == MockComponent
    comp = MockComponent()
    comp.store = "test_store"
    assert comp.store == "test_store"


def test_input_required():
    with pytest.raises(
        ComponentError,
        match="No input definition found in Component MockComponent. "
        "Create a method that returns a dataclass defining the input and "
        "decorate it with @component.input\(\) to fix the error.",
    ):

        @component
        class MockComponent:
            @component.output
            def output(self):
                class Output:
                    output_value: int

                return Output

            def run(self, data):
                return MockComponent.Output(output_value=1)


def test_output_required():
    with pytest.raises(
        ComponentError,
        match="No output definition found in Component MockComponent. "
        "Create a method that returns a dataclass defining the output and "
        "decorate it with @component.output\(\) to fix the error.",
    ):

        @component
        class MockComponent:
            @component.input
            def input(self):
                class Input:
                    input_value: int

                return Input

            def run(self, data):
                return 1


def test_only_single_input_defined():
    with pytest.raises(
        ComponentError,
        match="Multiple input definitions found for Component MockComponent",
    ):

        @component
        class MockComponent:
            @component.input
            def input(self):
                class Input:
                    input_value: int

                return Input

            @component.input
            def another_input(self):
                class Input:
                    input_value: int

                return Input

            @component.output
            def output(self):
                class Output:
                    output_value: int

                return Output

            def run(self, data):
                return self.output(output_value=1)


def test_only_single_output_defined():
    with pytest.raises(
        ComponentError,
        match="Multiple output definitions found for Component MockComponent",
    ):

        @component
        class MockComponent:
            @component.input
            def input(self):
                class Input:
                    input_value: int

                return Input

            @component.output
            def output(self):
                class Output:
                    output_value: int

                return Output

            @component.output
            def another_output(self):
                class Output:
                    output_value: int

                return Output

            def run(self, data):
                return self.output(output_value=1)


def test_check_for_run():
    with pytest.raises(ComponentError, match="must have a 'run\(\)' method"):

        @component
        class MockComponent:
            @component.input
            def input(self):
                class Input:
                    input_value: int

                return Input

            @component.output
            def output(self):
                class Output:
                    output_value: int

                return Output


def test_run_takes_only_one_param():
    with pytest.raises(ComponentError, match="must accept only a single parameter called 'data'."):

        @component
        class MockComponent:
            @component.input
            def input(self):
                class Input:
                    input_value: int

                return Input

            @component.output
            def output(self):
                class Output:
                    output_value: int

                return Output

            def run(self, data, something_else: int):
                return self.output(output_value=1)


def test_run_takes_only_kwarg_data():
    with pytest.raises(ComponentError, match="must accept a parameter called 'data'."):

        @component
        class MockComponent:
            @component.input
            def input(self):
                class Input:
                    input_value: int

                return Input

            @component.output
            def output(self):
                class Output:
                    output_value: int

                return Output

            def run(self, wrong_name):
                return self.output(output_value=1)
