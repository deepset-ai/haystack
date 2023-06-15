from typing import List

import pytest

from canals.component import component, ComponentError


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


def test_variadic_input():
    @component
    class MockComponent:
        @component.input(variadic=True)
        def input(self):
            class Input:
                values: List[int]

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=1)

    # Variadic input is verified when accessing the decorated property
    assert MockComponent().input


def test_variadic_input_with_more_than_one_param():
    @component
    class MockComponent:
        @component.input(variadic=True)
        def input(self):
            class Input:
                single_value: int
                values: List[int]

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=1)

    # Variadic input is verified when accessing the decorated property
    with pytest.raises(ComponentError, match="Variadic input dataclass Input must have only one field"):
        assert MockComponent().input


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
