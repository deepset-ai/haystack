from typing import List

from dataclasses import dataclass

import pytest

from canals.component import component, ComponentInput, VariadicComponentInput, ComponentOutput, ComponentError


def test_input_required():

    with pytest.raises(
        ComponentError,
        match="Components must either have an Input dataclass or a 'input_type' property that returns such dataclass",
    ):

        @component
        class MockComponent:
            @dataclass
            class Output(ComponentOutput):
                output_value: int

            def run(self, data) -> Output:
                return MockComponent.Output(output_value=1)


def test_output_required():

    with pytest.raises(
        ComponentError,
        match="Components must either have an Output dataclass or a 'output_type' property that returns such dataclass",
    ):

        @component
        class MockComponent:
            @dataclass
            class Input(ComponentInput):
                input_value: int

            def run(self, data: Input):
                return 1


def test_input_as_class():
    @component
    class MockComponent:
        @dataclass
        class Input(ComponentInput):
            single_value: int

        @dataclass
        class Output(ComponentOutput):
            output_value: int

        def run(self, data: Input) -> Output:
            return MockComponent.Output(output_value=1)


def test_variadic_input_as_class():
    @component
    class MockComponent:
        @dataclass
        class Input(VariadicComponentInput):
            single_value: int

        @dataclass
        class Output(ComponentOutput):
            output_value: int

        def run(self, data: Input) -> Output:
            return MockComponent.Output(output_value=1)


def test_variadic_input_with_more_than_one_param():

    with pytest.raises(ComponentError, match="Variadic inputs can contain only one variadic positional parameter."):

        @component
        class MockComponent:
            @dataclass
            class Input(VariadicComponentInput):
                single_value: int
                values: List[int]

            @dataclass
            class Output(ComponentOutput):
                output_value: int

            def run(self, data: Input) -> Output:
                return MockComponent.Output(output_value=1)


def test_input_as_class_must_be_a_dataclass():

    with pytest.raises(ComponentError, match="Input must be a dataclass"):

        @component
        class MockComponent:
            class Input(ComponentInput):
                single_value: int

            @dataclass
            class Output(ComponentOutput):
                output_value: int

            def run(self, data: Input) -> Output:
                return MockComponent.Output(output_value=1)


def test_input_as_class_must_inherit_from_componentinput():

    with pytest.raises(ComponentError, match="Input must inherit from ComponentInput"):

        @component
        class MockComponent:
            @dataclass
            class Input:
                single_value: int

            @dataclass
            class Output(ComponentOutput):
                output_value: int

            def run(self, data: Input) -> Output:
                return MockComponent.Output(output_value=1)


def test_input_as_property():
    @component
    class MockComponent:
        @dataclass
        class Output(ComponentOutput):
            output_value: int

        def input_type(self):
            return None

        def run(self, data) -> Output:
            return MockComponent.Output(output_value=1)


def test_output_as_class():
    @component
    class MockComponent:
        @dataclass
        class Input(ComponentInput):
            single_value: int

        @dataclass
        class Output(ComponentOutput):
            output_value: int

        def run(self, data: Input) -> Output:
            return MockComponent.Output(output_value=1)


def test_output_as_class_must_be_a_dataclass():

    with pytest.raises(ComponentError, match="Output must be a dataclass"):

        @component
        class MockComponent:
            @dataclass
            class Input(ComponentInput):
                single_value: int

            class Output(ComponentOutput):
                output_value: int

            def run(self, data: Input) -> Output:
                return MockComponent.Output()


def test_output_as_class_must_inherit_from_componentoutput():

    with pytest.raises(ComponentError, match="Output must inherit from ComponentOutput"):

        @component
        class MockComponent:
            @dataclass
            class Input(ComponentInput):
                single_value: int

            @dataclass
            class Output:
                output_value: int

            def run(self, data: Input) -> Output:
                return MockComponent.Output(output_value=1)


def test_output_as_property():
    @component
    class MockComponent:
        @dataclass
        class Input(ComponentInput):
            input_value: int

        def output_type(self):
            return None

        def run(self, data: Input):
            return self.output_type()


def test_check_for_run():

    with pytest.raises(ComponentError, match="must have a 'run\(\)' method"):

        @component
        class MockComponent:
            @dataclass
            class Input(ComponentInput):
                single_value: int

            @dataclass
            class Output(ComponentOutput):
                output_value: int


def test_run_takes_only_one_param():

    with pytest.raises(ComponentError, match="must accept only a single parameter called 'data'."):

        @component
        class MockComponent:
            @dataclass
            class Input(ComponentInput):
                single_value: int

            @dataclass
            class Output(ComponentOutput):
                output_value: int

            def run(self, data: Input, something_else: int) -> Output:
                return MockComponent.Output(output_value=1)


def test_run_takes_only_data():

    with pytest.raises(ComponentError, match="must accept a parameter called 'data'."):

        @component
        class MockComponent:
            @dataclass
            class Input(ComponentInput):
                single_value: int

            @dataclass
            class Output(ComponentOutput):
                output_value: int

            def run(self, wrong_name: Input) -> Output:
                return MockComponent.Output(output_value=1)


def test_run_data_must_be_typed_if_input_is_a_class():

    with pytest.raises(ComponentError, match="'data' must be typed"):

        @component
        class MockComponent:
            @dataclass
            class Input(ComponentInput):
                single_value: int

            @dataclass
            class Output(ComponentOutput):
                output_value: int

            def run(self, data) -> Output:
                return MockComponent.Output(output_value=1)


def test_run_data_needs_no_type_if_input_is_not_a_class():
    @component
    class MockComponent:
        @dataclass
        class Output(ComponentOutput):
            output_value: int

        def input_type(self):
            return None

        def run(self, data) -> Output:
            return MockComponent.Output(output_value=1)


def test_run_return_must_be_typed_if_output_is_a_class():

    with pytest.raises(ComponentError, match="must declare the type of its return value"):

        @component
        class MockComponent:
            @dataclass
            class Input(ComponentInput):
                single_value: int

            @dataclass
            class Output(ComponentOutput):
                output_value: int

            def run(self, data: Input):
                return MockComponent.Output(output_value=1)


def test_run_return_needs_no_type_if_output_is_not_a_class():
    @component
    class MockComponent:
        @dataclass
        class Input(ComponentInput):
            input_value: int

        def output_type(self):
            return None

        def run(self, data: Input):
            return self.output_type()(output_value=1)
