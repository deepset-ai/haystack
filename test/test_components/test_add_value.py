import pytest
from dataclasses import dataclass
from canals import component
from canals.testing.test_component import BaseTestComponent


@component
class AddFixedValue:
    """
    Adds the value of `add` to `value`. If not given, `add` defaults to 1.
    """

    @dataclass
    class Output:
        value: int

    def __init__(self, add: int = 1):
        self.defaults = {"add": add}

    def run(self, value: int, add: int) -> Output:
        return AddFixedValue.Output(value=value + add)


class TestAddFixedValue(BaseTestComponent):
    @pytest.fixture
    def components(self):
        return [AddFixedValue(), AddFixedValue(add=2)]

    def test_addvalue(self):
        component = AddFixedValue()
        results = component.run(value=50, add=10)
        assert results == AddFixedValue.Output(value=60)
        assert component._init_parameters == {}
