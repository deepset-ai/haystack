from typing import Optional

from dataclasses import dataclass
from canals import component


@component
class AddValue:
    """
    Adds the value of `add` to the value of the incoming connection.

    Single input, single output component.
    """

    @dataclass
    class Output:
        value: int

    def __init__(self, add: int = 1):
        """
        :param add: the default value to add.
        """
        self.add = add

    def run(self, value: int, add: Optional[int] = None) -> Output:
        """
        Sums the incoming value with the value to add.
        """
        if add is None:
            add = self.add
        return AddValue.Output(value=value + add)


def test_addvalue_default():
    component = AddValue()
    results = component.run(value=10)
    assert results == AddValue.Output(value=11)
    assert component.init_parameters == {}


def test_addvalue_only_init_params():
    component = AddValue(add=3)
    results = component.run(value=100)
    assert results == AddValue.Output(value=103)
    assert component.init_parameters == {"add": 3}


def test_addvalue_only_runtime_params():
    component = AddValue()
    results = component.run(value=50, add=10)
    assert results == AddValue.Output(value=60)
    assert component.init_parameters == {}


def test_addvalue_init_and_runtime_params():
    component = AddValue(add=3)
    results = component.run(value=50, add=6)
    assert results == AddValue.Output(value=56)
    assert component.init_parameters == {"add": 3}
