from typing import Dict, Any, List, Tuple

from canals import component


@component
class AddValue:
    def __init__(self, add: int = 1, input: str = "value", output: str = "value"):
        """
        Adds the value of `add` to the value of the incoming connection.

        Single input, single output component.

        :param add: the value to add. This is also a parameter.
        :param input: name of the input connection
        :param output: name of the output connection
        """
        self.add = add

        self.init_parameters = {"add": add, "input": input, "output": output}
        self.inputs = [input]
        self.outputs = [output]

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        sum = parameters.get(name, {}).get("add", self.add)
        sum += data[0][1]
        return ({self.outputs[0]: sum}, parameters)


def test_addvalue_default():
    component = AddValue()
    results = component.run(name="test_component", data=[("value", 10)], parameters={})
    assert results == ({"value": 11}, {})
    assert component.init_parameters == {"add": 1, "input": "value", "output": "value"}


def test_addvalue_init_params():
    component = AddValue(add=3, input="test_in", output="test_out")
    results = component.run(name="test_component", data=[("test_in", 10)], parameters={})
    assert results == ({"test_out": 13}, {})
    assert component.init_parameters == {"add": 3, "input": "test_in", "output": "test_out"}


def test_addvalue_runtime_params():
    component = AddValue(add=3, input="test_in", output="test_out")
    results = component.run(name="test_component", data=[("test_in", 10)], parameters={"test_component": {"add": 5}})
    assert results == ({"test_out": 15}, {"test_component": {"add": 5}})
    assert component.init_parameters == {"add": 3, "input": "test_in", "output": "test_out"}
