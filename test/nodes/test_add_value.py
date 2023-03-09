from typing import Dict, Any, List, Tuple

from canals import node


@node
class AddValue:
    def __init__(self, add: int = 1, input: str = "value", output: str = "value"):
        """
        Adds the value of `add` to the value of the incoming edge.

        Single input, single output node.

        :param add: the value to add. This is also a parameter.
        :param input: name of the input edge
        :param output: name of the output edge
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
    node = AddValue()
    results = node.run(name="test_node", data=[("value", 10)], parameters={})
    assert results == ({"value": 11}, {})
    assert node.init_parameters == {"add": 1, "input": "value", "output": "value"}


def test_addvalue_init_params():
    node = AddValue(add=3, input="test_in", output="test_out")
    results = node.run(name="test_node", data=[("test_in", 10)], parameters={})
    assert results == ({"test_out": 13}, {})
    assert node.init_parameters == {"add": 3, "input": "test_in", "output": "test_out"}


def test_addvalue_runtime_params():
    node = AddValue(add=3, input="test_in", output="test_out")
    results = node.run(name="test_node", data=[("test_in", 10)], parameters={"test_node": {"add": 5}})
    assert results == ({"test_out": 15}, {"test_node": {"add": 5}})
    assert node.init_parameters == {"add": 3, "input": "test_in", "output": "test_out"}
