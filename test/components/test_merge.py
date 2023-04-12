from typing import Dict, Any, List, Tuple

from canals import component


@component
class Merge:
    def __init__(
        self,
        inputs: List[str] = ["value"],
        output: str = "value",
    ):
        """
        Takes several input components and returns the first one that is not None.
        If no input connections received any value, returns None as well.

        Multi input, single output component. Doesn't take any parameter.

        :param inputs: the inputs to expect.
        :param output: the name of the output connection.
        """
        self.output = output
        self.init_parameters = {
            "inputs": inputs,
            "output": output,
        }
        self.inputs = inputs
        self.outputs = [output]

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        for _, value in data:
            if value is not None:
                return ({self.outputs[0]: value}, parameters)


def test_merge_default():
    component = Merge()
    results = component.run(name="test_component", data=[("value", 5)], parameters={})
    assert results == ({"value": 5}, {})
    assert component.init_parameters == {
        "inputs": ["value"],
        "output": "value",
    }


def test_merge_init_parameters():
    component = Merge(inputs=["test1", "test2", "test3"], output="test")
    results = component.run(name="test_component", data=[("test1", None), ("test2", 5), ("test3", 10)], parameters={})
    assert results == ({"test": 5}, {})
