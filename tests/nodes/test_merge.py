from typing import Dict, Any, List, Tuple

from canals import node


@node
class Merge:
    def __init__(
        self,
        inputs: List[str] = ["value"],
        output: str = "value",
    ):
        """
        Takes several input nodes and returns the first one that is not None.
        If no input edges received any value, returns None as well.

        Multi input, single output node. Doesn't take any parameter. Doesn't use stores.

        :param inputs: the inputs to expect.
        :param output: the name of the output edge.
        """
        self.output = output
        self.init_parameters = {
            "inputs": inputs,
            "output": output,
        }
        self.inputs = inputs
        self.outputs = [output]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        for _, value in data:
            if value is not None:
                return ({self.outputs[0]: value}, parameters)


def test_merge_default():
    node = Merge()
    results = node.run(name="test_node", data=[("value", 5)], parameters={}, stores={})
    assert results == ({"value": 5}, {})
    assert node.init_parameters == {
        "inputs": ["value"],
        "output": "value",
    }


def test_merge_init_parameters():
    node = Merge(inputs=["test1", "test2", "test3"], output="test")
    results = node.run(name="test_node", data=[("test1", None), ("test2", 5), ("test3", 10)], parameters={}, stores={})
    assert results == ({"test": 5}, {})
