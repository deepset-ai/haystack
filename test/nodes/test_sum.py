from typing import Dict, Any, List, Tuple

from canals import node


@node
class Sum:
    def __init__(self, inputs: List[str] = ["value"], output: str = "sum"):
        """
        Sums the values of all the input edges together.

        Multi input, single output node. Order of input edges is irrelevant.
        Doesn't use stores. Doesn't have parameters.

        :param inputs: list of edges to sum.
        :param output: name of the output edge.
        """
        self.init_parameters = {"inputs": inputs, "output": output}
        self.inputs = inputs
        self.outputs = [output]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        sum = 0
        for _, value in data:
            if value:
                sum += value

        return ({self.outputs[0]: sum}, parameters)


def test_sum_default():
    node = Sum()
    results = node.run(name="test_node", data=[("value", 10)], parameters={}, stores={})
    assert results == ({"sum": 10}, {})
    assert node.init_parameters == {"inputs": ["value"], "output": "sum"}


def test_sum_init_params():
    node = Sum(inputs=["value", "value"], output="test_out")
    results = node.run(name="test_node", data=[("value", 10), ("value", 4)], parameters={}, stores={})
    assert results == ({"test_out": 14}, {})
    assert node.init_parameters == {"inputs": ["value", "value"], "output": "test_out"}
