from typing import Dict, Any, List, Tuple

from canals import node


@node
class Subtract:
    def __init__(self, first_input: str, second_input: str, output: str = "diff"):
        """
        Subtracts the second edge's value from the first.

        Double input, single output node. Order of input edges is critical.
        Doesn't have parameters. Doesn't use stores.

        :param first_input: name of the edge carrying the value to subtract from.
        :param second_input: name of the edge carrying the value to subtract.
        :param output: name of the output edge.
        """
        self.init_parameters = {"first_input": first_input, "second_input": second_input, "output": output}
        self.inputs = [first_input, second_input]
        self.outputs = [output]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        first_value = [value for name, value in data if name == self.inputs[0]][0]
        second_value = [value for name, value in data if name == self.inputs[1]][0]

        return ({"diff": first_value - second_value}, parameters)


def test_subtract():
    node = Subtract(first_input="first", second_input="second", output="diff")
    results = node.run(name="test_node", data=[("second", 7), ("first", 10)], parameters={}, stores={})
    assert results == ({"diff": 3}, {})
    assert node.init_parameters == {"first_input": "first", "second_input": "second", "output": "diff"}
