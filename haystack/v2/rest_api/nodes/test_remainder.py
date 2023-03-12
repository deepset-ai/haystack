from typing import Dict, Any, List, Tuple, Optional

import pytest

from canals import node


@node
class Remainder:
    """
    Redirects the value, unchanged, along the edge corresponding to the remainder
    of a division. For example, if `divisor=3`, the value `5` would be sent along
    the second output edge.

    Single input, multi output decision node. Order of output edges is critical.
    Doesn't accept parameters (divisor is tied to the number of output edges).


    :param divisor: the number to divide the input value for.
    :param input: the name of the input edge.
    :param outputs: the name of the output edges. Must be equal in length to the
        divisor (if dividing by 3, you must give exactly three output names).
        Ordering is important.
    """

    def __init__(self, divisor: int = 2, input: str = "value", outputs: Optional[List[str]] = None):
        if outputs and len(outputs) != divisor:
            raise ValueError("The divisor must be equal to the number of output edges.")
        if not outputs:
            outputs = [str(remainder) for remainder in range(divisor)]
        self.divisor = divisor

        self.init_parameters = {"divisor": divisor, "input": input, "outputs": outputs}
        self.inputs = [input]
        self.outputs = outputs

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        remainder = data[0][1] % self.divisor
        return ({self.outputs[remainder]: data[0][1]}, parameters)


def test_remainder_default():
    node = Remainder()
    results = node.run(name="test_node", data=[("value", 10)], parameters={})
    assert results == ({"0": 10}, {})

    results = node.run(name="test_node", data=[("value", 11)], parameters={})
    assert results == ({"1": 11}, {})
    assert node.init_parameters == {"divisor": 2, "input": "value", "outputs": ["0", "1"]}


def test_remainder_default_output_for_divisor():
    node = Remainder(divisor=5)
    results = node.run(name="test_node", data=[("value", 10)], parameters={})
    assert results == ({"0": 10}, {})

    results = node.run(name="test_node", data=[("value", 13)], parameters={})
    assert results == ({"3": 13}, {})
    assert node.init_parameters == {"divisor": 5, "input": "value", "outputs": ["0", "1", "2", "3", "4"]}


def test_remainder_init_params():
    with pytest.raises(ValueError):
        node = Remainder(divisor=3, input="test", outputs=["one", "two"])

    with pytest.raises(ValueError):
        node = Remainder(divisor=3, input="test", outputs=["zero", "one", "two", "three"])

    node = Remainder(divisor=3, input="test", outputs=["zero", "one", "two"])
    results = node.run(name="test_node", data=[("value", 10)], parameters={})
    assert results == ({"one": 10}, {})
    assert node.init_parameters == {"divisor": 3, "input": "test", "outputs": ["zero", "one", "two"]}
