from typing import Dict, Any, List, Tuple

from canals import component


@component
class Sum:
    def __init__(self, inputs: List[str] = ["value"], output: str = "sum"):
        """
        Sums the values of all the input connections together.

        Multi input, single output component. Order of input connections is irrelevant.
         Doesn't have parameters.

        :param inputs: list of connections to sum.
        :param output: name of the output connection.
        """
        self.init_parameters = {"inputs": inputs, "output": output}
        self.inputs = inputs
        self.outputs = [output]

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        sum = 0
        for _, value in data:
            if value:
                sum += value

        return ({self.outputs[0]: sum}, parameters)


def test_sum_default():
    component = Sum()
    results = component.run(name="test_component", data=[("value", 10)], parameters={})
    assert results == ({"sum": 10}, {})
    assert component.init_parameters == {"inputs": ["value"], "output": "sum"}


def test_sum_init_params():
    component = Sum(inputs=["value", "value"], output="test_out")
    results = component.run(name="test_component", data=[("value", 10), ("value", 4)], parameters={})
    assert results == ({"test_out": 14}, {})
    assert component.init_parameters == {"inputs": ["value", "value"], "output": "test_out"}
