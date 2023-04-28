from typing import List

from dataclasses import dataclass
from canals import component


@component
class Sum:
    """
    Sums the values of all the input connections together.

    Multi input, single output component. Order of input connections is irrelevant.
    Doesn't have parameters.
    """

    @dataclass
    class Output:
        sum: int

    def run(self, values: List[int]) -> Output:
        return Sum.Output(sum=sum(values))


# def test_sum_default():
#     component = Sum()
#     results = component.run(name="test_component", data=[("value", 10)], parameters={})
#     assert results == ({"sum": 10}, {})
#     assert component.init_parameters == {}


# def test_sum_init_params():
#     component = Sum(inputs=["value", "value"], output="test_out")
#     results = component.run(name="test_component", data=[("value", 10), ("value", 4)], parameters={})
#     assert results == ({"test_out": 14}, {})
#     assert component.init_parameters == {"inputs": ["value", "value"], "output": "test_out"}
