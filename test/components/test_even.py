from typing import Optional

from dataclasses import dataclass
import pytest

from canals import component


@component
class Even:
    """
    Redirects the value, unchanged, along the 'even' connection if even, or along the 'odd' one if odd.
    """

    @dataclass
    class Output:
        even: Optional[int] = None
        odd: Optional[int] = None

    def run(self, value: int) -> Output:
        """
        :param value: The value to check for parity
        """
        remainder = value % 2
        if remainder:
            return Even.Output(odd=value)
        return Even.Output(even=value)


# def test_remainder_default():
#     component = Remainder()
#     results = component.run(name="test_component", data=[("value", 10)], parameters={})
#     assert results == ({"0": 10}, {})

#     results = component.run(name="test_component", data=[("value", 11)], parameters={})
#     assert results == ({"1": 11}, {})
#     assert component.init_parameters == {}


# def test_remainder_default_output_for_divisor():
#     component = Remainder(divisor=5)
#     results = component.run(name="test_component", data=[("value", 10)], parameters={})
#     assert results == ({"0": 10}, {})

#     results = component.run(name="test_component", data=[("value", 13)], parameters={})
#     assert results == ({"3": 13}, {})
#     assert component.init_parameters == {"divisor": 5}


# def test_remainder_init_params():
#     with pytest.raises(ValueError):
#         component = Remainder(divisor=3, input="test", outputs=["one", "two"])

#     with pytest.raises(ValueError):
#         component = Remainder(divisor=3, input="test", outputs=["zero", "one", "two", "three"])

#     component = Remainder(divisor=3, input="test", outputs=["zero", "one", "two"])
#     results = component.run(name="test_component", data=[("value", 10)], parameters={})
#     assert results == ({"one": 10}, {})
#     assert component.init_parameters == {"divisor": 3, "input": "test", "outputs": ["zero", "one", "two"]}
