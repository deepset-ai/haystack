from typing import Dict, Any, List, Tuple

from canals import component


@component
class Below:
    def __init__(
        self,
        threshold: int,
        input: str = "value",
        output_above: str = "above",
        output_below: str = "below",
    ):
        """
        Redirects the value, unchanged, along a different connection whether the value is above
        or below the given threshold.

        Single input, double output decision component.

        :param threshold: the number to compare the input value against. This is also a parameter.
        :param input: the name of the input connection.
        :param output_above: the name of the output connection if the value is above the threshold.
        :param output_below: the name of the output connection if the value is below the threshold.
        """
        self.threshold = threshold
        self.output_above = output_above
        self.output_below = output_below

        self.init_parameters = {
            "threshold": threshold,
            "input": input,
            "output_above": output_above,
            "output_below": output_below,
        }
        self.inputs = [input]
        self.outputs = [output_above, output_below]

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        threshold = parameters.get(name, {}).get("threshold", self.threshold)
        if data[0][1] < threshold:
            return ({self.output_below: data[0][1]}, parameters)
        else:
            return ({self.output_above: data[0][1]}, parameters)


def test_below_default():
    component = Below(threshold=10)
    results = component.run(name="test_component", data=[("value", 5)], parameters={})
    assert results == ({"below": 5}, {})

    results = component.run(name="test_component", data=[("value", 15)], parameters={})
    assert results == ({"above": 15}, {})

    assert component.init_parameters == {
        "threshold": 10,
        "input": "value",
        "output_above": "above",
        "output_below": "below",
    }


def test_below_init_parameters():
    component = Below(threshold=10, input="test", output_above="higher", output_below="lower")
    results = component.run(name="test_component", data=[("test", 5)], parameters={})
    assert results == ({"lower": 5}, {})

    results = component.run(name="test_component", data=[("test", 15)], parameters={})
    assert results == ({"higher": 15}, {})

    assert component.init_parameters == {
        "threshold": 10,
        "input": "test",
        "output_above": "higher",
        "output_below": "lower",
    }


def test_below_run_parameters():
    component = Below(threshold=10, input="test", output_above="higher", output_below="lower")
    results = component.run(name="test_component", data=[("test", 5)], parameters={"test_component": {"threshold": 1}})
    assert results == ({"higher": 5}, {"test_component": {"threshold": 1}})

    results = component.run(
        name="test_component", data=[("test", 15)], parameters={"test_component": {"threshold": 100}}
    )
    assert results == ({"lower": 15}, {"test_component": {"threshold": 100}})

    assert component.init_parameters == {
        "threshold": 10,
        "input": "test",
        "output_above": "higher",
        "output_below": "lower",
    }
