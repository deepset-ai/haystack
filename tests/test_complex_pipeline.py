from typing import Dict, Any, List, Tuple

from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from canals.nodes import *
from canals.nodes import node

import logging

logging.basicConfig(level=logging.DEBUG)


@node
class AddValue:
    """
    Single input, single output node
    """

    def __init__(self, add: int = 1, input_name: str = "value", output_name: str = "value"):
        self.add = add

        # Contract
        self.init_parameters = {"add": add}
        self.inputs = [input_name]
        self.outputs = [output_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        for _, value in data:
            value += self.add

        return {"value": value}


@node
class Sum:
    """
    Multi input, single output node
    """

    def __init__(self, inputs_name: str = "value", inputs_count: int = 2):
        # Contract
        self.init_parameters = {
            "inputs_count": inputs_count,
            "inputs_name": inputs_name,
        }
        self.inputs = [inputs_name] * inputs_count
        self.outputs = ["sum"]

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

        return {"sum": sum}


@node
class Subtract:
    """
    Multi input, single output node
    """

    def __init__(self, first_value: str, second_value: str):
        self.first_value = first_value
        self.second_value = second_value
        # Contract
        self.init_parameters = {
            "first_value": first_value,
            "second_value": second_value,
        }
        self.inputs = [first_value, second_value]
        self.outputs = ["diff"]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        if len(data) != 2:
            raise ValueError("Subtract takes exactly two values.")

        first_value = [value for name, value in data if name == self.first_value][0]
        second_value = [value for name, value in data if name == self.second_value][0]

        return {"diff": first_value - second_value}


@node
class Remainder:
    """
    Single input, multi output node, skipping all outputs except for one
    """

    def __init__(self, input_name: str = "value", divisor: int = 2):
        self.input_name = input_name
        self.divisor = divisor
        # Contract
        self.init_parameters = {"input_name": input_name, "divisor": divisor}
        self.inputs = [input_name]
        self.outputs = [f"remainder_is_{remainder}" for remainder in range(divisor)]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        divisor = parameters.get(name, {}).get("divisor", self.divisor)
        for _, value in data:
            remainder = value % divisor
        return {f"remainder_is_{remainder}": value}


@node
class Enumerate:
    """
    Single input, multi output node, returning on all output edges
    """

    def __init__(self, input_name: str, outputs_count: int):
        self.input_name = input_name
        self.outputs_count = outputs_count
        # Contract
        self.init_parameters = {
            "input_name": input_name,
            "outputs_count": outputs_count,
        }
        self.inputs = [input_name]
        self.outputs = [str(out) for out in range(outputs_count)]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        if len(data) != 1:
            raise ValueError("Enumerate takes one single input.")

        output = {str(value): data[0][1] for value in range(len(self.outputs))}
        return output


@node
class Greet:
    """
    Single input single output no-op node.
    """

    def __init__(self, edge: str = "any", message: str = "Hi!"):
        self.message = message
        # Contract
        self.init_parameters = {"edge": edge, "message": message}
        self.inputs = [edge]
        self.outputs = [edge]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        message = parameters.get(name, {}).get("message", self.message)
        print("\n#################################")
        print(message)
        print("#################################\n")
        return {data[0][0]: data[0][1]}


@node
class Rename:
    """
    Single input single output rename node
    """

    def __init__(self, input_name: str, output_name: str):
        # Contract
        self.init_parameters = {"input_name": input_name, "output_name": output_name}
        self.inputs = [input_name]
        self.outputs = [output_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        return {self.outputs[0]: data[0][1]}


@node
class Accumulate:
    """
    Stateful reusable node
    """

    def __init__(self, edge: str):
        self.sum = 0
        # Contract
        self.init_parameters = {"edge": edge}
        self.inputs = [edge]
        self.outputs = [edge]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        for _, value in data:
            self.sum += value
        return {data[0][0]: data[0][1]}


@node
class Replicate:
    """
    Replicates the input data on all given output edges
    """

    def __init__(self, input_value: str = "value", outputs: List[str] = []):
        self.outputs = outputs
        # Contract
        self.init_parameters = {"input_value": input_value, "outputs": outputs}
        self.inputs = [input_value]
        self.outputs = outputs

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        return {output: data[0][1] for output in self.outputs}


@node
class Below:
    def __init__(
        self,
        threshold: int = 10,
        input_name: str = "value",
        output_above: str = "above",
        output_below: str = "below",
    ):
        self.threshold = threshold
        self.output_above = output_above
        self.output_below = output_below

        # Contract
        self.init_parameters = {
            "threshold": threshold,
            "input_name": input_name,
            "output_above": output_above,
            "output_below": output_below,
        }
        self.inputs = [input_name]
        self.outputs = [output_above, output_below]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        if len(data) != 1:
            raise ValueError("Below takes one input value only")

        if data[0][1] < self.threshold:
            return {self.output_below: data[0][1]}
        else:
            return {self.output_above: data[0][1]}


@node
class Merge:
    """
    Returns one single output on the output edge, which corresponds to the value of the last input edge that is not None.
    If no input edges received any value, returns None as well.
    """

    def __init__(
        self,
        inputs_name: str = "value",
        inputs_count: int = 2,
        output_name: str = "value",
    ):
        self.output_name = output_name
        # Contract
        self.init_parameters = {
            "inputs_count": inputs_count,
            "inputs_name": inputs_name,
            "output_name": output_name,
        }
        self.inputs = [inputs_name] * inputs_count
        self.outputs = [output_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        output = None
        for _, value in data:
            if value is not None:
                output = value

        return {self.output_name: output}


@node
class Double:
    def __init__(self, input_name: str = "value", output_name: str = "value"):
        # Contract
        self.init_parameters = {"input_name": input_name, "output_name": output_name}
        self.inputs = [input_name]
        self.outputs = [output_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        for _, value in data:
            value *= 2

        return {self.outputs[0]: value}


def test_complex_pipeline(tmp_path):
    accumulate = Accumulate(edge="value")

    pipeline = Pipeline(max_loops_allowed=4)
    pipeline.add_node("greet_first", Greet(edge="value", message="Hello!"))
    pipeline.add_node("accumulate_1", accumulate)
    pipeline.add_node("add_two", AddValue(add=2))
    pipeline.add_node("parity_check", Remainder(divisor=2))
    pipeline.add_node("add_one", AddValue(add=1))
    pipeline.add_node("accumulate_2", accumulate)

    pipeline.add_node("rename_even_to_value", Rename(input_name="remainder_is_0", output_name="value"))
    pipeline.add_node("rename_odd_to_value", Rename(input_name="remainder_is_1", output_name="value"))
    pipeline.add_node("rename_0_to_value", Rename(input_name="0", output_name="value"))
    pipeline.add_node("rename_1_to_value", Rename(input_name="1", output_name="value"))
    pipeline.add_node("rename_above_to_value", Rename(input_name="above", output_name="value"))

    pipeline.add_node("loop_merger", Merge(inputs_name="value", inputs_count=2))
    pipeline.add_node("below_10", Below())
    pipeline.add_node("double", Double(input_name="below", output_name="value"))

    pipeline.add_node("greet_again", Greet(edge="value", message="Hello again!"))
    pipeline.add_node("sum", Sum(inputs_name="value", inputs_count=3))

    pipeline.add_node("greet_enumerator", Greet(edge="value", message="Hello from enumerator!"))
    pipeline.add_node("enumerate", Enumerate(input_name="value", outputs_count=2))
    pipeline.add_node("add_three", AddValue(add=3))

    pipeline.add_node("diff", Subtract(first_value="value", second_value="sum"))
    pipeline.add_node("greet_one_last_time", Greet(edge="diff", message="Bye bye!"))
    pipeline.add_node("replicate", Replicate(input_value="diff", outputs=["first", "second"]))
    pipeline.add_node("add_five", AddValue(add=5, input_name="first"))
    pipeline.add_node("add_four", AddValue(add=4, input_name="second"))
    pipeline.add_node("accumulate_3", accumulate)

    pipeline.connect(
        [
            "greet_first",
            "accumulate_1",
            "add_two",
            "parity_check.remainder_is_0",
            "rename_even_to_value",
            "greet_again",
            "sum",
            "diff",
            "greet_one_last_time",
            "replicate.first",
            "add_five",
        ]
    )
    pipeline.connect(["replicate.second", "add_four", "accumulate_3"])
    pipeline.connect(
        [
            "parity_check.remainder_is_1",
            "rename_odd_to_value",
            "add_one",
            "loop_merger",
            "below_10.below",
            "double",
            "loop_merger",
        ]
    )
    pipeline.connect(["below_10.above", "rename_above_to_value", "accumulate_2", "diff"])
    pipeline.connect(["greet_enumerator", "enumerate.1", "rename_1_to_value", "sum"])
    pipeline.connect(["enumerate.0", "rename_0_to_value", "add_three", "sum"])
    pipeline.draw(tmp_path / "complex_pipeline.png")

    results = pipeline.run({"value": 1})
    pprint(results)
    print("accumulated: ", accumulate.sum)

    assert results == {"add_five": [{"value": 16}], "accumulate_3": [{"value": 15}]}
    assert accumulate.sum == 32


if __name__ == "__main__":
    test_complex_pipeline(Path(__file__).parent)
