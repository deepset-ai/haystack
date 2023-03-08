from pathlib import Path
from pprint import pprint
import logging

from canals.pipeline import Pipeline
from tests.nodes import Accumulate, AddValue, Greet, Remainder, Rename, Merge, Below, Double, Sum, Repeat, Subtract

logging.basicConfig(level=logging.DEBUG)


def test_complex_pipeline(tmp_path):
    accumulate = Accumulate(edge="value")

    pipeline = Pipeline(max_loops_allowed=4)
    pipeline.add_node("greet_first", Greet(edge="value", message="Hello!"))
    pipeline.add_node("accumulate_1", accumulate)
    pipeline.add_node("add_two", AddValue(add=2))
    pipeline.add_node("parity_check", Remainder(divisor=2))
    pipeline.add_node("add_one", AddValue(add=1, input="1"))
    pipeline.add_node("accumulate_2", accumulate)
    pipeline.add_node("rename_above_to_value", Rename(input="above", output="value"))

    pipeline.add_node("loop_merger", Merge(inputs=["value", "value"]))
    pipeline.add_node("below_10", Below(threshold=10))
    pipeline.add_node("double", Double(input="below", output="value"))

    pipeline.add_node("greet_again", Greet(edge="0", message="Hello again!"))
    pipeline.add_node("sum", Sum(inputs=["1", "0", "value"]))

    pipeline.add_node("greet_enumerator", Greet(edge="value", message="Hello from enumerator!"))
    pipeline.add_node("enumerate", Repeat(input="value", outputs=["0", "1"]))
    pipeline.add_node("add_three", AddValue(add=3, input="0"))

    pipeline.add_node("diff", Subtract(first_input="value", second_input="sum"))
    pipeline.add_node("greet_one_last_time", Greet(edge="diff", message="Bye bye!"))
    pipeline.add_node("replicate", Repeat(input="diff", outputs=["first", "second"]))
    pipeline.add_node("add_five", AddValue(add=5, input="first"))
    pipeline.add_node("add_four", AddValue(add=4, input="second"))
    pipeline.add_node("accumulate_3", accumulate)

    pipeline.connect("greet_first", "accumulate_1")
    pipeline.connect("accumulate_1", "add_two")
    pipeline.connect("add_two", "parity_check")

    pipeline.connect("parity_check.0", "greet_again")
    pipeline.connect("greet_again", "sum")
    pipeline.connect("sum", "diff")
    pipeline.connect("diff", "greet_one_last_time")
    pipeline.connect("greet_one_last_time", "replicate")
    pipeline.connect("replicate.first", "add_five")
    pipeline.connect("replicate.second", "add_four")
    pipeline.connect("add_four", "accumulate_3")

    pipeline.connect("parity_check.1", "add_one")
    pipeline.connect("add_one", "loop_merger")
    pipeline.connect("loop_merger", "below_10")

    pipeline.connect("below_10.below", "double")
    pipeline.connect("double", "loop_merger")

    pipeline.connect("below_10.above", "rename_above_to_value")
    pipeline.connect("rename_above_to_value", "accumulate_2")
    pipeline.connect("accumulate_2", "diff")

    pipeline.connect("greet_enumerator", "enumerate")
    pipeline.connect("enumerate.1", "sum")

    pipeline.connect("enumerate.0", "add_three")
    pipeline.connect("add_three", "sum")

    pipeline.draw(tmp_path / "complex_pipeline.png")

    results = pipeline.run({"value": 1})
    pprint(results)
    print("accumulated: ", accumulate.state)

    assert results == {"add_five": [{"value": 16}], "accumulate_3": [{"value": 15}]}
    assert accumulate.state == 32


if __name__ == "__main__":
    test_complex_pipeline(Path(__file__).parent)
