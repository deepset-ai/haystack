from pathlib import Path
from pprint import pprint
import logging

from canals.pipeline import Pipeline
from test.test_components import (
    Accumulate,
    AddFixedValue,
    Greet,
    Parity,
    Threshold,
    Double,
    Sum,
    Repeat,
    Subtract,
    MergeLoop,
)

logging.basicConfig(level=logging.DEBUG)


def test_complex_pipeline(tmp_path):
    accumulate = Accumulate()

    pipeline = Pipeline(max_loops_allowed=4)
    pipeline.add_component("greet_first", Greet(message="Hello, the value is {value}."))
    pipeline.add_component("accumulate_1", accumulate)
    pipeline.add_component("add_two", AddFixedValue(add=2))
    pipeline.add_component("parity", Parity())
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("accumulate_2", accumulate)

    pipeline.add_component("loop_merger", MergeLoop(expected_type=int))
    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("double", Double())

    pipeline.add_component("greet_again", Greet(message="Hello again, now the value is {value}."))
    pipeline.add_component("sum", Sum())

    pipeline.add_component("greet_enumerator", Greet(message="Hello from enumerator, here the value became {value}."))
    pipeline.add_component("enumerate", Repeat(outputs=["first", "second"]))
    pipeline.add_component("add_three", AddFixedValue(add=3))

    pipeline.add_component("diff", Subtract())
    pipeline.add_component("greet_one_last_time", Greet(message="Bye bye! The value here is {value}!"))
    pipeline.add_component("replicate", Repeat(outputs=["first", "second"]))
    pipeline.add_component("add_five", AddFixedValue(add=5))
    pipeline.add_component("add_four", AddFixedValue(add=4))
    pipeline.add_component("accumulate_3", accumulate)

    pipeline.connect("greet_first", "accumulate_1")
    pipeline.connect("accumulate_1", "add_two")
    pipeline.connect("add_two", "parity")

    pipeline.connect("parity.even", "greet_again")
    pipeline.connect("greet_again", "sum")
    pipeline.connect("sum", "diff.first_value")
    pipeline.connect("diff", "greet_one_last_time")
    pipeline.connect("greet_one_last_time", "replicate")
    pipeline.connect("replicate.first", "add_five.value")
    pipeline.connect("replicate.second", "add_four.value")
    pipeline.connect("add_four", "accumulate_3")

    pipeline.connect("parity.odd", "add_one.value")
    pipeline.connect("add_one", "loop_merger")
    pipeline.connect("loop_merger", "below_10")

    pipeline.connect("below_10.below", "double")
    pipeline.connect("double", "loop_merger")

    pipeline.connect("below_10.above", "accumulate_2")
    pipeline.connect("accumulate_2", "diff.second_value")

    pipeline.connect("greet_enumerator", "enumerate")
    pipeline.connect("enumerate.second", "sum")

    pipeline.connect("enumerate.first", "add_three.value")
    pipeline.connect("add_three", "sum")

    pipeline.draw(tmp_path / "complex_pipeline.png")

    results = pipeline.run({"greet_first": {"value": 1}, "greet_enumerator": {"value": 1}})
    pprint(results)
    print("accumulated: ", accumulate.state)

    assert results == {"accumulate_3": {"value": 9}, "add_five": {"value": -7}}
    assert accumulate.state == 9


if __name__ == "__main__":
    test_complex_pipeline(Path(__file__).parent)
