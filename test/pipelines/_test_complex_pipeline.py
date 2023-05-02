from pathlib import Path
from pprint import pprint
import logging

from canals.pipeline import Pipeline
from test.components import Accumulate, AddValue, Greet, Parity, Threshold, Double, Sum, Repeat, Subtract

logging.basicConfig(level=logging.DEBUG)


def test_complex_pipeline(tmp_path):
    accumulate = Accumulate()

    pipeline = Pipeline(max_loops_allowed=4)
    pipeline.add_component("greet_first", Greet(message="Hello, the value is {value}."))
    pipeline.add_component("accumulate_1", accumulate)
    pipeline.add_component("add_two", AddValue(add=2))
    pipeline.add_component("parity", Parity())
    pipeline.add_component("add_one", AddValue(add=1))
    pipeline.add_component("accumulate_2", accumulate)

    pipeline.add_component("below_10", Threshold(threshold=10))
    pipeline.add_component("double", Double())

    pipeline.add_component("greet_again", Greet(message="Hello again, now the value is {value}."))
    pipeline.add_component("sum", Sum())

    pipeline.add_component("greet_enumerator", Greet(message="Hello from enumerator, here the value became {value}."))
    pipeline.add_component("enumerate", Repeat(repeat=2))
    pipeline.add_component("add_three", AddValue(add=3))

    pipeline.add_component("diff", Subtract())
    pipeline.add_component("greet_one_last_time", Greet(message="Bye bye! The value here is {value}!"))
    pipeline.add_component("replicate", Repeat(repeat=2))
    pipeline.add_component("add_five", AddValue(add=5))
    pipeline.add_component("add_four", AddValue(add=4))
    pipeline.add_component("accumulate_3", accumulate)

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

    try:
        pipeline.draw(tmp_path / "complex_pipeline.png")
    except ImportError:
        logging.warning("pygraphviz not found, pipeline is not being drawn.")

    results = pipeline.run({"value": 1})
    pprint(results)
    print("accumulated: ", accumulate.state)

    assert results == {"add_five": [{"value": 16}], "accumulate_3": [{"value": 15}]}
    assert accumulate.state == 32


if __name__ == "__main__":
    test_complex_pipeline(Path(__file__).parent)
