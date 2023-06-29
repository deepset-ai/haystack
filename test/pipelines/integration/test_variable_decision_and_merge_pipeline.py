# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.sample_components import AddFixedValue, Remainder, Double, Sum

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    add_one = AddFixedValue()
    summer = Sum(inputs=["in_1", "in_2", "in_3", "in_4"])

    pipeline = Pipeline()
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("parity", Remainder())
    pipeline.add_component("add_ten", AddFixedValue(add=10))
    pipeline.add_component("double", Double())
    pipeline.add_component("add_four", AddFixedValue(add=4))
    pipeline.add_component("add_one_again", add_one)
    pipeline.add_component("sum", summer)

    pipeline.connect("add_one", "parity")
    pipeline.connect("parity.remainder_is_0", "add_ten.value")
    pipeline.connect("parity.remainder_is_1", "double")
    pipeline.connect("add_one", "sum.in_1")
    pipeline.connect("add_ten", "sum.in_2")
    pipeline.connect("double", "sum.in_3")
    pipeline.connect("parity.remainder_is_1", "add_four.value")
    pipeline.connect("add_four", "add_one_again")
    pipeline.connect("add_one_again", "sum.in_4")

    pipeline.draw(tmp_path / "variable_decision_and_merge_pipeline.png")

    results = pipeline.run({"add_one": add_one.input(value=1)})
    pprint(results)
    assert results == {"sum": summer.output(total=14)}

    results = pipeline.run({"add_one": add_one.input(value=2)})
    pprint(results)
    assert results == {"sum": summer.output(total=17)}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
