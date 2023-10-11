# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from sample_components import AddFixedValue, Remainder, Double, Sum

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("add_one", AddFixedValue())
    pipeline.add_component("parity", Remainder(divisor=2))
    pipeline.add_component("add_ten", AddFixedValue(add=10))
    pipeline.add_component("double", Double())
    pipeline.add_component("add_four", AddFixedValue(add=4))
    pipeline.add_component("add_one_again", AddFixedValue())
    pipeline.add_component("sum", Sum(inputs=["in_1", "in_2", "in_3", "in_4"]))

    pipeline.connect("add_one.result", "parity.value")
    pipeline.connect("parity.remainder_is_0", "add_ten.value")
    pipeline.connect("parity.remainder_is_1", "double.value")
    pipeline.connect("add_one.result", "sum.in_1")
    pipeline.connect("add_ten.result", "sum.in_2")
    pipeline.connect("double.value", "sum.in_3")
    pipeline.connect("parity.remainder_is_1", "add_four.value")
    pipeline.connect("add_four.result", "add_one_again.value")
    pipeline.connect("add_one_again.result", "sum.in_4")

    pipeline.draw(tmp_path / "variable_decision_and_merge_pipeline.png")

    results = pipeline.run({"add_one": {"value": 1}})
    pprint(results)
    assert results == {"sum": {"total": 14}}

    results = pipeline.run({"add_one": {"value": 2}})
    pprint(results)
    assert results == {"sum": {"total": 17}}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
