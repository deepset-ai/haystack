# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.components.others import Multiplexer
from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import Accumulate, AddFixedValue, Threshold

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    pipeline = Pipeline(max_loops_allowed=10)
    add_one = AddFixedValue(add=1)
    multiplexer = Multiplexer(type_=int)
    below_10 = Threshold(threshold=10)
    below_5 = Threshold(threshold=5)
    add_three = AddFixedValue(add=3)
    accumulator = Accumulate()
    add_two = AddFixedValue(add=2)
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("multiplexer", multiplexer)
    pipeline.add_component("below_10", below_10)
    pipeline.add_component("below_5", below_5)
    pipeline.add_component("add_three", add_three)
    pipeline.add_component("accumulator", accumulator)
    pipeline.add_component("add_two", add_two)

    pipeline.connect(add_one.outputs.result, multiplexer.inputs.value)
    pipeline.connect(multiplexer.outputs.value, below_10.inputs.value)
    pipeline.connect(below_10.outputs.below, accumulator.inputs.value)
    pipeline.connect(accumulator.outputs.value, below_5.inputs.value)
    pipeline.connect(below_5.outputs.above, add_three.inputs.value)
    pipeline.connect(below_5.outputs.below, multiplexer.inputs.value)
    pipeline.connect(add_three.outputs.result, multiplexer.inputs.value)
    pipeline.connect(below_10.outputs.above, add_two.inputs.value)

    pipeline.draw(tmp_path / "double_loop_pipeline.png")

    results = pipeline.run({"add_one": {"value": 3}})

    assert results == {"add_two": {"result": 13}}
    assert accumulator.state == 8
