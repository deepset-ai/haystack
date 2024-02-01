# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import AddFixedValue, Double, Repeat

logging.basicConfig(level=logging.DEBUG)


def test_pipeline():
    pipeline = Pipeline()
    add_one = AddFixedValue(add=1)
    repeat = Repeat(outputs=["first", "second"])
    add_ten = AddFixedValue(add=10)
    double = Double()
    add_three = AddFixedValue(add=3)
    add_one_again = AddFixedValue(add=1)
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("repeat", repeat)
    pipeline.add_component("add_ten", add_ten)
    pipeline.add_component("double", double)
    pipeline.add_component("add_three", add_three)
    pipeline.add_component("add_one_again", add_one_again)

    pipeline.connect(add_one.outputs.result, repeat.inputs.value)
    pipeline.connect(repeat.outputs.first, add_ten.inputs.value)
    pipeline.connect(repeat.outputs.second, double.inputs.value)
    pipeline.connect(repeat.outputs.second, add_three.inputs.value)
    pipeline.connect(add_three.outputs.result, add_one_again.inputs.value)

    results = pipeline.run({"add_one": {"value": 1}})
    assert results == {"add_one_again": {"result": 6}, "add_ten": {"result": 12}, "double": {"value": 4}}
