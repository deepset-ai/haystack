# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path

from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import AddFixedValue, Double, Parity, Subtract

logging.basicConfig(level=logging.DEBUG)


def test_pipeline():
    pipeline = Pipeline()
    add_one = AddFixedValue()
    parity = Parity()
    add_ten = AddFixedValue(add=10)
    double = Double()
    add_four = AddFixedValue(add=4)
    add_two = AddFixedValue()
    add_two_as_well = AddFixedValue()
    diff = Subtract()
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("parity", parity)
    pipeline.add_component("add_ten", add_ten)
    pipeline.add_component("double", double)
    pipeline.add_component("add_four", add_four)
    pipeline.add_component("add_two", add_two)
    pipeline.add_component("add_two_as_well", add_two_as_well)
    pipeline.add_component("diff", diff)

    pipeline.connect(add_one.outputs.result, parity.inputs.value)
    pipeline.connect(parity.outputs.even, add_four.inputs.value)
    pipeline.connect(parity.outputs.odd, double.inputs.value)
    pipeline.connect(add_ten.outputs.result, diff.inputs.first_value)
    pipeline.connect(double.outputs.value, diff.inputs.second_value)
    pipeline.connect(parity.outputs.odd, add_ten.inputs.value)
    pipeline.connect(add_four.outputs.result, add_two.inputs.value)
    pipeline.connect(add_four.outputs.result, add_two_as_well.inputs.value)

    results = pipeline.run({"add_one": {"value": 1}, "add_two": {"add": 2}, "add_two_as_well": {"add": 2}})
    assert results == {"add_two": {"result": 8}, "add_two_as_well": {"result": 8}}

    results = pipeline.run({"add_one": {"value": 2}, "add_two": {"add": 2}, "add_two_as_well": {"add": 2}})
    assert results == {"diff": {"difference": 7}}
