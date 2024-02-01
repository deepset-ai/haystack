# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import AddFixedValue, Double, Remainder, Sum

logging.basicConfig(level=logging.DEBUG)


def test_pipeline():
    pipeline = Pipeline()
    add_one = AddFixedValue()
    parity = Remainder(divisor=2)
    add_ten = AddFixedValue(add=10)
    double = Double()
    add_four = AddFixedValue(add=4)
    add_one_again = AddFixedValue()
    sum = Sum()
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("parity", parity)
    pipeline.add_component("add_ten", add_ten)
    pipeline.add_component("double", double)
    pipeline.add_component("add_four", add_four)
    pipeline.add_component("add_one_again", add_one_again)
    pipeline.add_component("sum", sum)

    pipeline.connect(add_one.outputs.result, parity.inputs.value)
    pipeline.connect(parity.outputs.remainder_is_0, add_ten.inputs.value)
    pipeline.connect(parity.outputs.remainder_is_1, double.inputs.value)
    pipeline.connect(add_one.outputs.result, sum.inputs.values)
    pipeline.connect(add_ten.outputs.result, sum.inputs.values)
    pipeline.connect(double.outputs.value, sum.inputs.values)
    pipeline.connect(parity.outputs.remainder_is_1, add_four.inputs.value)
    pipeline.connect(add_four.outputs.result, add_one_again.inputs.value)
    pipeline.connect(add_one_again.outputs.result, sum.inputs.values)

    results = pipeline.run({"add_one": {"value": 1}})
    assert results == {"sum": {"total": 14}}

    results = pipeline.run({"add_one": {"value": 2}})
    assert results == {"sum": {"total": 17}}
