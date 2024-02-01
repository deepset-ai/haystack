# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.components.others import Multiplexer
from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import (
    Accumulate,
    AddFixedValue,
    Double,
    Greet,
    Parity,
    Repeat,
    Subtract,
    Sum,
    Threshold,
)

logging.basicConfig(level=logging.DEBUG)


def test_complex_pipeline():
    pipeline = Pipeline(max_loops_allowed=2)

    greet_first = Greet(message="Hello, the value is {value}.")
    accumulate_1 = Accumulate()
    add_two = AddFixedValue(add=2)
    parity = Parity()
    add_one = AddFixedValue(add=1)
    accumulate_2 = Accumulate()
    multiplexer = Multiplexer(type_=int)
    below_10 = Threshold(threshold=10)
    double = Double()
    greet_again = Greet(message="Hello again, now the value is {value}.")
    sum = Sum()
    greet_enumerator = Greet(message="Hello from enumerator, here the value became {value}.")
    enumerate = Repeat(outputs=["first", "second"])
    add_three = AddFixedValue(add=3)
    diff = Subtract()
    greet_one_last_time = Greet(message="Bye bye! The value here is {value}!")
    replicate = Repeat(outputs=["first", "second"])
    add_five = AddFixedValue(add=5)
    add_four = AddFixedValue(add=4)
    accumulate_3 = Accumulate()

    pipeline.add_component("greet_first", greet_first)
    pipeline.add_component("accumulate_1", accumulate_1)
    pipeline.add_component("add_two", add_two)
    pipeline.add_component("parity", parity)
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("accumulate_2", accumulate_2)

    pipeline.add_component("multiplexer", multiplexer)
    pipeline.add_component("below_10", below_10)
    pipeline.add_component("double", double)

    pipeline.add_component("greet_again", greet_again)
    pipeline.add_component("sum", sum)

    pipeline.add_component("greet_enumerator", greet_enumerator)
    pipeline.add_component("enumerate", enumerate)
    pipeline.add_component("add_three", add_three)

    pipeline.add_component("diff", diff)
    pipeline.add_component("greet_one_last_time", greet_one_last_time)
    pipeline.add_component("replicate", replicate)
    pipeline.add_component("add_five", add_five)
    pipeline.add_component("add_four", add_four)
    pipeline.add_component("accumulate_3", accumulate_3)

    pipeline.connect(greet_first.outputs.value, accumulate_1.inputs.value)
    pipeline.connect(accumulate_1.outputs.value, add_two.inputs.value)
    pipeline.connect(add_two.outputs.result, parity.inputs.value)

    pipeline.connect(parity.outputs.even, greet_again.inputs.value)
    pipeline.connect(greet_again.outputs.value, sum.inputs.values)
    pipeline.connect(sum.outputs.total, diff.inputs.first_value)
    pipeline.connect(diff.outputs.difference, greet_one_last_time.inputs.value)
    pipeline.connect(greet_one_last_time.outputs.value, replicate.inputs.value)
    pipeline.connect(replicate.outputs.first, add_five.inputs.value)
    pipeline.connect(replicate.outputs.second, add_four.inputs.value)
    pipeline.connect(add_four.outputs.result, accumulate_3.inputs.value)

    pipeline.connect(parity.outputs.odd, add_one.inputs.value)
    pipeline.connect(add_one.outputs.result, multiplexer.inputs.value)
    pipeline.connect(multiplexer.outputs.value, below_10.inputs.value)

    pipeline.connect(below_10.outputs.below, double.inputs.value)
    pipeline.connect(double.outputs.value, multiplexer.inputs.value)

    pipeline.connect(below_10.outputs.above, accumulate_2.inputs.value)
    pipeline.connect(accumulate_2.outputs.value, diff.inputs.second_value)

    pipeline.connect(greet_enumerator.outputs.value, enumerate.inputs.value)
    pipeline.connect(enumerate.outputs.second, sum.inputs.values)

    pipeline.connect(enumerate.outputs.first, add_three.inputs.value)
    pipeline.connect(add_three.outputs.result, sum.inputs.values)

    results = pipeline.run({"greet_first": {"value": 1}, "greet_enumerator": {"value": 1}})
    assert results == {"accumulate_3": {"value": -7}, "add_five": {"result": -6}}
