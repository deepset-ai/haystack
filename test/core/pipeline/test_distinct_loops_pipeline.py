# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path

from haystack.components.others import Multiplexer
from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import AddFixedValue, Remainder

logging.basicConfig(level=logging.DEBUG)


def test_pipeline_equally_long_branches():
    pipeline = Pipeline(max_loops_allowed=10)
    multiplexer = Multiplexer(type_=int)
    remainder = Remainder(divisor=3)
    add_one = AddFixedValue(add=1)
    add_two = AddFixedValue(add=2)
    pipeline.add_component("multiplexer", multiplexer)
    pipeline.add_component("remainder", remainder)
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("add_two", add_two)

    pipeline.connect(multiplexer.outputs.value, remainder.inputs.value)
    pipeline.connect(remainder.outputs.remainder_is_1, add_two.inputs.value)
    pipeline.connect(remainder.outputs.remainder_is_2, add_one.inputs.value)
    pipeline.connect(add_two.outputs.result, multiplexer.inputs.value)
    pipeline.connect(add_one.outputs.result, multiplexer.inputs.value)

    results = pipeline.run({"multiplexer": {"value": 0}})
    assert results == {"remainder": {"remainder_is_0": 0}}

    results = pipeline.run({"multiplexer": {"value": 3}})
    assert results == {"remainder": {"remainder_is_0": 3}}

    results = pipeline.run({"multiplexer": {"value": 4}})
    assert results == {"remainder": {"remainder_is_0": 6}}

    results = pipeline.run({"multiplexer": {"value": 5}})
    assert results == {"remainder": {"remainder_is_0": 6}}

    results = pipeline.run({"multiplexer": {"value": 6}})
    assert results == {"remainder": {"remainder_is_0": 6}}


def test_pipeline_differing_branches():
    pipeline = Pipeline(max_loops_allowed=10)
    multiplexer = Multiplexer(type_=int)
    remainder = Remainder(divisor=3)
    add_one = AddFixedValue(add=1)
    add_two_1 = AddFixedValue(add=1)
    add_two_2 = AddFixedValue(add=1)
    pipeline.add_component("multiplexer", multiplexer)
    pipeline.add_component("remainder", remainder)
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("add_two_1", add_two_1)
    pipeline.add_component("add_two_2", add_two_2)

    pipeline.connect(multiplexer.outputs.value, remainder.inputs.value)
    pipeline.connect(remainder.outputs.remainder_is_1, add_two_1.inputs.value)
    pipeline.connect(add_two_1.outputs.result, add_two_2.inputs.value)
    pipeline.connect(add_two_2.outputs.result, multiplexer.inputs.value)
    pipeline.connect(remainder.outputs.remainder_is_2, add_one.inputs.value)
    pipeline.connect(add_one.outputs.result, multiplexer.inputs.value)

    results = pipeline.run({"multiplexer": {"value": 0}})
    assert results == {"remainder": {"remainder_is_0": 0}}

    results = pipeline.run({"multiplexer": {"value": 3}})
    assert results == {"remainder": {"remainder_is_0": 3}}

    results = pipeline.run({"multiplexer": {"value": 4}})
    assert results == {"remainder": {"remainder_is_0": 6}}

    results = pipeline.run({"multiplexer": {"value": 5}})
    assert results == {"remainder": {"remainder_is_0": 6}}

    results = pipeline.run({"multiplexer": {"value": 6}})
    assert results == {"remainder": {"remainder_is_0": 6}}


def test_pipeline_differing_branches_variadic():
    pipeline = Pipeline(max_loops_allowed=10)
    multiplexer = Multiplexer(type_=int)
    remainder = Remainder(divisor=3)
    add_one = AddFixedValue(add=1)
    add_two_1 = AddFixedValue(add=1)
    add_two_2 = AddFixedValue(add=1)
    pipeline.add_component("multiplexer", multiplexer)
    pipeline.add_component("remainder", remainder)
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("add_two_1", add_two_1)
    pipeline.add_component("add_two_2", add_two_2)

    pipeline.connect(multiplexer.outputs.value, remainder.inputs.value)
    pipeline.connect(remainder.outputs.remainder_is_1, add_two_1.inputs.value)
    pipeline.connect(add_two_1.outputs.result, add_two_2.inputs.value)
    pipeline.connect(add_two_2.outputs.result, multiplexer.inputs.value)
    pipeline.connect(remainder.outputs.remainder_is_2, add_one.inputs.value)
    pipeline.connect(add_one.outputs.result, multiplexer.inputs.value)

    results = pipeline.run({"multiplexer": {"value": 0}})
    assert results == {"remainder": {"remainder_is_0": 0}}

    results = pipeline.run({"multiplexer": {"value": 3}})
    assert results == {"remainder": {"remainder_is_0": 3}}

    results = pipeline.run({"multiplexer": {"value": 4}})
    assert results == {"remainder": {"remainder_is_0": 6}}

    results = pipeline.run({"multiplexer": {"value": 5}})
    assert results == {"remainder": {"remainder_is_0": 6}}

    results = pipeline.run({"multiplexer": {"value": 6}})
    assert results == {"remainder": {"remainder_is_0": 6}}
