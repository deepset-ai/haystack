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
    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("multiplexer.value", "remainder.value")
    pipeline.connect("remainder.remainder_is_1", "add_two.value")
    pipeline.connect("remainder.remainder_is_2", "add_one.value")
    pipeline.connect("add_two", "multiplexer.value")
    pipeline.connect("add_one", "multiplexer.value")

    pipeline.draw(Path(__file__).parent / Path(__file__).name.replace(".py", ".png"))

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
    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("add_two_1", AddFixedValue(add=1))
    pipeline.add_component("add_two_2", AddFixedValue(add=1))

    pipeline.connect("multiplexer.value", "remainder.value")
    pipeline.connect("remainder.remainder_is_1", "add_two_1.value")
    pipeline.connect("add_two_1", "add_two_2.value")
    pipeline.connect("add_two_2", "multiplexer")
    pipeline.connect("remainder.remainder_is_2", "add_one.value")
    pipeline.connect("add_one", "multiplexer")

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
    pipeline.add_component("multiplexer", Multiplexer(type_=int))
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("add_two_1", AddFixedValue(add=1))
    pipeline.add_component("add_two_2", AddFixedValue(add=1))

    pipeline.connect("multiplexer", "remainder.value")
    pipeline.connect("remainder.remainder_is_1", "add_two_1.value")
    pipeline.connect("add_two_1", "add_two_2.value")
    pipeline.connect("add_two_2", "multiplexer.value")
    pipeline.connect("remainder.remainder_is_2", "add_one.value")
    pipeline.connect("add_one", "multiplexer.value")

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
