# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import *
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from sample_components import AddFixedValue, MergeLoop, Remainder, FirstIntSelector

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline_equally_long_branches(tmp_path):
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("merge", MergeLoop(expected_type=int, inputs=["in", "in_1", "in_2"]))
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("add_two", AddFixedValue(add=2))

    pipeline.connect("merge.value", "remainder.value")
    pipeline.connect("remainder.remainder_is_1", "add_two.value")
    pipeline.connect("remainder.remainder_is_2", "add_one.value")
    pipeline.connect("add_two", "merge.in_2")
    pipeline.connect("add_one", "merge.in_1")

    pipeline.draw(tmp_path / "distinct_loops_pipeline_same_branches.png")

    results = pipeline.run({"merge": {"in": 0}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 0}}

    results = pipeline.run({"merge": {"in": 3}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 3}}

    results = pipeline.run({"merge": {"in": 4}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 6}}

    results = pipeline.run({"merge": {"in": 5}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 6}}

    results = pipeline.run({"merge": {"in": 6}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 6}}


def test_pipeline_differing_branches(tmp_path):
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("merge", MergeLoop(expected_type=int, inputs=["in", "in_1", "in_2"]))
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("add_two_1", AddFixedValue(add=1))
    pipeline.add_component("add_two_2", AddFixedValue(add=1))

    pipeline.connect("merge.value", "remainder.value")
    pipeline.connect("remainder.remainder_is_1", "add_two_1.value")
    pipeline.connect("add_two_1", "add_two_2.value")
    pipeline.connect("add_two_2", "merge.in_2")
    pipeline.connect("remainder.remainder_is_2", "add_one.value")
    pipeline.connect("add_one", "merge.in_1")

    pipeline.draw(tmp_path / "distinct_loops_pipeline_different_branches.png")

    results = pipeline.run({"merge": {"in": 0}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 0}}

    results = pipeline.run({"merge": {"in": 3}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 3}}

    results = pipeline.run({"merge": {"in": 4}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 6}}

    results = pipeline.run({"merge": {"in": 5}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 6}}

    results = pipeline.run({"merge": {"in": 6}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 6}}


def test_pipeline_differing_branches_variadic(tmp_path):
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("merge", FirstIntSelector())
    pipeline.add_component("remainder", Remainder(divisor=3))
    pipeline.add_component("add_one", AddFixedValue(add=1))
    pipeline.add_component("add_two_1", AddFixedValue(add=1))
    pipeline.add_component("add_two_2", AddFixedValue(add=1))

    pipeline.connect("merge", "remainder.value")
    pipeline.connect("remainder.remainder_is_1", "add_two_1.value")
    pipeline.connect("add_two_1", "add_two_2.value")
    pipeline.connect("add_two_2", "merge.inputs")
    pipeline.connect("remainder.remainder_is_2", "add_one.value")
    pipeline.connect("add_one", "merge.inputs")

    pipeline.draw(tmp_path / "distinct_loops_pipeline_different_branches_variadic.png")

    results = pipeline.run({"merge": {"inputs": 0}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 0}}

    results = pipeline.run({"merge": {"inputs": 3}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 3}}

    results = pipeline.run({"merge": {"inputs": 4}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 6}}

    results = pipeline.run({"merge": {"inputs": 5}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 6}}

    results = pipeline.run({"merge": {"inputs": 6}})
    pprint(results)
    assert results == {"remainder": {"remainder_is_0": 6}}


if __name__ == "__main__":
    test_pipeline_equally_long_branches(Path(__file__).parent)
    test_pipeline_differing_branches(Path(__file__).parent)
    test_pipeline_differing_branches_variadic(Path(__file__).parent)
