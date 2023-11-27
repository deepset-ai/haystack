# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from pathlib import Path
from pprint import pprint

from canals import component
from canals.pipeline import Pipeline
from sample_components import AddFixedValue, SelfLoop

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline_one_node(tmp_path):
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("self_loop", SelfLoop())
    pipeline.connect("self_loop.current_value", "self_loop.values")

    pipeline.draw(tmp_path / "self_looping_pipeline_one_node.png")

    results = pipeline.run({"self_loop": {"values": 5}})
    pprint(results)

    assert results["self_loop"]["final_result"] == 0


def test_pipeline(tmp_path):
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("add_1", AddFixedValue())
    pipeline.add_component("self_loop", SelfLoop())
    pipeline.add_component("add_2", AddFixedValue())
    pipeline.connect("add_1", "self_loop.values")
    pipeline.connect("self_loop.current_value", "self_loop.values")
    pipeline.connect("self_loop.final_result", "add_2.value")

    pipeline.draw(tmp_path / "self_looping_pipeline.png")

    results = pipeline.run({"add_1": {"value": 5}})
    pprint(results)

    assert results["add_2"]["result"] == 1


if __name__ == "__main__":
    test_pipeline_one_node(Path(__file__).parent)
    test_pipeline(Path(__file__).parent)
