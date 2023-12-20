# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.core.component import component
from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import AddFixedValue, SelfLoop

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline_one_node():
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("self_loop", SelfLoop())
    pipeline.connect("self_loop.current_value", "self_loop.values")

    results = pipeline.run({"self_loop": {"values": 5}})
    assert results["self_loop"]["final_result"] == 0


def test_pipeline():
    pipeline = Pipeline(max_loops_allowed=10)
    pipeline.add_component("add_1", AddFixedValue())
    pipeline.add_component("self_loop", SelfLoop())
    pipeline.add_component("add_2", AddFixedValue())
    pipeline.connect("add_1", "self_loop.values")
    pipeline.connect("self_loop.current_value", "self_loop.values")
    pipeline.connect("self_loop.final_result", "add_2.value")

    results = pipeline.run({"add_1": {"value": 5}})
    assert results["add_2"]["result"] == 1
