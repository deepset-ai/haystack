# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import AddFixedValue, SelfLoop

logging.basicConfig(level=logging.DEBUG)


def test_pipeline_one_node():
    pipeline = Pipeline(max_loops_allowed=10)
    self_loop = SelfLoop()
    pipeline.add_component("self_loop", self_loop)
    pipeline.connect(self_loop.outputs.current_value, self_loop.inputs.values)

    results = pipeline.run({"self_loop": {"values": 5}})
    assert results["self_loop"]["final_result"] == 0


def test_pipeline():
    pipeline = Pipeline(max_loops_allowed=10)
    add_1 = AddFixedValue()
    self_loop = SelfLoop()
    add_2 = AddFixedValue()
    pipeline.add_component("add_1", add_1)
    pipeline.add_component("self_loop", self_loop)
    pipeline.add_component("add_2", add_2)
    pipeline.connect(add_1.outputs.result, self_loop.inputs.values)
    pipeline.connect(self_loop.outputs.current_value, self_loop.inputs.values)
    pipeline.connect(self_loop.outputs.final_result, add_2.inputs.value)

    results = pipeline.run({"add_1": {"value": 5}})
    assert results["add_2"]["result"] == 1
