# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from canals.pipeline import Pipeline, marshal_pipelines, unmarshal_pipelines, PipelineConnectError, PipelineMaxLoops
from canals.component import Component
from test.sample_components import AddFixedValue, Greet, Threshold, MergeLoop

import logging

logging.basicConfig(level=logging.DEBUG)


def test_connect():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    pipe.connect("first", "second")

    assert list(pipe.graph.edges) == [("first", "second", "value/value")]


def test_connect_nonexisting_from_component():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    with pytest.raises(ValueError, match="Component named third not found in the pipeline"):
        pipe.connect("third", "second")


def test_connect_nonexisting_to_component():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    with pytest.raises(ValueError, match="Component named third not found in the pipeline"):
        pipe.connect("first", "third")


def test_connect_nonexisting_from_socket():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    with pytest.raises(PipelineConnectError, match="first.wrong does not exist"):
        pipe.connect("first.wrong", "second")


def test_connect_nonexisting_to_socket():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    with pytest.raises(PipelineConnectError, match="second.wrong does not exist"):
        pipe.connect("first", "second.wrong")


def test_connect_mismatched_components():
    add = AddFixedValue()
    greet = Greet()

    pipe = Pipeline()
    pipe.add_component("first", add)
    pipe.add_component("second", greet)
    with pytest.raises(
        PipelineConnectError, match="Cannot connect 'first' with 'second': no matching connections available."
    ):
        pipe.connect("first", "second.message")


def test_connect_many_outputs_to_the_same_input():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    pipe.add_component("third", add_2)
    pipe.connect("first.value", "second.value")
    with pytest.raises(PipelineConnectError, match="second.value is already connected to first"):
        pipe.connect("third.value", "second.value")


def test_max_loops():
    add = AddFixedValue()
    threshold = Threshold(threshold=100)
    merge = MergeLoop(expected_type=int)

    pipe = Pipeline(max_loops_allowed=10)
    pipe.add_component("add", add)
    pipe.add_component("threshold", threshold)
    pipe.add_component("merge", merge)
    pipe.connect("threshold.below", "add.value")
    pipe.connect("add.value", "merge.value_1")
    pipe.connect("merge.value", "threshold.value")
    with pytest.raises(PipelineMaxLoops):
        pipe.run({"merge": merge.input(value_2=1)})
