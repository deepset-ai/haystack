# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List
import pytest

from canals import Pipeline
from canals.errors import PipelineMaxLoops
from sample_components import AddFixedValue, Threshold, MergeLoop

import logging

logging.basicConfig(level=logging.DEBUG)


def test_max_loops():
    pipe = Pipeline(max_loops_allowed=10)
    pipe.add_component("add", AddFixedValue())
    pipe.add_component("threshold", Threshold(threshold=100))
    pipe.add_component("merge", MergeLoop(expected_type=int, inputs=["value_1", "value_2"]))
    pipe.connect("threshold.below", "add.value")
    pipe.connect("add.result", "merge.value_1")
    pipe.connect("merge.value", "threshold.value")
    with pytest.raises(PipelineMaxLoops):
        pipe.run({"merge": {"value_2": 1}})
