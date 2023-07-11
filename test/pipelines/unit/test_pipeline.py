# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List
import pytest

from canals.pipeline import Pipeline, PipelineMaxLoops
from test.sample_components import AddFixedValue, Threshold, MergeLoop

import logging

logging.basicConfig(level=logging.DEBUG)


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
