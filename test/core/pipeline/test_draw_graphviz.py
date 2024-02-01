# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import filecmp
import os

import pytest

from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.draw.draw import _draw
from haystack.testing.sample_components import Double

pygraphviz = pytest.importorskip("pygraphviz")


@pytest.mark.integration
def test_draw_pygraphviz(tmp_path, test_files):
    pipe = Pipeline()
    comp1 = Double()
    comp2 = Double()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    pipe.connect(comp1.outputs.value, comp2.inputs.value)

    _draw(pipe.graph, tmp_path / "test_pipe.jpg", engine="graphviz")
    assert os.path.exists(tmp_path / "test_pipe.jpg")
    assert filecmp.cmp(tmp_path / "test_pipe.jpg", test_files / "pipeline_draw" / "pygraphviz.jpg")
