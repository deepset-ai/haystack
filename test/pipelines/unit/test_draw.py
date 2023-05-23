import os
import sys
import filecmp
from hashlib import md5

import pytest

from canals.pipeline import Pipeline

from test.sample_components import Double


@pytest.mark.skipif(sys.platform.startswith("win"), reason="pygraphviz is not really available in Windows")
def test_draw_pygraphviz(tmp_path, test_files):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    pipe.draw(tmp_path / "test_pipe.jpg", engine="graphviz")
    assert os.path.exists(tmp_path / "test_pipe.jpg")
    assert filecmp.cmp(tmp_path / "test_pipe.jpg", test_files / "pipeline_draw" / "pygraphviz.jpg")


def test_draw_mermaid_img(tmp_path, test_files):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    pipe.draw(tmp_path / "test_pipe.jpg", engine="mermaid-img")
    assert os.path.exists(tmp_path / "test_pipe.jpg")
    assert filecmp.cmp(tmp_path / "test_pipe.jpg", test_files / "mermaid_mock" / "test_response.png")


def test_draw_mermaid_txt(tmp_path):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    pipe.draw(tmp_path / "test_pipe.md", engine="mermaid-text")
    assert os.path.exists(tmp_path / "test_pipe.md")
    assert (
        open(tmp_path / "test_pipe.md", "r").read()
        == """graph TD;
comp1 -- value -> value --> comp2
comp2 -- value -> value --> comp1"""
    )
