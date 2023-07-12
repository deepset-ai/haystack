# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import filecmp

from unittest.mock import patch, MagicMock
import pytest
import requests

from canals.pipeline import Pipeline
from canals.errors import PipelineDrawingError
from canals.draw import draw, convert

from test.sample_components import Double


@pytest.mark.skipif(sys.platform.lower().startswith("darwin"), reason="the available graphviz version is too recent")
@pytest.mark.skipif(sys.platform.lower().startswith("win"), reason="pygraphviz is not really available in Windows")
def test_draw_pygraphviz(tmp_path, test_files):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    draw(pipe.graph, tmp_path / "test_pipe.jpg", engine="graphviz")
    assert os.path.exists(tmp_path / "test_pipe.jpg")
    assert filecmp.cmp(tmp_path / "test_pipe.jpg", test_files / "pipeline_draw" / "pygraphviz.jpg")


def test_draw_mermaid_img(tmp_path, test_files):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    draw(pipe.graph, tmp_path / "test_pipe.jpg", engine="mermaid-img")
    assert os.path.exists(tmp_path / "test_pipe.jpg")
    assert filecmp.cmp(tmp_path / "test_pipe.jpg", test_files / "mermaid_mock" / "test_response.png")


def test_draw_mermaid_img_failing_request(tmp_path):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    with patch("canals.draw.mermaid.requests.get") as mock_get:

        def raise_for_status(self):
            raise requests.HTTPError()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.content = '{"error": "too many requests"}'
        mock_response.raise_for_status = raise_for_status
        mock_get.return_value = mock_response

        with pytest.raises(PipelineDrawingError, match="There was an issue with https://mermaid.ink/"):
            draw(pipe.graph, tmp_path / "test_pipe.jpg", engine="mermaid-img")


def test_draw_mermaid_txt(tmp_path):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    draw(pipe.graph, tmp_path / "test_pipe.md", engine="mermaid-text")
    assert os.path.exists(tmp_path / "test_pipe.md")
    assert (
        open(tmp_path / "test_pipe.md", "r").read()
        == """graph TD;
comp1 -- value -> value --> comp2
comp2 -- value -> value --> comp1"""
    )


def test_draw_unknown_engine(tmp_path):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    with pytest.raises(ValueError, match="Unknown rendering engine 'unknown'"):
        draw(pipe.graph, tmp_path / "test_pipe.jpg", engine="unknown")


def test_convert_unknown_engine(tmp_path):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    with pytest.raises(ValueError, match="Unknown rendering engine 'unknown'"):
        convert(pipe.graph, engine="unknown")
