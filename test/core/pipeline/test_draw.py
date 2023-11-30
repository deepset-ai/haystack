# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
import filecmp

from unittest.mock import patch, MagicMock
import pytest
import requests

from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.draw.draw import _draw, _convert
from haystack.core.errors import PipelineDrawingError
from haystack.testing.sample_components import Double, AddFixedValue


@pytest.mark.integration
def test_draw_mermaid_image(tmp_path, test_files):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    _draw(pipe.graph, tmp_path / "test_pipe.jpg", engine="mermaid-image")
    assert os.path.exists(tmp_path / "test_pipe.jpg")
    assert filecmp.cmp(tmp_path / "test_pipe.jpg", test_files / "mermaid_mock" / "test_response.png")


@pytest.mark.integration
def test_draw_mermaid_img_failing_request(tmp_path):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    with patch("haystack.core.pipeline.draw.mermaid.requests.get") as mock_get:

        def raise_for_status(self):
            raise requests.HTTPError()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.content = '{"error": "too many requests"}'
        mock_response.raise_for_status = raise_for_status
        mock_get.return_value = mock_response

        with pytest.raises(PipelineDrawingError, match="There was an issue with https://mermaid.ink/"):
            _draw(pipe.graph, tmp_path / "test_pipe.jpg", engine="mermaid-image")


@pytest.mark.integration
def test_draw_mermaid_text(tmp_path):
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue(add=3))
    pipe.add_component("comp2", Double())
    pipe.connect("comp1.result", "comp2.value")
    pipe.connect("comp2.value", "comp1.value")

    _draw(pipe.graph, tmp_path / "test_pipe.md", engine="mermaid-text")
    assert os.path.exists(tmp_path / "test_pipe.md")
    assert (
        open(tmp_path / "test_pipe.md", "r").read()
        == """
%%{ init: {'theme': 'neutral' } }%%

graph TD;

comp1["<b>comp1</b><br><small><i>AddFixedValue<br><br>Optional inputs:<ul style='text-align:left;'><li>add (Optional[int])</li></ul></i></small>"]:::component -- "result -> value<br><small><i>int</i></small>" --> comp2["<b>comp2</b><br><small><i>Double</i></small>"]:::component
comp2["<b>comp2</b><br><small><i>Double</i></small>"]:::component -- "value -> value<br><small><i>int</i></small>" --> comp1["<b>comp1</b><br><small><i>AddFixedValue<br><br>Optional inputs:<ul style='text-align:left;'><li>add (Optional[int])</li></ul></i></small>"]:::component

classDef component text-align:center;
"""
    )


def test_draw_unknown_engine(tmp_path):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    with pytest.raises(ValueError, match="Unknown rendering engine 'unknown'"):
        _draw(pipe.graph, tmp_path / "test_pipe.jpg", engine="unknown")


def test_convert_unknown_engine(tmp_path):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    with pytest.raises(ValueError, match="Unknown rendering engine 'unknown'"):
        _convert(pipe.graph, engine="unknown")
