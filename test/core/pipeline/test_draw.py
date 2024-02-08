# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest
import requests

from haystack.core.errors import PipelineDrawingError
from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.draw import _draw, _prepare_for_drawing, _to_mermaid_image, _to_mermaid_text
from haystack.testing.sample_components import AddFixedValue, Double


@patch("haystack.core.pipeline.draw._to_mermaid_image")
def test_draw_does_not_edit_graph(mock_to_mermaid_image, tmp_path):
    mock_to_mermaid_image.return_value = b"some_image_data"

    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    before_draw = pipe.to_dict()
    image_path = tmp_path / "test.png"
    pipe.draw(path=image_path)

    assert before_draw == pipe.to_dict()

    assert image_path.read_bytes() == mock_to_mermaid_image.return_value


@patch("haystack.core.pipeline.draw._to_mermaid_image")
@patch("IPython.core.getipython.get_ipython")
@patch("IPython.display.Image")
@patch("IPython.display.display")
def test_draw_display_in_notebook(mock_ipython_display, mock_ipython_image, mock_get_ipython, mock_to_mermaid_image):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    mock_to_mermaid_image.return_value = b"some_image_data"
    mock_get_ipython.return_value = MagicMock(config={"IPKernelApp": True})

    _draw(pipe.graph)
    mock_ipython_image.assert_called_once_with(b"some_image_data")
    mock_ipython_display.assert_called_once()


@patch("haystack.core.pipeline.draw._to_mermaid_image")
@patch("IPython.core.getipython.get_ipython")
@patch("IPython.display.Image")
@patch("IPython.display.display")
def test_draw_display_in_notebook_saves_image(
    mock_ipython_display, mock_ipython_image, mock_get_ipython, mock_to_mermaid_image, tmp_path
):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    mock_to_mermaid_image.return_value = b"some_image_data"
    mock_get_ipython.return_value = MagicMock(config={"IPKernelApp": True})

    image_path = tmp_path / "test.png"
    _draw(pipe.graph, path=image_path)

    assert image_path.read_bytes() == mock_to_mermaid_image.return_value


@patch("haystack.core.pipeline.draw._to_mermaid_image")
def test_draw_raises_if_no_path_not_in_notebook(mock_to_mermaid_image, tmp_path, monkeypatch):
    # Simulate not being in a notebook
    monkeypatch.delattr("IPython.core.getipython")

    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    with pytest.raises(ValueError):
        _draw(pipe.graph)


@pytest.mark.integration
def test_to_mermaid_image(test_files):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    image_data = _to_mermaid_image(_prepare_for_drawing(pipe.graph))
    test_image = test_files / "mermaid_mock" / "test_response.png"
    assert test_image.read_bytes() == image_data


@pytest.mark.integration
def test_to_mermaid_image_failing_request(tmp_path):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    with patch("haystack.core.pipeline.draw.requests.get") as mock_get:

        def raise_for_status(self):
            raise requests.HTTPError()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.content = '{"error": "too many requests"}'
        mock_response.raise_for_status = raise_for_status
        mock_get.return_value = mock_response

        with pytest.raises(PipelineDrawingError, match="There was an issue with https://mermaid.ink/"):
            _to_mermaid_image(_prepare_for_drawing(pipe.graph))


@pytest.mark.integration
def test_to_mermaid_text(tmp_path):
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue(add=3))
    pipe.add_component("comp2", Double())
    pipe.connect("comp1.result", "comp2.value")
    pipe.connect("comp2.value", "comp1.value")

    text = _to_mermaid_text(_prepare_for_drawing(pipe.graph))
    assert (
        text
        == """
%%{ init: {'theme': 'neutral' } }%%

graph TD;

comp1["<b>comp1</b><br><small><i>AddFixedValue<br><br>Optional inputs:<ul style='text-align:left;'><li>add (Optional[int])</li></ul></i></small>"]:::component -- "result -> value<br><small><i>int</i></small>" --> comp2["<b>comp2</b><br><small><i>Double</i></small>"]:::component
comp2["<b>comp2</b><br><small><i>Double</i></small>"]:::component -- "value -> value<br><small><i>int</i></small>" --> comp1["<b>comp1</b><br><small><i>AddFixedValue<br><br>Optional inputs:<ul style='text-align:left;'><li>add (Optional[int])</li></ul></i></small>"]:::component

classDef component text-align:center;
"""
    )
