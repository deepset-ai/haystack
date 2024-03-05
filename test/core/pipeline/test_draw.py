# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest
import requests

from haystack.core.errors import PipelineDrawingError
from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.draw import _to_mermaid_image, _to_mermaid_text
from haystack.testing.sample_components import AddFixedValue, Double


@pytest.mark.flaky(reruns=5, reruns_delay=5)
@pytest.mark.integration
def test_to_mermaid_image():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    image_data = _to_mermaid_image(pipe.graph)
    # We just verify we received some data as testing the actual image is not reliable
    assert image_data


@patch("haystack.core.pipeline.draw.requests")
def test_to_mermaid_image_does_not_edit_graph(mock_requests):
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue(add=3))
    pipe.add_component("comp2", Double())
    pipe.connect("comp1.result", "comp2.value")
    pipe.connect("comp2.value", "comp1.value")

    mock_requests.get.return_value = MagicMock(status_code=200)
    expected_pipe = pipe.to_dict()
    _to_mermaid_image(pipe.graph)
    assert expected_pipe == pipe.to_dict()


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
            _to_mermaid_image(pipe.graph)


def test_to_mermaid_text():
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue(add=3))
    pipe.add_component("comp2", Double())
    pipe.connect("comp1.result", "comp2.value")
    pipe.connect("comp2.value", "comp1.value")

    text = _to_mermaid_text(pipe.graph)
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


def test_to_mermaid_text_does_not_edit_graph():
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue(add=3))
    pipe.add_component("comp2", Double())
    pipe.connect("comp1.result", "comp2.value")
    pipe.connect("comp2.value", "comp1.value")

    expected_pipe = pipe.to_dict()
    _to_mermaid_text(pipe.graph)
    assert expected_pipe == pipe.to_dict()
