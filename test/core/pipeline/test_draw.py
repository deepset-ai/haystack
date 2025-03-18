# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest
import requests
import time

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


@patch("haystack.core.pipeline.draw.requests")
def test_to_mermaid_image_applies_timeout(mock_requests):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    mock_requests.get.return_value = MagicMock(status_code=200)
    _to_mermaid_image(pipe.graph, timeout=1)

    assert mock_requests.get.call_args[1]["timeout"] == 1


@patch("haystack.core.pipeline.draw.requests")
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

        with pytest.raises(PipelineDrawingError, match="There was an issue with https://mermaid.ink"):
            _to_mermaid_image(pipe.graph)


@patch("haystack.core.pipeline.draw.requests")
def test_to_mermaid_image_retries_on_failure(mock_requests):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.HTTPError()

    mock_requests.get.return_value = mock_response

    with pytest.raises(PipelineDrawingError, match="There was an issue with https://mermaid.ink"):
        _to_mermaid_image(pipe.graph, max_retries=2, initial_delay=0.1)

    assert mock_requests.get.call_count == 2


@patch("haystack.core.pipeline.draw.requests")
@patch("time.sleep")
def test_to_mermaid_image_exponential_backoff(mock_sleep, mock_requests):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.HTTPError()
    mock_requests.get.return_value = mock_response

    with pytest.raises(PipelineDrawingError):
        _to_mermaid_image(pipe.graph, max_retries=3, initial_delay=0.1)

    assert mock_requests.get.call_count == 3
    assert mock_sleep.call_count == 2  # 2 sleeps, since there are 3 retries.

    expected_delays = [0.1, 0.2]
    actual_delays = [call_args[0][0] for call_args in mock_sleep.call_args_list]
    assert actual_delays == pytest.approx(expected_delays)


def test_to_mermaid_text():
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue(add=3))
    pipe.add_component("comp2", Double())
    pipe.connect("comp1.result", "comp2.value")
    pipe.connect("comp2.value", "comp1.value")

    init_params = {"theme": "neutral"}
    text = _to_mermaid_text(pipe.graph, init_params)
    assert (
        text
        == """
%%{ init: {'theme': 'neutral'} }%%

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
    init_params = {"theme": "neutral"}
    _to_mermaid_text(pipe.graph, init_params)
    assert expected_pipe == pipe.to_dict()


@pytest.mark.integration
@pytest.mark.parametrize(
    "params",
    [
        {"format": "img", "type": "png", "theme": "dark"},
        {"format": "svg", "theme": "forest"},
        {"format": "pdf", "fit": True, "theme": "neutral"},
    ],
)
def test_to_mermaid_image_valid_formats(params):
    # Test valid formats
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    image_data = _to_mermaid_image(pipe.graph, params=params)
    assert image_data  # Ensure some data is returned


def test_to_mermaid_image_invalid_format():
    # Test invalid format
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    with pytest.raises(ValueError, match="Invalid image format:"):
        _to_mermaid_image(pipe.graph, params={"format": "invalid_format"})


@pytest.mark.integration
def test_to_mermaid_image_missing_theme():
    # Test default theme (neutral)
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    params = {"format": "img"}
    image_data = _to_mermaid_image(pipe.graph, params=params)

    assert image_data  # Ensure some data is returned


def test_to_mermaid_image_invalid_scale():
    # Test invalid scale
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    with pytest.raises(ValueError, match="Scale must be a number between 1 and 3."):
        _to_mermaid_image(pipe.graph, params={"format": "img", "scale": 5})


def test_to_mermaid_image_scale_without_dimensions():
    # Test scale without width/height
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    with pytest.raises(ValueError, match="Scale is only allowed when width or height is set."):
        _to_mermaid_image(pipe.graph, params={"format": "img", "scale": 2})


@patch("haystack.core.pipeline.draw.requests.get")
def test_to_mermaid_image_server_error(mock_get):
    # Test server failure
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    def raise_for_status(self):
        raise requests.HTTPError()

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.content = '{"error": "server error"}'
    mock_response.raise_for_status = raise_for_status
    mock_get.return_value = mock_response

    with pytest.raises(PipelineDrawingError, match="There was an issue with https://mermaid.ink"):
        _to_mermaid_image(pipe.graph)


def test_to_mermaid_image_invalid_server_url():
    # Test invalid server URL
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue(add=3))
    pipe.add_component("comp2", Double())
    pipe.connect("comp1.result", "comp2.value")
    pipe.connect("comp2.value", "comp1.value")

    server_url = "https://invalid.server"

    with pytest.raises(PipelineDrawingError, match=f"There was an issue with {server_url}"):
        _to_mermaid_image(pipe.graph, server_url=server_url)
