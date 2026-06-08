# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import httpx
import pytest

from haystack.core.errors import PipelineDrawingError
from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.draw import _to_mermaid_image, _to_mermaid_text, _validate_image_response
from haystack.testing.sample_components import AddFixedValue, Double


@pytest.mark.skip(reason="Temporarily skipped due to mermaid.ink issues")
@pytest.mark.integration
def test_to_mermaid_image():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    image_data = _to_mermaid_image(pipe.graph)
    # We just verify we received some data as testing the actual image is not reliable
    assert image_data


@patch("haystack.core.pipeline.draw.httpx")
def test_to_mermaid_image_does_not_edit_graph(mock_httpx):
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue(add=3))
    pipe.add_component("comp2", Double())
    pipe.connect("comp1.result", "comp2.value")
    pipe.connect("comp2.value", "comp1.value")

    mock_httpx.get.return_value = MagicMock(
        status_code=200, content=b"\x89PNG\r\n\x1a\n", headers={"content-type": "image/png"}
    )
    expected_pipe = pipe.to_dict()
    _to_mermaid_image(pipe.graph)
    assert expected_pipe == pipe.to_dict()


@patch("haystack.core.pipeline.draw.httpx")
def test_to_mermaid_image_applies_timeout(mock_httpx):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    mock_httpx.get.return_value = MagicMock(
        status_code=200, content=b"\x89PNG\r\n\x1a\n", headers={"content-type": "image/png"}
    )
    _to_mermaid_image(pipe.graph, timeout=1)

    assert mock_httpx.get.call_args[1]["timeout"] == 1


def test_to_mermaid_image_failing_request(tmp_path):
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    with patch("haystack.core.pipeline.draw.httpx.get") as mock_get:

        def raise_for_status(self):
            raise httpx.HTTPError("error")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.content = '{"error": "too many requests"}'
        mock_response.raise_for_status = raise_for_status
        mock_get.return_value = mock_response

        with pytest.raises(PipelineDrawingError, match="There was an issue with https://mermaid.ink"):
            _to_mermaid_image(pipe.graph)


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

comp1["<b>comp1</b><br><small><i>AddFixedValue<br><br>Optional inputs:<ul style='text-align:left;'><li>add (int | None)</li></ul></i></small>"]:::component -- "result -> value<br><small><i>int</i></small>" --> comp2["<b>comp2</b><br><small><i>Double</i></small>"]:::component
comp2["<b>comp2</b><br><small><i>Double</i></small>"]:::component -- "value -> value<br><small><i>int</i></small>" --> comp1["<b>comp1</b><br><small><i>AddFixedValue<br><br>Optional inputs:<ul style='text-align:left;'><li>add (int | None)</li></ul></i></small>"]:::component

classDef component text-align:center;

"""  # noqa: E501
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


@pytest.mark.skip(reason="This is a nice to have, but frequently fails due to mermaid.ink issues")
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


@patch("haystack.core.pipeline.draw.httpx.get")
def test_to_mermaid_image_server_error(mock_get):
    # Test server failure
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    def raise_for_status(self):
        raise httpx.HTTPError("error")

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


@pytest.mark.parametrize(
    "params, content, content_type",
    [
        ({"format": "img", "type": "png"}, b"\x89PNG\r\n\x1a\n" + b"rest", "image/png"),
        ({"format": "img", "type": "jpeg"}, b"\xff\xd8\xff" + b"rest", "image/jpeg"),
        ({"format": "img", "type": "webp"}, b"RIFF\x00\x00\x00\x00WEBPrest", "image/webp"),
        ({"format": "svg"}, b'<?xml version="1.0"?><svg></svg>', "image/svg+xml"),
        ({"format": "svg"}, b"<svg xmlns='...'></svg>", "image/svg+xml"),
        ({"format": "pdf"}, b"%PDF-1.7\nrest", "application/pdf"),
    ],
)
def test_validate_image_response_accepts_expected_formats(params, content, content_type):
    resp = MagicMock(content=content, headers={"content-type": content_type})
    # Should not raise
    _validate_image_response(resp, params)


def test_validate_image_response_rejects_empty_body():
    resp = MagicMock(content=b"", headers={"content-type": "image/png"})
    with pytest.raises(PipelineDrawingError, match="empty response"):
        _validate_image_response(resp, {"format": "img", "type": "png"})


def test_validate_image_response_rejects_mismatched_body():
    # A server returning an HTML error page (or attacker-controlled payload) while the caller
    # expects a PNG must be rejected so it never gets written to disk.
    resp = MagicMock(content=b"<html><body>error</body></html>", headers={"content-type": "image/png"})
    with pytest.raises(PipelineDrawingError, match="does not look like a valid PNG image"):
        _validate_image_response(resp, {"format": "img", "type": "png"})


def test_validate_image_response_warns_on_spoofed_content_type_but_relies_on_body(caplog):
    # Content-Type is server-controlled, so a wrong header only warns as long as the body is valid.
    resp = MagicMock(content=b"\x89PNG\r\n\x1a\n" + b"rest", headers={"content-type": "text/html"})
    _validate_image_response(resp, {"format": "img", "type": "png"})
    assert "unexpected Content-Type" in caplog.text


@patch("haystack.core.pipeline.draw.httpx.get")
def test_to_mermaid_image_rejects_non_image_response(mock_get):
    # End-to-end: a 200 response with non-image content must not be returned for writing to disk.
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    mock_get.return_value = MagicMock(
        status_code=200, content=b"#!/bin/sh\nrm -rf /\n", headers={"content-type": "image/png"}
    )

    with pytest.raises(PipelineDrawingError, match="does not look like a valid PNG image"):
        _to_mermaid_image(pipe.graph)
