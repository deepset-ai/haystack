# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from haystack.core.pipeline import Pipeline
from haystack.core.super_component import SuperComponent
from haystack.testing.sample_components import AddFixedValue, Double


@pytest.fixture
def sample_super_component():
    """Creates a sample SuperComponent for testing visualization methods"""
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue(add=3))
    pipe.add_component("comp2", Double())
    pipe.connect("comp1.result", "comp2.value")

    return SuperComponent(pipeline=pipe)


@patch("haystack.core.pipeline.Pipeline.show")
def test_show_delegates_to_pipeline(mock_show, sample_super_component):
    """Test that SuperComponent.show() correctly delegates to Pipeline.show() with all parameters"""

    server_url = "https://custom.mermaid.server"
    params = {"theme": "dark", "format": "svg"}
    timeout = 60

    sample_super_component.show(server_url=server_url, params=params, timeout=timeout)
    mock_show.assert_called_once_with(server_url=server_url, params=params, timeout=timeout)


@patch("haystack.core.pipeline.Pipeline.draw")
def test_draw_delegates_to_pipeline(mock_draw, sample_super_component, tmp_path):
    """Test that SuperComponent.draw() correctly delegates to Pipeline.draw() with all parameters"""

    path = tmp_path / "test_pipeline.png"
    server_url = "https://custom.mermaid.server"
    params = {"theme": "dark", "format": "png"}
    timeout = 60

    sample_super_component.draw(path=path, server_url=server_url, params=params, timeout=timeout)
    mock_draw.assert_called_once_with(path=path, server_url=server_url, params=params, timeout=timeout)


@patch("haystack.core.pipeline.Pipeline.show")
def test_show_with_default_parameters(mock_show, sample_super_component):
    """Test that SuperComponent.show() works with default parameters"""

    sample_super_component.show()
    mock_show.assert_called_once_with(server_url="https://mermaid.ink", params=None, timeout=30)


@patch("haystack.core.pipeline.Pipeline.draw")
def test_draw_with_default_parameters(mock_draw, sample_super_component, tmp_path):
    """Test that SuperComponent.draw() works with default parameters except path"""

    path = tmp_path / "test_pipeline.png"

    sample_super_component.draw(path=path)
    mock_draw.assert_called_once_with(path=path, server_url="https://mermaid.ink", params=None, timeout=30)
