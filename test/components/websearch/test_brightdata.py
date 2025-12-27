# SPDX-FileCopyrightText: 2025-present deepset GmbH
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from haystack import Document
from haystack.components.websearch.brightdata import BrightDataWebSearch, BrightDataWebSearchError
from haystack.utils import Secret

EXAMPLE_BRIGHT_DATA_RESPONSE = {
    "organic": [
        {"link": "https://example.com/page1", "title": "Example title 1", "description": "Snippet 1"},
        {"link": "https://example.org/page2", "title": "Example title 2", "description": "Snippet 2"},
    ]
}


@patch("haystack.components.websearch.brightdata.requests.post")
def test_brightdatawebsearch_basic(mock_post: Mock) -> None:
    mock_response = Mock()
    mock_response.json.return_value = EXAMPLE_BRIGHT_DATA_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    comp = BrightDataWebSearch(api_token=Secret.from_token("test-api-token"), zone="unblocker", top_k=1)
    result = comp.run(query="pizza")

    docs = result["documents"]
    links = result["links"]

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].meta["link"] == "https://example.com/page1"
    assert links == ["https://example.com/page1"]


@patch("haystack.components.websearch.brightdata.requests.post")
def test_brightdatawebsearch_allowed_domains_filter(mock_post: Mock) -> None:
    mock_response = Mock()
    mock_response.json.return_value = EXAMPLE_BRIGHT_DATA_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    comp = BrightDataWebSearch(
        api_token=Secret.from_token("test-api-token"), zone="unblocker", allowed_domains=["example.org"]
    )
    result = comp.run(query="test")

    assert all("example.org" in d.meta["link"] for d in result["documents"])
    assert all("example.org" in link for link in result["links"])


@patch("haystack.components.websearch.brightdata.requests.post")
def test_brightdatawebsearch_timeout(mock_post: Mock) -> None:
    from requests import Timeout as RequestsTimeout

    mock_post.side_effect = RequestsTimeout("timeout")
    comp = BrightDataWebSearch(api_token=Secret.from_token("test-api-token"), zone="unblocker")

    with pytest.raises(TimeoutError):
        comp.run(query="test")


@patch("haystack.components.websearch.brightdata.requests.post")
def test_brightdatawebsearch_request_exception(mock_post: Mock) -> None:
    from requests import RequestException

    mock_post.side_effect = RequestException("boom")
    comp = BrightDataWebSearch(api_token=Secret.from_token("test-api-token"), zone="unblocker")

    with pytest.raises(BrightDataWebSearchError):
        comp.run(query="test")


@patch("haystack.components.websearch.brightdata.requests.post")
def test_brightdatawebsearch_page_number_sets_start(mock_post: Mock) -> None:
    mock_response = Mock()
    mock_response.json.return_value = EXAMPLE_BRIGHT_DATA_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    comp = BrightDataWebSearch(api_token=Secret.from_token("test-api-token"), zone="unblocker")
    comp.run(query="pizza", page_number=2)

    called_url = mock_post.call_args.kwargs["json"]["url"]
    assert "start=10" in called_url


def test_brightdatawebsearch_requires_zone() -> None:
    with pytest.raises(ValueError):
        BrightDataWebSearch(api_token=Secret.from_token("test-api-token"), zone=None)  # type: ignore[arg-type]
