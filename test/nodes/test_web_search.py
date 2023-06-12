import os
from unittest.mock import MagicMock, patch

import pytest

from haystack.nodes.search_engine import WebSearch
from haystack.schema import Document

try:
    import googleapiclient

    googleapi_installed = True
except ImportError:
    googleapi_installed = False


@pytest.mark.skipif(
    not os.environ.get("SERPERDEV_API_KEY", None),
    reason="Please export an env var called SERPERDEV_API_KEY containing the serper.dev API key to run this test.",
)
@pytest.mark.integration
def test_web_search():
    ws = WebSearch(api_key=os.environ.get("SERPERDEV_API_KEY", None))
    result, _ = ws.run(query="Who is the boyfriend of Olivia Wilde?")
    assert "documents" in result
    assert len(result["documents"]) > 0
    assert isinstance(result["documents"][0], Document)


@pytest.mark.skipif(
    not os.environ.get("SERPERDEV_API_KEY", None),
    reason="Please export an env var called SERPERDEV_API_KEY containing the serper.dev API key to run this test.",
)
@pytest.mark.integration
def test_web_search_with_site_keyword():
    ws = WebSearch(api_key=os.environ.get("SERPERDEV_API_KEY", None))
    result, _ = ws.run(query='site:lifewire.com OR site:nasa.gov "electric vehicles"')
    assert "documents" in result
    assert len(result["documents"]) > 0
    assert isinstance(result["documents"][0], Document)
    assert all(
        ["nasa" in doc.meta["link"] or "lifewire" in doc.meta["link"] for doc in result["documents"]]
    ), "Some documents are not from the specified sites lifewire.com or nasa.gov."


@pytest.mark.unit
def test_web_search_with_google_api_provider():
    if not googleapi_installed:
        pytest.skip("google-api-python-client is not installed, skipping test.")

    GOOGLE_API_KEY = "dummy_api_key"
    SEARCH_ENGINE_ID = "dummy_search_engine_id"
    query = "The founder of Python"

    with patch("haystack.nodes.search_engine.WebSearch.run") as mock_run:
        mock_run.return_value = ([{"content": "Guido van Rossum"}], None)
        ws = WebSearch(
            api_key=GOOGLE_API_KEY,
            search_engine_provider="GoogleAPI",
            search_engine_kwargs={"engine_id": SEARCH_ENGINE_ID},
        )
        result, _ = ws.run(query=query)

        mock_run.assert_called_once_with(query=query)

        assert "guido" in result[0]["content"].lower()


@pytest.mark.unit
def test_web_search_with_google_api_client():
    if not googleapi_installed:
        pytest.skip("google-api-python-client is not installed, skipping test.")

    GOOGLE_API_KEY = "dummy_api_key"
    SEARCH_ENGINE_ID = "dummy_search_engine_id"
    query = "The founder of Python"

    with patch("googleapiclient.discovery.build") as mock_build:
        mock_service = MagicMock()
        mock_cse = MagicMock()
        mock_list = MagicMock()

        mock_build.return_value = mock_service
        mock_service.cse.return_value = mock_cse
        mock_cse.list.return_value = mock_list
        mock_list.execute.return_value = {
            "items": [
                {
                    "title": "Guido van Rossum",
                    "snippet": "The founder of Python programming language.",
                    "link": "https://example.com/guido",
                }
            ]
        }

        ws = WebSearch(
            api_key=GOOGLE_API_KEY,
            search_engine_provider="GoogleAPI",
            search_engine_kwargs={"engine_id": SEARCH_ENGINE_ID},
        )
        _, _ = ws.run(query=query)

        mock_build.assert_called_once_with("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        mock_service.cse.assert_called_once()
        mock_cse.list.assert_called_once_with(q=query, cx=SEARCH_ENGINE_ID, num=10)
        mock_list.execute.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [1, 3, 6])
def test_web_search_top_k(mock_web_search, top_k):
    ws = WebSearch(api_key="some_invalid_key")
    result, _ = ws.run(query="Who is the boyfriend of Olivia Wilde?", top_k=top_k)
    assert "documents" in result
    assert len(result["documents"]) == top_k
    assert all(isinstance(doc, Document) for doc in result["documents"])
