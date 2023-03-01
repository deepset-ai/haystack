import os

import pytest

from haystack.nodes.search_engine import WebSearch
from haystack.schema import Document


@pytest.mark.skipif(
    not os.environ.get("SERPERDEV_API_KEY", None),
    reason="Please export an env var called SERPERDEV_API_KEY containing the serper.dev API key to run this test.",
)
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
def test_web_search_with_site_keyword():
    ws = WebSearch(api_key=os.environ.get("SERPERDEV_API_KEY", None))
    result, _ = ws.run(query='site:lifewire.com OR site:nasa.gov "electric vehicles"')
    assert "documents" in result
    assert len(result["documents"]) > 0
    assert isinstance(result["documents"][0], Document)
    assert all(
        ["nasa" in doc.meta["link"] or "lifewire" in doc.meta["link"] for doc in result["documents"]]
    ), "Some documents are not from the specified sites lifewire.com or nasa.gov."
