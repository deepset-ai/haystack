import os
import pytest
from haystack.preview import Document
from haystack.preview.components.websearch.serper_dev import SerperDevWebSearch


@pytest.mark.skipif(
    not os.environ.get("SERPERDEV_API_KEY", None),
    reason="Export an env var called SERPERDEV_API_KEY containing the SerperDev API key to run this test.",
)
def test_web_search_top_k():
    ws = SerperDevWebSearch(api_key=os.environ.get("SERPERDEV_API_KEY", None), top_k=10)
    results = ws.run(query="Who is the boyfriend of Olivia Wilde?")["documents"]
    assert len(results) == 10
    assert all(isinstance(doc, Document) for doc in results)
