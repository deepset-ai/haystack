import json
import pytest
from pathlib import Path

import pytest
from selenium import webdriver

from haystack.nodes.connector import Crawler
from haystack.schema import Document


TEST_BASE_URL = "http://test.test.test.test.test"  # Non existing but valid URL

TEST_HOME_PAGE = """
<html>
<head>
    <title>Test Home Page for Crawler</title>
</head>
<body>
    <p>test page content</p>
    <a href="/page1">page 1</a>
    <a href="/page2">page 2</a>
</body>
</html>
"""

TEST_PAGE1 = """
<html>
<head>
    <title>Test Page 1 for Crawler</title>
</head>
<body>
    <p>test page 1 content</p>
    <a href="/">home</a>
    <a href="/page2">page 2</a>
</body>
</html>
"""
TEST_PAGE2 = """
<html>
<head>
    <title>Test Page 2 for Crawler</title>
</head>
<body>
    <p>test page 2 content</p>
    <a href="/">home</a>
    <a href="/page1">page 1</a>
</body>
</html>
"""
TEST_UNWANTED = """
<html>
<head>
    <title>Should never see this page</title>
</head>
<body>
    <p>Should never see this page!</p>
</body>
</html>
"""


@pytest.fixture(autouse=True)
def mock_webdriver(request, monkeypatch):
    # Do not patch integration tests
    if "integration" in request.keywords:
        return
    
    def mock_get(self, url: str) -> None:
        if url == TEST_BASE_URL:
            return TEST_HOME_PAGE
        if "page1" in url:
            return TEST_PAGE1
        if "page2" in url:
            return TEST_PAGE2
        else:
            return TEST_UNWANTED

    monkeypatch.setattr(webdriver.Chrome, "get", mock_get)


#
# Integration
#

@pytest.mark.integration
def test_crawler(tmp_path):
    tmp_dir = tmp_path
    url = ["https://haystack.deepset.ai/"]

    crawler = Crawler(output_dir=tmp_dir)
    docs_path = crawler.crawl(urls=url, crawler_depth=0)
    results, _ = crawler.run(urls=url, crawler_depth=0, return_documents=True)
    documents = results["documents"]

    for json_file, document in zip(docs_path, documents):
        assert isinstance(json_file, Path)
        assert isinstance(document, Document)

        with open(json_file.absolute(), "r") as read_file:
            file_content = json.load(read_file)
            assert file_content["meta"] == document.meta
            assert file_content["content"] == document.content


#
# Unit tests
#

def test_crawler_url_none_exception(tmp_path):
    crawler = Crawler(tmp_path)
    with pytest.raises(ValueError):
        crawler.crawl()


def test_crawler_depth_0_single_url(tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    paths = crawler.crawl(urls=[TEST_BASE_URL], crawler_depth=0)
    assert len(paths) == 1


def test_crawler_depth_0_many_urls(tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    _urls = [
        TEST_BASE_URL,
        TEST_BASE_URL+"/page1",
        TEST_BASE_URL+"/page2",
    ]
    paths = crawler.crawl(urls=_urls, crawler_depth=0)
    assert len(paths) == 3


def test_crawler_depth_1_single_url(tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    paths = crawler.crawl(urls=[TEST_BASE_URL], crawler_depth=1)
    assert len(paths) == 3


def test_crawler_output_file_structure(tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    paths = crawler.crawl(urls=[TEST_BASE_URL], crawler_depth=0)
    with open(paths[0].absolute(), "r") as doc_file:
        data = json.load(doc_file)
        assert "content" in data
        assert "meta" in data
        assert isinstance(data["content"], str)
        assert len(data["content"].split()) > 2


def test_crawler_filter_urls(tmp_path):
    crawler = Crawler(output_dir=tmp_path)

    assert len(crawler.crawl(urls=[TEST_BASE_URL], filter_urls=["page1"], crawler_depth=1)) == 1
    assert not crawler.crawl(urls=[TEST_BASE_URL], filter_urls=["page3"], crawler_depth=1)
    assert not crawler.crawl(urls=[TEST_BASE_URL], filter_urls=["google\.com"], crawler_depth=1)


def test_crawler_content(tmp_path):
    expected_results = [
        {"url": TEST_BASE_URL, "partial_content": ["test page content"]},
        {"url": TEST_BASE_URL + "/page1", "partial_content": ["test page 1 content"]},
        {"url": TEST_BASE_URL + "/page2", "partial_content": ["test page 2 content"]},
    ]

    crawler = Crawler(output_dir=tmp_path)
    for result in expected_results:
        paths = crawler.crawl(urls=[result["url"]], crawler_depth=0)
        with open(paths[0].absolute(), "r") as read_file:
            content = json.load(read_file)
            assert result["partial_content"] in content["content"]


def test_crawler_return_document(tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    documents, _ = crawler.run(urls=[TEST_BASE_URL], crawler_depth=0, return_documents=True)
    paths, _ = crawler.run(urls=[TEST_BASE_URL], crawler_depth=0, return_documents=False)

    for path, document in zip(paths["paths"], documents["documents"]):
        with open(path.absolute(), "r") as doc_file:
            file_content = json.load(doc_file)
            assert file_content["meta"] == document.meta
            assert file_content["content"] == document.content
