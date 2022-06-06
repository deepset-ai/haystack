import os
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from haystack.nodes.connector import Crawler
from haystack.schema import Document

from ..conftest import SAMPLES_PATH


@pytest.fixture(scope="session")
def test_url():
    return f"file://{SAMPLES_PATH.absolute()}/crawler"


def content_match(crawler: Crawler, url: str, crawled_page: Path):
    """
    :param crawler: the tested Crawler object
    :param base_url: the URL from test_url fixture
    :param page_name: the expected page
    :param crawled_page: the output of Crawler (one element of the paths list)
    """
    crawler.driver.get(url)
    body = crawler.driver.find_element_by_tag_name("body")
    expected_crawled_content = body.text

    with open(crawled_page, "r") as crawled_file:
        page_data = json.load(crawled_file)
        return page_data["content"] == expected_crawled_content


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


def test_crawler_depth_0_single_url(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    paths = crawler.crawl(urls=[test_url + "/index.html"], crawler_depth=0)
    assert len(paths) == 1
    assert content_match(crawler, test_url + "/index.html", paths[0])


def test_crawler_depth_0_many_urls(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    _urls = [test_url + "/index.html", test_url + "/page1.html"]
    paths = crawler.crawl(urls=_urls, crawler_depth=0)
    assert len(paths) == 2
    assert content_match(crawler, test_url + "/index.html", paths[0])
    assert content_match(crawler, test_url + "/page1.html", paths[1])


def test_crawler_depth_1_single_url(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    paths = crawler.crawl(urls=[test_url + "/index.html"], crawler_depth=1)
    assert len(paths) == 3
    assert content_match(crawler, test_url + "/index.html", paths[0])
    assert content_match(crawler, test_url + "/page1.html", paths[1])
    assert content_match(crawler, test_url + "/page2.html", paths[2])


def test_crawler_output_file_structure(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    paths = crawler.crawl(urls=[test_url + "/index.html"], crawler_depth=0)
    assert content_match(crawler, test_url + "/index.html", paths[0])

    with open(paths[0].absolute(), "r") as doc_file:
        data = json.load(doc_file)
        assert "content" in data
        assert "meta" in data
        assert isinstance(data["content"], str)
        assert len(data["content"].split()) > 2


def test_crawler_filter_urls(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)

    paths = crawler.crawl(urls=[test_url + "/index.html"], filter_urls=["index"], crawler_depth=1)
    assert len(paths) == 1
    assert content_match(crawler, test_url + "/index.html", paths[0])

    # Note: filter_urls can exclude pages listed in `urls` as well
    paths = crawler.crawl(urls=[test_url + "/index.html"], filter_urls=["page1"], crawler_depth=1)
    assert len(paths) == 1
    assert content_match(crawler, test_url + "/page1.html", paths[0])
    assert not crawler.crawl(urls=[test_url + "/index.html"], filter_urls=["google\.com"], crawler_depth=1)


def test_crawler_return_document(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    documents, _ = crawler.run(urls=[test_url + "/index.html"], crawler_depth=0, return_documents=True)
    paths, _ = crawler.run(urls=[test_url + "/index.html"], crawler_depth=0, return_documents=False)

    for path, document in zip(paths["paths"], documents["documents"]):
        with open(path.absolute(), "r") as doc_file:
            file_content = json.load(doc_file)
            assert file_content["meta"] == document.meta
            assert file_content["content"] == document.content
