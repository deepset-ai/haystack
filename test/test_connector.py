import json
from pathlib import Path

import pytest
from haystack.connector import Crawler


def test_crawler_url_none_exception(tmp_path):
    tmp_dir = tmp_path / "crawled_files"
    with pytest.raises(ValueError):
        Crawler(tmp_dir).crawl()


def test_crawler_depth(tmp_path):
    tmp_dir = tmp_path / "crawled_files"
    _url = ["https://haystack.deepset.ai/docs/v0.9.0/get_startedmd"]
    crawler = Crawler(output_dir=tmp_dir)
    doc_path = crawler.crawl(urls=_url, crawler_depth=0)
    assert len(doc_path) == 1

    _urls = [
        "https://haystack.deepset.ai/docs/v0.9.0/get_startedmd",
        "https://haystack.deepset.ai/docs/v0.8.0/get_startedmd",
        "https://haystack.deepset.ai/docs/v0.7.0/get_startedmd",
    ]
    doc_path = crawler.crawl(urls=_urls, crawler_depth=0)
    assert len(doc_path) == 3

    doc_path = crawler.crawl(urls=_url, crawler_depth=1)
    assert len(doc_path) > 1

    for json_file in doc_path:
        assert isinstance(json_file, Path)
        with open(json_file.absolute(), "r") as read_file:
            data = json.load(read_file)
            assert 'text' in data
            assert 'meta' in data
            assert isinstance(data['text'], str)
            assert len(data['text'].split()) > 2


def test_crawler_filter_urls(tmp_path):
    tmp_dir = tmp_path / "crawled_files"
    _url = ["https://haystack.deepset.ai/docs/v0.8.0/"]

    crawler = Crawler(output_dir=tmp_dir)
    doc_path = crawler.crawl(urls=_url, filter_urls=["haystack\.deepset\.ai\/docs\/v0\.9\.0\/"])
    assert len(doc_path) == 0

    doc_path = crawler.crawl(urls=_url, filter_urls=["haystack\.deepset\.ai\/docs\/v0\.8\.0\/"])
    assert len(doc_path) > 0

    doc_path = crawler.crawl(urls=_url, filter_urls=["google\.com"])
    assert len(doc_path) == 0
