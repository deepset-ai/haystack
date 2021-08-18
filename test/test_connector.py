import json
from pathlib import Path
from re import search

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


def test_crawler_content(tmp_path):
    tmp_dir = tmp_path / "crawled_files"

    partial_content_match: list = [
        {"url": "https://haystack.deepset.ai/docs/v0.7.0/intromd",
         "partial_content": ["What is Haystack",
                             "Utilize all transformer based models",
                             "a Retriever-Reader pipeline in order",
                             "Passing on only a small candidate set",
                             "fast indexing and querying",
                             "Fine-tune models to your own domain",
                             "smoothly switch when new ones get published"]},
        {"url": "https://haystack.deepset.ai/docs/v0.6.0/use_casesmd",
         "partial_content": ["Semantic Search System",
                             "Store your documents in the database of ",
                             "results are chosen based on compatibility in",
                             "Apply a set of standard questions to each document",
                             "Return a NO_ANSWER if a given document",
                             "like what is the revenue forecast for 2020?",
                             "overview of academic papers and internal business"]}]

    crawler = Crawler(output_dir=tmp_dir)
    for _dict in partial_content_match:
        url: str = _dict['url']
        partial_content: list = _dict['partial_content']

        doc_path = crawler.crawl(urls=[url], crawler_depth=0)
        assert len(doc_path) == 1

        for json_file in doc_path:
            assert isinstance(json_file, Path)
            with open(json_file.absolute(), "r") as read_file:
                content = json.load(read_file)
                assert isinstance(content['text'], str)
                for partial_line in partial_content:
                    assert search(partial_line, content['text'])
                    assert partial_line in content['text']
