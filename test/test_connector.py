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
    _url = ["https://haystack.deepset.ai/overview/get-started"]
    crawler = Crawler(output_dir=tmp_dir)
    doc_path = crawler.crawl(urls=_url, crawler_depth=0)
    assert len(doc_path) == 1

    _urls = [
        "https://haystack.deepset.ai/overview/v0.8.0/get-started",
        "https://haystack.deepset.ai/overview/v0.7.0/get-started",
        "https://haystack.deepset.ai/overview/v0.6.0/get-started",
    ]
    doc_path = crawler.crawl(urls=_urls, crawler_depth=0)
    assert len(doc_path) == 3

    doc_path = crawler.crawl(urls=_url, crawler_depth=1)
    assert len(doc_path) > 1

    for json_file in doc_path:
        assert isinstance(json_file, Path)
        with open(json_file.absolute(), "r") as read_file:
            data = json.load(read_file)
            assert 'content' in data
            assert 'meta' in data
            assert isinstance(data['content'], str)
            assert len(data['content'].split()) > 2


def test_crawler_filter_urls(tmp_path):
    tmp_dir = tmp_path / "crawled_files"
    _url = ["https://haystack.deepset.ai/overview/v0.8.0/"]

    crawler = Crawler(output_dir=tmp_dir)
    doc_path = crawler.crawl(urls=_url, filter_urls=["haystack\.deepset\.ai\/overview\/v0\.9\.0\/"])
    assert len(doc_path) == 0

    doc_path = crawler.crawl(urls=_url, filter_urls=["haystack\.deepset\.ai\/overview\/v0\.8\.0\/"])
    assert len(doc_path) > 0

    doc_path = crawler.crawl(urls=_url, filter_urls=["google\.com"])
    assert len(doc_path) == 0


def test_crawler_content(tmp_path):
    tmp_dir = tmp_path / "crawled_files"

    partial_content_match: list = [
        {"url": "https://haystack.deepset.ai/overview/v0.7.0/intro",
         "partial_content": ["What is Haystack",
                             "Utilize all transformer based models",
                             "a Retriever-Reader pipeline in order",
                             "Passing on only a small candidate set",
                             "fast indexing and querying",
                             "Fine-tune models to your own domain",
                             "smoothly switch when new ones get published"]},
        {"url": "https://haystack.deepset.ai/overview/v0.7.0/use-cases",
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
                assert isinstance(content['content'], str)
                for partial_line in partial_content:
                    assert search(partial_line, content['content'])
                    assert partial_line in content['content']


def test_crawler_return_document(tmp_path):
    tmp_dir = tmp_path / "crawled_files"
    _url = ["https://haystack.deepset.ai/docs/v0.5.0/intromd"]

    crawler = Crawler(output_dir=tmp_dir)
    docs_path = crawler.crawl(urls=_url, crawler_depth=1)
    results, _ = crawler.run(urls=_url, crawler_depth=1, return_documents=True)
    documents = results['documents']

    for json_file, document in zip(docs_path, documents):
        assert isinstance(json_file, Path)
        assert isinstance(document, dict)

        with open(json_file.absolute(), "r") as read_file:
            file_content = json.load(read_file)
            assert file_content['meta'] == document['meta']
            assert file_content['content'] == document['content']
