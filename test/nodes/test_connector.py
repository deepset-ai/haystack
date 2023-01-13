from typing import List

import json
from pathlib import Path
import re
import hashlib
import os

import pytest
from selenium.webdriver.common.by import By


from haystack.nodes.connector import Crawler
from haystack.schema import Document

from ..conftest import SAMPLES_PATH


@pytest.fixture(scope="session")
def test_url():
    return (SAMPLES_PATH / "crawler").absolute().as_uri()


def content_match(crawler: Crawler, url: str, crawled_page: Path):
    """
    :param crawler: the tested Crawler object
    :param url: the URL of the expected page
    :param crawled_page: the output of Crawler (one element of the paths list)
    """
    crawler.driver.get(url)

    body = crawler.driver.find_element(by=By.TAG_NAME, value="body")

    if crawler.extract_hidden_text:
        expected_crawled_content = body.get_attribute("textContent")
    else:
        expected_crawled_content = body.text

    with open(crawled_page, "r") as crawled_file:
        page_data = json.load(crawled_file)
        return page_data["content"] == expected_crawled_content


def content_in_results(crawler: Crawler, url: str, results: List[Path], expected_matches_count=1):
    """
    Makes sure there is exactly one matching page in the list of pages returned
    by the crawler.

    :param crawler: the tested Crawler object
    :param url: the URL of the page to find in the results
    :param results: the crawler's output (list of paths)
    :param expected_matches_count: how many copies of this page should be present in the results (default 1)
    """
    return sum(content_match(crawler, url, path) for path in results) == expected_matches_count


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
    assert content_in_results(crawler, test_url + "/index.html", paths)
    assert content_in_results(crawler, test_url + "/page1.html", paths)


def test_crawler_depth_1_single_url(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    paths = crawler.crawl(urls=[test_url + "/index.html"], crawler_depth=1)
    assert len(paths) == 3
    assert content_in_results(crawler, test_url + "/index.html", paths)
    assert content_in_results(crawler, test_url + "/page1.html", paths)
    assert content_in_results(crawler, test_url + "/page2.html", paths)


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
    assert not crawler.crawl(urls=[test_url + "/index.html"], filter_urls=["google.com"], crawler_depth=1)


def test_crawler_return_document(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    documents, _ = crawler.run(urls=[test_url + "/index.html"], crawler_depth=0, return_documents=True)
    paths, _ = crawler.run(urls=[test_url + "/index.html"], crawler_depth=0, return_documents=False)

    for path, document in zip(paths["paths"], documents["documents"]):
        with open(path.absolute(), "r") as doc_file:
            file_content = json.load(doc_file)
            assert file_content["meta"] == document.meta
            assert file_content["content"] == document.content


def test_crawler_extract_hidden_text(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    documents, _ = crawler.run(
        urls=[test_url + "/page_w_hidden_text.html"], extract_hidden_text=True, crawler_depth=0, return_documents=True
    )
    crawled_content = documents["documents"][0].content
    assert "hidden text" in crawled_content

    documents, _ = crawler.run(
        urls=[test_url + "/page_w_hidden_text.html"], extract_hidden_text=False, crawler_depth=0, return_documents=True
    )
    crawled_content = documents["documents"][0].content
    assert "hidden text" not in crawled_content


def test_crawler_loading_wait_time(test_url, tmp_path):
    loading_wait_time = 3
    crawler = Crawler(output_dir=tmp_path)
    paths = crawler.crawl(urls=[test_url + "/page_dynamic.html"], crawler_depth=1, loading_wait_time=loading_wait_time)

    assert len(paths) == 4

    with open(f"{SAMPLES_PATH.absolute()}/crawler/page_dynamic_result.txt", "r") as dynamic_result:
        dynamic_result_text = dynamic_result.readlines()
        for path in paths:
            with open(path, "r") as crawled_file:
                page_data = json.load(crawled_file)
                if page_data["meta"]["url"] == test_url + "/page_dynamic.html":
                    content = page_data["content"].split("\n")
                    for line in dynamic_result_text:
                        assert dynamic_result_text[line].stip() == content[line].stip()

    assert content_in_results(crawler, test_url + "/index.html", paths)
    assert content_in_results(crawler, test_url + "/page1.html", paths)
    assert content_in_results(crawler, test_url + "/page2.html", paths)


def test_crawler_default_naming_function(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)

    link = f"{test_url}/page_with_a_very_long_name_to_do_some_tests_Now_let's_add_some_text_just_to_pass_the_129_chars_mark_and_trigger_the_chars_limit_of_the_default_naming_function.html"
    file_name_link = re.sub("[<>:'/\\|?*\0 ]", "_", link[:129])
    file_name_hash = hashlib.md5(f"{link}".encode("utf-8")).hexdigest()
    expected_crawled_file_path = f"{tmp_path}/{file_name_link}_{file_name_hash[-6:]}.json"

    paths = crawler.crawl(urls=[link], crawler_depth=0)

    assert os.path.exists(paths[0])
    assert paths[0] == Path(expected_crawled_file_path)


def test_crawler_naming_function(test_url, tmp_path):
    crawler = Crawler(
        output_dir=tmp_path, crawler_naming_function=lambda link, text: re.sub("[<>:'/\\|?*\0 ]", "_", link)
    )

    link = f"{test_url}/page_dynamic.html"
    file_name_link = re.sub("[<>:'/\\|?*\0 ]", "_", link)
    expected_crawled_file_path = tmp_path / f"{file_name_link}.json"

    paths = crawler.crawl(urls=[test_url + "/page_dynamic.html"], crawler_depth=0)

    assert os.path.exists(paths[0])
    assert paths[0] == expected_crawled_file_path
