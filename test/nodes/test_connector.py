from typing import List

import json
from pathlib import Path
import re
import hashlib
import os
from unittest.mock import patch

import pytest

from selenium.webdriver.common.by import By

from haystack.nodes.connector.crawler import Crawler
from haystack.schema import Document


@pytest.fixture()
def test_url(samples_path):
    return (samples_path / "crawler").absolute().as_uri()


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


@pytest.mark.unit
@patch("haystack.nodes.connector.crawler.webdriver")
def test_crawler_url_none_exception(webdriver):
    crawler = Crawler()
    with pytest.raises(ValueError):
        crawler.crawl()


@pytest.mark.integration
def test_crawler(tmp_path):
    tmp_dir = tmp_path
    url = ["https://haystack.deepset.ai/"]

    crawler = Crawler(output_dir=tmp_dir, file_path_meta_field_name="file_path")

    documents = crawler.crawl(urls=url, crawler_depth=0)
    docs_path = [Path(doc.meta["file_path"]) for doc in documents]

    results, _ = crawler.run(urls=url, crawler_depth=0)
    docs_result = results["documents"]

    for json_file, document in zip(docs_path, docs_result):
        assert isinstance(json_file, Path)
        assert isinstance(document, Document)

        with open(json_file.absolute(), "r") as read_file:
            file_content = json.load(read_file)
            assert file_content["meta"] == document.meta
            assert file_content["content"] == document.content


@pytest.mark.integration
def test_crawler_depth_0_single_url(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path, crawler_depth=0, file_path_meta_field_name="file_path")
    documents = crawler.crawl(urls=[test_url + "/index.html"])
    assert len(documents) == 1
    assert content_match(crawler, test_url + "/index.html", documents[0].meta["file_path"])


@pytest.mark.integration
def test_crawler_depth_0_many_urls(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path, file_path_meta_field_name="file_path")
    _urls = [test_url + "/index.html", test_url + "/page1.html"]
    documents = crawler.crawl(urls=_urls, crawler_depth=0)
    assert len(documents) == 2
    paths = [doc.meta["file_path"] for doc in documents]
    assert content_in_results(crawler, test_url + "/index.html", paths)
    assert content_in_results(crawler, test_url + "/page1.html", paths)


@pytest.mark.integration
def test_crawler_depth_1_single_url(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path, file_path_meta_field_name="file_path")
    documents = crawler.crawl(urls=[test_url + "/index.html"], crawler_depth=1)
    assert len(documents) == 3
    paths = [doc.meta["file_path"] for doc in documents]
    assert content_in_results(crawler, test_url + "/index.html", paths)
    assert content_in_results(crawler, test_url + "/page1.html", paths)
    assert content_in_results(crawler, test_url + "/page2.html", paths)


@pytest.mark.integration
def test_crawler_output_file_structure(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path, file_path_meta_field_name="file_path")
    documents = crawler.crawl(urls=[test_url + "/index.html"], crawler_depth=0)
    path = Path(documents[0].meta["file_path"])
    assert content_match(crawler, test_url + "/index.html", path)

    with open(path.absolute(), "r") as doc_file:
        data = json.load(doc_file)
        assert "content" in data
        assert "meta" in data
        assert isinstance(data["content"], str)
        assert len(data["content"].split()) > 2


@pytest.mark.integration
def test_crawler_filter_urls(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path, file_path_meta_field_name="file_path")

    documents = crawler.crawl(urls=[test_url + "/index.html"], filter_urls=["index"], crawler_depth=1)
    assert len(documents) == 1
    assert content_match(crawler, test_url + "/index.html", documents[0].meta["file_path"])


@pytest.mark.integration
def test_crawler_extract_hidden_text(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    documents, _ = crawler.run(urls=[test_url + "/page_w_hidden_text.html"], extract_hidden_text=True, crawler_depth=0)
    crawled_content = documents["documents"][0].content
    assert "hidden text" in crawled_content

    documents, _ = crawler.run(urls=[test_url + "/page_w_hidden_text.html"], extract_hidden_text=False, crawler_depth=0)
    crawled_content = documents["documents"][0].content
    assert "hidden text" not in crawled_content


@pytest.mark.integration
def test_crawler_loading_wait_time(test_url, tmp_path, samples_path):
    loading_wait_time = 3
    crawler = Crawler(output_dir=tmp_path, file_path_meta_field_name="file_path")
    documents = crawler.crawl(
        urls=[test_url + "/page_dynamic.html"], crawler_depth=1, loading_wait_time=loading_wait_time
    )

    assert len(documents) == 4

    paths = [doc.meta["file_path"] for doc in documents]

    with open(f"{samples_path.absolute()}/crawler/page_dynamic_result.txt", "r") as dynamic_result:
        dynamic_result_text = dynamic_result.readlines()
        for path in paths:
            with open(path, "r") as crawled_file:
                page_data = json.load(crawled_file)
                if page_data["meta"]["url"] == test_url + "/page_dynamic.html":
                    content = page_data["content"].split("\n")

                    print(page_data["content"])
                    print("------")

                    for line in range(len(dynamic_result_text)):
                        assert dynamic_result_text[line].strip() == content[line].strip()

    assert content_in_results(crawler, test_url + "/index.html", paths)
    assert content_in_results(crawler, test_url + "/page1.html", paths)
    assert content_in_results(crawler, test_url + "/page2.html", paths)


@pytest.mark.integration
def test_crawler_default_naming_function(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path, file_path_meta_field_name="file_path")

    link = f"{test_url}/page_with_a_very_long_name_to_do_some_tests_Now_let's_add_some_text_just_to_pass_the_129_chars_mark_and_trigger_the_chars_limit_of_the_default_naming_function.html"
    file_name_link = re.sub("[<>:'/\\|?*\0 ]", "_", link[:129])
    file_name_hash = hashlib.md5(f"{link}".encode("utf-8")).hexdigest()
    expected_crawled_file_path = f"{tmp_path}/{file_name_link}_{file_name_hash[-6:]}.json"

    documents = crawler.crawl(urls=[link], crawler_depth=0)

    path = Path(documents[0].meta["file_path"])
    assert os.path.exists(path)
    assert path == Path(expected_crawled_file_path)


@pytest.mark.integration
def test_crawler_naming_function(test_url, tmp_path):
    crawler = Crawler(
        output_dir=tmp_path,
        file_path_meta_field_name="file_path",
        crawler_naming_function=lambda link, text: re.sub("[<>:'/\\|?*\0 ]", "_", link),
    )

    link = f"{test_url}/page_dynamic.html"
    file_name_link = re.sub("[<>:'/\\|?*\0 ]", "_", link)
    expected_crawled_file_path = tmp_path / f"{file_name_link}.json"

    documents = crawler.crawl(urls=[test_url + "/page_dynamic.html"], crawler_depth=0)
    path = Path(documents[0].meta["file_path"])
    assert os.path.exists(path)
    assert path == expected_crawled_file_path


@pytest.mark.integration
def test_crawler_not_save_file(test_url):
    crawler = Crawler()
    documents = crawler.crawl(urls=[test_url + "/index.html"], crawler_depth=0)
    assert documents[0].meta.get("file_path", None) is None


@pytest.mark.integration
def test_crawler_custom_meta_file_path_name(test_url, tmp_path):
    crawler = Crawler()
    documents = crawler.crawl(
        urls=[test_url + "/index.html"], crawler_depth=0, output_dir=tmp_path, file_path_meta_field_name="custom"
    )
    assert documents[0].meta.get("custom", None) is not None


@pytest.mark.integration
def test_crawler_depth_2_single_url(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path, file_path_meta_field_name="file_path")
    documents = crawler.crawl(urls=[test_url + "/index.html"], crawler_depth=2)
    assert len(documents) == 6
    paths = [doc.meta["file_path"] for doc in documents]
    assert content_in_results(crawler, test_url + "/index.html", paths)
    assert content_in_results(crawler, test_url + "/page1.html", paths)
    assert content_in_results(crawler, test_url + "/page2.html", paths)
    assert content_in_results(crawler, test_url + "/page1_subpage1.html", paths)
    assert content_in_results(crawler, test_url + "/page1_subpage2.html", paths)
    assert content_in_results(crawler, test_url + "/page2_subpage1.html", paths)


@pytest.mark.integration
def test_crawler_depth_2_multiple_urls(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path, file_path_meta_field_name="file_path")
    _urls = [test_url + "/index.html", test_url + "/page1.html"]
    documents = crawler.crawl(urls=_urls, crawler_depth=2)
    assert len(documents) == 6
    paths = [doc.meta["file_path"] for doc in documents]
    assert content_in_results(crawler, test_url + "/index.html", paths)
    assert content_in_results(crawler, test_url + "/page1.html", paths)
    assert content_in_results(crawler, test_url + "/page2.html", paths)
    assert content_in_results(crawler, test_url + "/page1_subpage1.html", paths)
    assert content_in_results(crawler, test_url + "/page1_subpage2.html", paths)
    assert content_in_results(crawler, test_url + "/page2_subpage1.html", paths)
