import os
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from haystack.nodes.connector import Crawler
from haystack.schema import Document


TEST_HOME_PAGE = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Home Page for Crawler</title>
</head>
<body>
    <p>test page content</p>
    <a href="BASE_URL/page1">page 1</a>
    <a href="BASE_URL/page2">page 2</a>
</body>
</html>
"""

TEST_PAGE1 = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Page 1 for Crawler</title>
</head>
<body>
    <p>test page 1 content</p>
    <a href="BASE_URL">home</a>
    <a href="BASE_URL/page2">page 2</a>
</body>
</html>
"""
TEST_PAGE2 = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Page 2 for Crawler</title>
</head>
<body>
    <p>test page 2 content</p>
    <a href="BASE_URL">home</a>
    <a href="BASE_URL/page1">page 1</a>
</body>
</html>
"""


@pytest.fixture(scope="session")
def test_url():
    tmpdir = Path(tempfile.mkdtemp())
    base_url = tmpdir / "haystack_test_webpages"
    os.mkdir(base_url)

    with open(base_url/"index.html", 'w') as page:
        page.write(TEST_HOME_PAGE.replace("BASE_URL", str(base_url)))

    with open(base_url/"page1.html", 'w') as page:
        page.write(TEST_PAGE1.replace("BASE_URL", str(base_url)))

    with open(base_url/"page2.html", 'w') as page:
        page.write(TEST_PAGE2.replace("BASE_URL", str(base_url)))

    yield f"file://{base_url.absolute()}"

    shutil.rmtree(tmpdir)


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


def test_crawler_depth_0_many_urls(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    _urls = [test_url + "/index.html", test_url + "/page1.html", test_url + "/page2.html"]
    paths = crawler.crawl(urls=_urls, crawler_depth=0)
    assert len(paths) == 3


def test_crawler_depth_1_single_url(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    paths = crawler.crawl(urls=[test_url + "/index.html"], crawler_depth=1)
    assert len(paths) == 3


def test_crawler_output_file_structure(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    paths = crawler.crawl(urls=[test_url + "/index.html"], crawler_depth=0)
    with open(paths[0].absolute(), "r") as doc_file:
        data = json.load(doc_file)
        assert "content" in data
        assert "meta" in data
        assert isinstance(data["content"], str)
        assert len(data["content"].split()) > 2


def test_crawler_filter_urls(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)

    print(crawler.crawl(urls=[test_url + "/index.html"], filter_urls=["page1"], crawler_depth=1))
    assert len(crawler.crawl(urls=[test_url + "/index.html"], filter_urls=["page1"], crawler_depth=1)) == 1
    assert not crawler.crawl(urls=[test_url + "/index.html"], filter_urls=["page3"], crawler_depth=1)
    assert not crawler.crawl(urls=[test_url + "/index.html"], filter_urls=["google\.com"], crawler_depth=1)


def test_crawler_content(test_url, tmp_path):
    expected_results = [
        {"url": test_url + "/index.html", "partial_content": "test page content"},
        {"url": test_url + "/page1.html", "partial_content": "test page 1 content"},
        {"url": test_url + "/page2.html", "partial_content": "test page 2 content"},
    ]

    crawler = Crawler(output_dir=tmp_path)
    for result in expected_results:
        paths = crawler.crawl(urls=[result["url"]], crawler_depth=0)
        with open(paths[0].absolute(), "r") as read_file:
            content = json.load(read_file)
            assert result["partial_content"] in content["content"]


def test_crawler_return_document(test_url, tmp_path):
    crawler = Crawler(output_dir=tmp_path)
    documents, _ = crawler.run(urls=[test_url + "/index.html"], crawler_depth=0, return_documents=True)
    paths, _ = crawler.run(urls=[test_url + "/index.html"], crawler_depth=0, return_documents=False)

    for path, document in zip(paths["paths"], documents["documents"]):
        with open(path.absolute(), "r") as doc_file:
            file_content = json.load(doc_file)
            assert file_content["meta"] == document.meta
            assert file_content["content"] == document.content
