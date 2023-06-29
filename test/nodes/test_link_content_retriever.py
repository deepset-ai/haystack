from unittest.mock import Mock, patch
import logging
import pytest
import requests
from requests import Response

from haystack import Document
from haystack.nodes import LinkContentRetriever


@pytest.fixture
def mocked_requests():
    with patch("haystack.nodes.retriever.link_content.requests") as mock_requests:
        mock_response = Mock()
        mock_requests.get.return_value = mock_response
        mock_response.status_code = 200
        mock_response.text = "Sample content from webpage"
        yield mock_requests


@pytest.fixture
def mocked_article_extractor():
    with patch("boilerpy3.extractors.ArticleExtractor.get_content", return_value="Sample content from webpage"):
        yield


@pytest.mark.unit
def test_init():
    r = LinkContentRetriever()

    assert r.pre_processor is None
    assert isinstance(r.content_handlers, dict)
    assert "html" in r.content_handlers


@pytest.mark.unit
def test_init_with_preprocessor():
    pre_processor_mock = Mock()
    r = LinkContentRetriever(pre_processor=pre_processor_mock)
    assert r.pre_processor == pre_processor_mock
    assert isinstance(r.content_handlers, dict)
    assert "html" in r.content_handlers


@pytest.mark.unit
def test_call(mocked_requests, mocked_article_extractor):
    url = "https://haystack.deepset.ai/"

    pre_processor_mock = Mock()
    pre_processor_mock.process.return_value = [Document("Sample content from webpage")]

    r = LinkContentRetriever(pre_processor_mock)
    result = r(url=url, doc_kwargs={"text": "Sample content from webpage"})

    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].content == "Sample content from webpage"


@pytest.mark.unit
def test_call_no_url(mocked_requests, mocked_article_extractor):
    pre_processor_mock = Mock()
    pre_processor_mock.process.return_value = [Document("Sample content from webpage")]

    reader_no_url = LinkContentRetriever(pre_processor=pre_processor_mock)
    with pytest.raises(requests.exceptions.InvalidURL, match="Invalid or missing URL"):
        reader_no_url(url="")


@pytest.mark.unit
def test_call_invalid_url(caplog, mocked_requests, mocked_article_extractor):
    url = "invalid-url"

    r = LinkContentRetriever()
    with pytest.raises(requests.exceptions.InvalidURL):
        r(url=url)


@pytest.mark.unit
def test_call_no_preprocessor(mocked_requests, mocked_article_extractor):
    url = "https://www.example.com"
    r = LinkContentRetriever()

    result_no_preprocessor = r(url=url)

    assert len(result_no_preprocessor) == 1
    assert isinstance(result_no_preprocessor[0], Document)
    assert result_no_preprocessor[0].content == "Sample content from webpage"


@pytest.mark.unit
def test_call_correct_arguments(mocked_requests, mocked_article_extractor):
    url = "https://www.example.com"

    r = LinkContentRetriever()
    r(url=url)

    # Check the arguments that requests.get was called with
    args, kwargs = mocked_requests.get.call_args
    assert args[0] == url
    assert kwargs["timeout"] == 3
    assert kwargs["headers"] == r._request_headers()

    # another variant
    url = "https://deepset.ai"
    r(url=url, timeout=10)
    # Check the arguments that requests.get was called with
    args, kwargs = mocked_requests.get.call_args
    assert args[0] == url
    assert kwargs["timeout"] == 10
    assert kwargs["headers"] == r._request_headers()


@pytest.mark.unit
def test_fetch_default_empty_content(mocked_requests):
    url = "https://www.example.com"
    timeout = 10
    content_text = ""

    with patch("boilerpy3.extractors.ArticleExtractor.get_content", return_value=content_text):
        r = LinkContentRetriever()
        result = r(url=url, timeout=timeout)

    assert "text" not in result
    assert isinstance(result, list) and len(result) == 0


@pytest.mark.unit
def test_fetch_exception_during_content_extraction(caplog, mocked_requests):
    caplog.set_level(logging.DEBUG)
    url = "https://www.example.com"

    with patch("boilerpy3.extractors.ArticleExtractor.get_content", side_effect=Exception("Could not extract content")):
        r = LinkContentRetriever()
        result = r(url=url)

    assert "text" not in result
    assert "Couldn't extract content from URL" in caplog.text


@pytest.mark.unit
def test_fetch_exception_during_request_get(caplog):
    caplog.set_level(logging.DEBUG)
    url = "https://www.example.com"

    with patch("haystack.nodes.retriever.link_content.requests.get", side_effect=requests.RequestException()):
        r = LinkContentRetriever()
        r(url=url)
    assert "Error retrieving URL" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("error_code", [403, 404, 500])
def test_handle_various_response_errors(caplog, mocked_requests, error_code: int):
    caplog.set_level(logging.DEBUG)

    url = "https://some-problematic-url.com"

    # we don't throw exceptions, there might be many of them
    # we log them on debug level
    mock_response = Response()
    mock_response.status_code = error_code
    mocked_requests.get.return_value = mock_response

    r = LinkContentRetriever()
    r(url=url)

    assert f"Error retrieving URL {url}: Status Code - {error_code}" in caplog.text


@pytest.mark.unit
def test_is_valid_url():
    reader = LinkContentRetriever()

    valid_urls = [
        "http://www.google.com",
        "https://www.google.com",
        "http://google.com",
        "https://google.com",
        "http://localhost",
        "https://localhost",
        "http://127.0.0.1",
        "https://127.0.0.1",
        "http://[::1]",
        "https://[::1]",
        "http://example.com/path/to/page?name=value",
        "https://example.com/path/to/page?name=value",
        "http://example.com:8000",
        "https://example.com:8000",
    ]

    for url in valid_urls:
        assert reader._is_valid_url(url), f"Expected {url} to be valid"


@pytest.mark.unit
def test_is_invalid_url():
    reader = LinkContentRetriever()

    invalid_urls = [
        "http://",
        "https://",
        "http:",
        "https:",
        "www.google.com",
        "google.com",
        "localhost",
        "127.0.0.1",
        "[::1]",
        "/path/to/page",
        "/path/to/page?name=value",
        ":8000",
        "example.com",
        "http:///example.com",
        "https:///example.com",
    ]

    for url in invalid_urls:
        assert not reader._is_valid_url(url), f"Expected {url} to be invalid"


@pytest.mark.integration
def test_call_with_valid_url_on_live_web():
    """
    Test that LinkContentReader can fetch content from a valid URL
    """

    reader = LinkContentRetriever()
    docs = reader(url="https://docs.haystack.deepset.ai/", timeout=2)

    assert len(docs) >= 1
    assert isinstance(docs[0], Document)
    assert "Haystack" in docs[0].content
