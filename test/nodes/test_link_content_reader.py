from unittest.mock import Mock, patch
import logging
import pytest
import requests
from requests.models import Response

from haystack import Document
from haystack.nodes import LinkContentReader


@pytest.fixture
def mocked_requests():
    with patch("haystack.nodes.retriever.link_content_reader.requests") as mock_requests:
        mock_response = Mock()
        mock_requests.get.return_value = mock_response
        mock_response.status_code = 200
        mock_response.text = "Sample content from webpage"
        yield mock_requests


@pytest.fixture
def mocked_article_extractor():
    with patch("boilerpy3.extractors.ArticleExtractor.get_content", return_value="Sample content from webpage"):
        yield


@pytest.fixture
def mock_head_response_factory():
    class MockResponseContext:
        def __init__(self, content_type):
            self.content_type = content_type

        def __enter__(self):
            self.mock_response = Response()
            self.mock_response.status_code = 200
            self.mock_response.headers = {"content-type": self.content_type}
            self.patcher = patch("requests.head", return_value=self.mock_response)
            self.patcher.start()

        def __exit__(self, *args):
            self.patcher.stop()

    def _mock_head_response(content_type):
        return MockResponseContext(content_type)

    return _mock_head_response


@pytest.mark.unit
def test_init():
    url = "https://www.example.com"
    pre_processor_mock = Mock()
    reader = LinkContentReader(url, pre_processor_mock)

    assert reader.url == url
    assert reader.pre_processor == pre_processor_mock
    assert isinstance(reader.handlers, dict)
    assert "default" in reader.handlers


@pytest.mark.unit
def test_init_with_preprocessor():
    # Initialize without a URL
    pre_processor_mock = Mock()
    reader_no_url = LinkContentReader(pre_processor=pre_processor_mock)
    assert reader_no_url.url is None
    assert reader_no_url.pre_processor == pre_processor_mock
    assert isinstance(reader_no_url.handlers, dict)
    assert "default" in reader_no_url.handlers


@pytest.mark.unit
def test_init_with_url():
    # Initialize without a preprocessor
    url = "https://haystack.deepset.ai/"
    reader_no_preprocessor = LinkContentReader(url)
    assert reader_no_preprocessor.url == url
    assert reader_no_preprocessor.pre_processor is None
    assert isinstance(reader_no_preprocessor.handlers, dict)
    assert "default" in reader_no_preprocessor.handlers


@pytest.mark.unit
def test_call(mocked_requests, mocked_article_extractor):
    # Mock the PreProcessor
    pre_processor_mock = Mock()
    pre_processor_mock.process.return_value = [Document("Sample content from webpage")]

    url = "https://www.example.com"
    reader = LinkContentReader(url, pre_processor_mock)
    result = reader(doc_kwargs={"text": "Sample content from webpage"})

    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].content == "Sample content from webpage"


@pytest.mark.unit
def test_call_no_url(mocked_requests, mocked_article_extractor):
    # Mock the PreProcessor
    pre_processor_mock = Mock()
    pre_processor_mock.process.return_value = [Document("Sample content from webpage")]

    # Test without a URL
    reader_no_url = LinkContentReader(pre_processor=pre_processor_mock)
    with pytest.raises(ValueError):
        reader_no_url()


@pytest.mark.unit
def test_call_no_preprocessor(mocked_requests, mocked_article_extractor):
    # Test without a pre_processor
    url = "https://www.example.com"
    reader_no_preprocessor = LinkContentReader(url)

    result_no_preprocessor = reader_no_preprocessor()

    assert len(result_no_preprocessor) == 1
    assert isinstance(result_no_preprocessor[0], Document)
    assert result_no_preprocessor[0].content == "Sample content from webpage"


@pytest.mark.unit
def test_call_correct_arguments(mocked_requests, mocked_article_extractor):
    url = "https://www.example.com"

    reader = LinkContentReader(url)
    reader()

    # Check the arguments that requests.get was called with
    args, kwargs = mocked_requests.get.call_args
    assert args[0] == url
    assert kwargs["timeout"] == 3
    assert kwargs["headers"] == reader.request_headers()

    # another variant
    url = "https://deepset.ai"
    reader(url=url, timeout=10)
    # Check the arguments that requests.get was called with
    args, kwargs = mocked_requests.get.call_args
    assert args[0] == url
    assert kwargs["timeout"] == 10
    assert kwargs["headers"] == reader.request_headers()


@pytest.mark.unit
def test_fetch_default_valid_url(mocked_requests, mocked_article_extractor):
    url = "https://www.example.com"
    timeout = 10

    reader = LinkContentReader(url=url)
    result = reader.fetch_default(url, timeout)

    assert result["text"] == "Sample content from webpage"
    assert result["url"] == url


@pytest.mark.unit
def test_fetch_default_invalid_url(caplog, mocked_requests, mocked_article_extractor):
    url = "invalid-url"

    reader = LinkContentReader(url=url)
    result = reader.fetch_default(url)

    assert result == {"url": url}
    assert "Invalid URL" in caplog.text


@pytest.mark.unit
def test_fetch_default_empty_content(mocked_requests):
    url = "https://www.example.com"
    timeout = 10
    content_text = ""

    with patch("boilerpy3.extractors.ArticleExtractor.get_content", return_value=content_text):
        reader = LinkContentReader(url=url)
        result = reader.fetch_default(url, timeout)

    assert "text" not in result
    assert result.get("url") == url


@pytest.mark.unit
def test_fetch_exception_during_content_extraction(caplog, mocked_requests):
    caplog.set_level(logging.DEBUG)
    url = "https://www.example.com"

    with patch("boilerpy3.extractors.ArticleExtractor.get_content", side_effect=Exception("Could not extract content")):
        reader = LinkContentReader(url=url)
        result = reader.fetch_default(url)

    assert "text" not in result
    assert result.get("url") == url
    assert "Couldn't extract content from URL" in caplog.text


@pytest.mark.unit
def test_fetch_exception_during_request_get(caplog):
    caplog.set_level(logging.DEBUG)
    url = "https://www.example.com"

    with patch("haystack.nodes.retriever.link_content_reader.requests.get", side_effect=requests.RequestException()):
        reader = LinkContentReader(url=url)
        result = reader.fetch_default(url)

    assert "text" not in result
    assert result.get("url") == url
    assert "Error retrieving URL" in caplog.text


@pytest.mark.unit
def test_get_handler_pdf_content_type(mock_head_response_factory):
    # we only support html for now, pdf and other types will be added later
    # default handler should be used until we add support for other types
    url = "http://example.com/some.pdf"
    handler_expected = "pdf"
    reader = LinkContentReader()
    with mock_head_response_factory("application/pdf"):
        handler = reader.get_handler(url)
    assert handler == handler_expected


@pytest.mark.unit
def test_get_handler_html_content_type(mock_head_response_factory):
    # we only support html for now, pdf and other types will be added later
    # default handler should be used until we add support for other types
    url = "http://example.com/some.html"
    handler_expected = "default"
    reader = LinkContentReader()
    with mock_head_response_factory("text/html"):
        handler = reader.get_handler(url)
    assert handler == handler_expected


@pytest.mark.unit
def test_get_handler_no_content_type(mock_head_response_factory):
    # if we can't get content type from the response, default handler should be used
    url = "http://example.com/no_content_type"
    handler_expected = "default"

    reader = LinkContentReader()
    with mock_head_response_factory(""):
        handler = reader.get_handler(url)
    assert handler == handler_expected


@pytest.mark.unit
def test_is_valid_url():
    reader = LinkContentReader()

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
        assert reader.is_valid_url(url), f"Expected {url} to be valid"


@pytest.mark.unit
def test_is_invalid_url():
    reader = LinkContentReader()

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
        assert not reader.is_valid_url(url), f"Expected {url} to be invalid"


@pytest.mark.integration
def test_call_with_valid_url():
    """
    Test that LinkContentReader can fetch content from a valid URL
    """

    reader = LinkContentReader(url="https://docs.haystack.deepset.ai/")
    docs = reader(timeout=2)

    assert len(docs) >= 1
    assert isinstance(docs[0], Document)
    assert "Haystack" in docs[0].content
