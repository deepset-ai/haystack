from typing import Optional
from unittest.mock import Mock, patch
import logging
import pytest
import requests
from requests import Response

from haystack import Document
from haystack.nodes import LinkContentFetcher


@pytest.fixture
def mocked_requests():
    with patch("haystack.nodes.retriever.link_content.requests") as mock_requests:
        mock_response = Mock()
        mock_requests.get.return_value = mock_response
        mock_response.status_code = 200
        mock_response.text = "Sample content from webpage"
        mock_response.headers = {"Content-Type": "text/html"}
        yield mock_requests


@pytest.fixture
def mocked_article_extractor():
    with patch("boilerpy3.extractors.ArticleExtractor.get_content", return_value="Sample content from webpage"):
        yield


@pytest.mark.unit
def test_init():
    """
    Checks the initialization of the LinkContentFetcher without a preprocessor.
    """
    r = LinkContentFetcher()

    assert r.processor is None
    assert isinstance(r.handlers, dict)
    assert "text/html" in r.handlers
    assert "application/pdf" in r.handlers


@pytest.mark.unit
def test_init_with_preprocessor():
    """
    Checks the initialization of the LinkContentFetcher with a preprocessor.
    """
    pre_processor_mock = Mock()
    r = LinkContentFetcher(processor=pre_processor_mock)
    assert r.processor == pre_processor_mock
    assert isinstance(r.handlers, dict)
    assert "text/html" in r.handlers
    assert "application/pdf" in r.handlers


@pytest.mark.unit
def test_init_with_content_handlers():
    """
    Checks the initialization of the LinkContentFetcher content handlers.
    """

    def fake_but_valid_video_content_handler(response: Response) -> Optional[str]:
        pass

    r = LinkContentFetcher(content_handlers={"video/mp4": fake_but_valid_video_content_handler})

    assert isinstance(r.handlers, dict)
    assert "text/html" in r.handlers
    assert "application/pdf" in r.handlers
    assert "video/mp4" in r.handlers


@pytest.mark.unit
def test_init_with_content_handlers_override():
    """
    Checks the initialization of the LinkContentFetcher content handlers but with pdf handler overridden.
    """

    def new_pdf_content_handler(response: Response) -> Optional[str]:
        pass

    r = LinkContentFetcher(content_handlers={"application/pdf": new_pdf_content_handler})

    assert isinstance(r.handlers, dict)
    assert "text/html" in r.handlers
    assert "application/pdf" in r.handlers
    assert r.handlers["application/pdf"] == new_pdf_content_handler


@pytest.mark.unit
def test_init_with_invalid_content_handlers():
    """
    Checks the initialization of the LinkContentFetcher content handlers fails with invalid content handlers.
    """

    # invalid because it does not have the correct signature
    def invalid_video_content_handler() -> Optional[str]:
        pass

    with pytest.raises(ValueError, match="handler must accept"):
        LinkContentFetcher(content_handlers={"video/mp4": invalid_video_content_handler})


@pytest.mark.unit
def test_fetch(mocked_requests, mocked_article_extractor):
    """
    Checks if the LinkContentFetcher is able to fetch content.
    """
    url = "https://haystack.deepset.ai/"

    pre_processor_mock = Mock()
    pre_processor_mock.process.return_value = [Document("Sample content from webpage")]

    r = LinkContentFetcher(processor=pre_processor_mock)
    result = r.fetch(url=url, doc_kwargs={"text": "Sample content from webpage"})

    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].content == "Sample content from webpage"


@pytest.mark.unit
def test_fetch_no_url(mocked_requests, mocked_article_extractor):
    """
    Ensures an InvalidURL exception is raised when URL is missing.
    """
    pre_processor_mock = Mock()
    pre_processor_mock.process.return_value = [Document("Sample content from webpage")]

    retriever_no_url = LinkContentFetcher(processor=pre_processor_mock)
    with pytest.raises(requests.exceptions.InvalidURL, match="Invalid or missing URL"):
        retriever_no_url.fetch(url="")


@pytest.mark.unit
def test_fetch_invalid_url(caplog, mocked_requests, mocked_article_extractor):
    """
    Ensures an InvalidURL exception is raised when the URL is invalid.
    """
    url = "this-is-invalid-url"

    r = LinkContentFetcher()
    with pytest.raises(requests.exceptions.InvalidURL):
        r.fetch(url=url)


@pytest.mark.unit
def test_fetch_no_preprocessor(mocked_requests, mocked_article_extractor):
    """
    Checks if the LinkContentFetcher can fetch content without a preprocessor.
    """
    url = "https://www.example.com"
    r = LinkContentFetcher()

    result_no_preprocessor = r.fetch(url=url)

    assert len(result_no_preprocessor) == 1
    assert isinstance(result_no_preprocessor[0], Document)
    assert result_no_preprocessor[0].content == "Sample content from webpage"


@pytest.mark.unit
def test_fetch_correct_arguments(mocked_requests, mocked_article_extractor):
    """
    Ensures that requests.get is called with correct arguments.
    """
    url = "https://www.example.com"

    r = LinkContentFetcher()
    r.fetch(url=url)

    # Check the arguments that requests.get was called with
    args, kwargs = mocked_requests.get.call_args
    assert args[0] == url
    assert kwargs["timeout"] == 3
    assert kwargs["headers"] == r._REQUEST_HEADERS

    # another variant
    url = "https://deepset.ai"
    r.fetch(url=url, timeout=10)
    # Check the arguments that requests.get was called with
    args, kwargs = mocked_requests.get.call_args
    assert args[0] == url
    assert kwargs["timeout"] == 10
    assert kwargs["headers"] == r._REQUEST_HEADERS


@pytest.mark.unit
def test_fetch_default_empty_content(mocked_requests):
    """
    Checks handling of content extraction returning empty content.
    """
    url = "https://www.example.com"
    timeout = 10
    content_text = ""
    r = LinkContentFetcher()

    with patch("boilerpy3.extractors.ArticleExtractor.get_content", return_value=content_text):
        result = r.fetch(url=url, timeout=timeout)

    assert "text" not in result
    assert isinstance(result, list) and len(result) == 0


@pytest.mark.unit
def test_fetch_exception_during_content_extraction_no_raise_on_failure(caplog, mocked_requests):
    """
    Checks the behavior when there's an exception during content extraction, and raise_on_failure is set to False.
    """
    caplog.set_level(logging.WARNING)
    url = "https://www.example.com"
    r = LinkContentFetcher()

    with patch("boilerpy3.extractors.ArticleExtractor.get_content", side_effect=Exception("Could not extract content")):
        result = r.fetch(url=url)

    assert "text" not in result
    assert "failed to extract content from" in caplog.text


@pytest.mark.unit
def test_fetch_exception_during_content_extraction_raise_on_failure(caplog, mocked_requests):
    """
    Checks the behavior when there's an exception during content extraction, and raise_on_failure is set to True.
    """
    caplog.set_level(logging.WARNING)
    url = "https://www.example.com"
    r = LinkContentFetcher(raise_on_failure=True)

    with patch("boilerpy3.extractors.ArticleExtractor.get_content", side_effect=Exception("Could not extract content")):
        with pytest.raises(Exception, match="Could not extract content"):
            r.fetch(url=url)


@pytest.mark.unit
def test_fetch_exception_during_request_get_no_raise_on_failure(caplog):
    """
    Checks the behavior when there's an exception during request.get, and raise_on_failure is set to False.
    """
    caplog.set_level(logging.WARNING)
    url = "https://www.example.com"
    r = LinkContentFetcher()

    with patch("haystack.nodes.retriever.link_content.requests.get", side_effect=requests.RequestException()):
        r.fetch(url=url)
    assert f"Couldn't retrieve content from {url}" in caplog.text


@pytest.mark.unit
def test_fetch_exception_during_request_get_raise_on_failure(caplog):
    """
    Checks the behavior when there's an exception during request.get, and raise_on_failure is set to True.
    """
    caplog.set_level(logging.WARNING)
    url = "https://www.example.com"
    r = LinkContentFetcher(raise_on_failure=True)

    with patch("haystack.nodes.retriever.link_content.requests.get", side_effect=requests.RequestException()):
        with pytest.raises(requests.RequestException):
            r.fetch(url=url)


@pytest.mark.unit
@pytest.mark.parametrize("error_code", [403, 404, 500])
def test_handle_various_response_errors(caplog, mocked_requests, error_code: int):
    """
    Tests the handling of various HTTP error responses.
    """
    caplog.set_level(logging.WARNING)

    url = "https://some-problematic-url.com"

    # we don't throw exceptions, there might be many of them
    # we log them on debug level
    mock_response = Response()
    mock_response.status_code = error_code
    mocked_requests.get.return_value = mock_response

    r = LinkContentFetcher()
    docs = r.fetch(url=url)

    assert f"Couldn't retrieve content from {url}" in caplog.text
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].content == ""


@pytest.mark.unit
@pytest.mark.parametrize("error_code", [403, 404, 500])
def test_handle_http_error(mocked_requests, error_code: int):
    """
    Checks the behavior when there's an HTTPError raised, and raise_on_failure is set to True.
    """

    url = "https://some-problematic-url.com"

    # we don't throw exceptions, there might be many of them
    # we log them on debug level
    mock_response = Response()
    mock_response.status_code = error_code
    mocked_requests.get.return_value = mock_response

    r = LinkContentFetcher(raise_on_failure=True)
    with pytest.raises(requests.HTTPError):
        r.fetch(url=url)


@pytest.mark.unit
def test_is_valid_url():
    """
    Checks the _is_valid_url function with a set of valid URLs.
    """
    retriever = LinkContentFetcher()

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
        assert retriever._is_valid_url(url), f"Expected {url} to be valid"


@pytest.mark.unit
def test_is_invalid_url():
    """
    Checks the _is_valid_url function with a set of invalid URLs.
    """
    retriever = LinkContentFetcher()

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
        "",
        None,
    ]

    for url in invalid_urls:
        assert not retriever._is_valid_url(url), f"Expected {url} to be invalid"


@pytest.mark.unit
def test_switch_user_agent_on_failed_request():
    """
    Test that LinkContentFetcher switches user agents on failed requests
    """
    url = "http://fakeurl.com"
    retry_attempts = 2
    lc = LinkContentFetcher(user_agents=["ua1", "ua2"], retry_attempts=retry_attempts)
    with patch("haystack.nodes.retriever.link_content.requests.get") as mocked_get:
        mocked_get.return_value.raise_for_status.side_effect = requests.HTTPError()
        lc._get_response(url)

    assert mocked_get.call_count == retry_attempts
    assert mocked_get.call_args_list[0][1]["headers"]["User-Agent"] == "ua1"
    assert mocked_get.call_args_list[1][1]["headers"]["User-Agent"] == "ua2"


@pytest.mark.unit
def test_valid_requests_dont_switch_agent(mocked_requests):
    """
    Test that LinkContentFetcher doesn't switch user agents on valid requests
    """
    lcf = LinkContentFetcher()

    # Make first valid request
    lcf._get_response("http://example.com")

    # Make second valid request
    lcf._get_response("http://example.com")

    # Assert that requests.get was called twice with the same default user agents
    assert mocked_requests.get.call_count == 2
    assert (
        mocked_requests.get.call_args_list[0][1]["headers"]["User-Agent"]
        == mocked_requests.get.call_args_list[1][1]["headers"]["User-Agent"]
    )


@pytest.mark.integration
def test_call_with_valid_url_on_live_web():
    """
    Test that LinkContentFetcher can fetch content from a valid URL
    """

    retriever = LinkContentFetcher()
    docs = retriever.fetch(url="https://docs.haystack.deepset.ai/", timeout=2)

    assert len(docs) >= 1
    assert isinstance(docs[0], Document)
    assert "Haystack" in docs[0].content


@pytest.mark.integration
def test_retrieve_with_valid_url_on_live_web():
    """
    Test that LinkContentFetcher can fetch content from a valid URL using the run method
    """

    retriever = LinkContentFetcher()
    docs, _ = retriever.run(query="https://docs.haystack.deepset.ai/")
    docs = docs["documents"]

    assert len(docs) >= 1
    assert isinstance(docs[0], Document)
    assert "Haystack" in docs[0].content


@pytest.mark.integration
def test_retrieve_with_invalid_url():
    """
    Test that LinkContentFetcher raises ValueError when trying to fetch content from an invalid URL
    """

    retriever = LinkContentFetcher()
    with pytest.raises(ValueError):
        retriever.run(query="")


@pytest.mark.integration
def test_retrieve_batch():
    """
    Test that LinkContentFetcher can fetch content from a valid URL using the retrieve_batch method
    """

    retriever = LinkContentFetcher()
    docs, _ = retriever.run_batch(queries=["https://docs.haystack.deepset.ai/", "https://deepset.ai/"])
    assert docs

    docs = docs["documents"]
    # no processor is applied, so each query should return a list of documents with one entry
    assert len(docs) == 2 and len(docs[0]) == 1 and len(docs[1]) == 1

    # each query should return a list of documents
    assert isinstance(docs[0], list) and isinstance(docs[0][0], Document)
    assert isinstance(docs[1], list) and isinstance(docs[1][0], Document)
