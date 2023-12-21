from unittest.mock import patch, Mock

import pytest
import requests

from haystack.components.fetchers.link_content import (
    LinkContentFetcher,
    text_content_handler,
    binary_content_handler,
    DEFAULT_USER_AGENT,
)

HTML_URL = "https://docs.haystack.deepset.ai/docs"
TEXT_URL = "https://raw.githubusercontent.com/deepset-ai/haystack/main/README.md"
PDF_URL = "https://raw.githubusercontent.com/deepset-ai/haystack/b5987a6d8d0714eb2f3011183ab40093d2e4a41a/e2e/samples/pipelines/sample_pdf_1.pdf"


@pytest.fixture
def mock_get_link_text_content():
    with patch("haystack.components.fetchers.link_content.requests") as mock_run:
        mock_run.get.return_value = Mock(
            status_code=200, text="Example test response", headers={"Content-Type": "text/plain"}
        )
        yield mock_run


@pytest.fixture
def mock_get_link_content(test_files_path):
    with patch("haystack.components.fetchers.link_content.requests") as mock_run:
        mock_run.get.return_value = Mock(
            status_code=200,
            content=open(test_files_path / "pdf" / "sample_pdf_1.pdf", "rb").read(),
            headers={"Content-Type": "application/pdf"},
        )
        yield mock_run


class TestLinkContentFetcher:
    def test_init(self):
        fetcher = LinkContentFetcher()
        assert fetcher.raise_on_failure is True
        assert fetcher.user_agents == [DEFAULT_USER_AGENT]
        assert fetcher.retry_attempts == 2
        assert fetcher.timeout == 3
        assert fetcher.handlers == {
            "text/html": text_content_handler,
            "text/plain": text_content_handler,
            "application/pdf": binary_content_handler,
            "application/octet-stream": binary_content_handler,
        }
        assert hasattr(fetcher, "_get_response")

    def test_init_with_params(self):
        fetcher = LinkContentFetcher(raise_on_failure=False, user_agents=["test"], retry_attempts=1, timeout=2)
        assert fetcher.raise_on_failure is False
        assert fetcher.user_agents == ["test"]
        assert fetcher.retry_attempts == 1
        assert fetcher.timeout == 2

    def test_run_text(self):
        correct_response = b"Example test response"
        with patch("haystack.components.fetchers.link_content.requests") as mock_run:
            mock_run.get.return_value = Mock(
                status_code=200, text="Example test response", headers={"Content-Type": "text/plain"}
            )
            fetcher = LinkContentFetcher()
            streams = fetcher.run(urls=["https://www.example.com"])["streams"]
            first_stream = streams[0]
            assert first_stream.data == correct_response
            assert first_stream.meta["content_type"] == "text/plain"

    def test_run_html(self):
        correct_response = b"<h1>Example test response</h1>"
        with patch("haystack.components.fetchers.link_content.requests") as mock_run:
            mock_run.get.return_value = Mock(
                status_code=200, text="<h1>Example test response</h1>", headers={"Content-Type": "text/html"}
            )
            fetcher = LinkContentFetcher()
            streams = fetcher.run(urls=["https://www.example.com"])["streams"]
            first_stream = streams[0]
            assert first_stream.data == correct_response
            assert first_stream.meta["content_type"] == "text/html"

    def test_run_binary(self, test_files_path):
        file_bytes = open(test_files_path / "pdf" / "sample_pdf_1.pdf", "rb").read()
        with patch("haystack.components.fetchers.link_content.requests") as mock_run:
            mock_run.get.return_value = Mock(
                status_code=200, content=file_bytes, headers={"Content-Type": "application/pdf"}
            )
            fetcher = LinkContentFetcher()
            streams = fetcher.run(urls=["https://www.example.com"])["streams"]
            first_stream = streams[0]
            assert first_stream.data == file_bytes
            assert first_stream.meta["content_type"] == "application/pdf"

    def test_run_bad_status_code(self):
        empty_byte_stream = b""
        fetcher = LinkContentFetcher(raise_on_failure=False)
        mock_response = Mock(status_code=403)
        with patch("haystack.components.fetchers.link_content.requests") as mock_run:
            mock_run.get.return_value = mock_response
            streams = fetcher.run(urls=["https://www.example.com"])["streams"]

        # empty byte stream is returned because raise_on_failure is False
        assert len(streams) == 1
        first_stream = streams[0]
        assert first_stream.data == empty_byte_stream
        assert first_stream.meta["content_type"] == "text/html"

    @pytest.mark.integration
    def test_link_content_fetcher_html(self):
        fetcher = LinkContentFetcher()
        streams = fetcher.run([HTML_URL])["streams"]
        first_stream = streams[0]
        assert "Haystack" in first_stream.data.decode("utf-8")
        assert first_stream.meta["content_type"] == "text/html"
        assert "url" in first_stream.meta and first_stream.meta["url"] == HTML_URL

    @pytest.mark.integration
    def test_link_content_fetcher_text(self):
        fetcher = LinkContentFetcher()
        streams = fetcher.run([TEXT_URL])["streams"]
        first_stream = streams[0]
        assert "Haystack" in first_stream.data.decode("utf-8")
        assert first_stream.meta["content_type"] == "text/plain"
        assert "url" in first_stream.meta and first_stream.meta["url"] == TEXT_URL

    @pytest.mark.integration
    def test_link_content_fetcher_pdf(self):
        fetcher = LinkContentFetcher()
        streams = fetcher.run([PDF_URL])["streams"]
        assert len(streams) == 1
        first_stream = streams[0]
        assert first_stream.meta["content_type"] in ("application/octet-stream", "application/pdf")
        assert "url" in first_stream.meta and first_stream.meta["url"] == PDF_URL

    @pytest.mark.integration
    def test_link_content_fetcher_multiple_different_content_types(self):
        """
        This test is to ensure that the fetcher can handle a list of URLs that contain different content types.
        """
        fetcher = LinkContentFetcher()
        streams = fetcher.run([PDF_URL, HTML_URL])["streams"]
        assert len(streams) == 2
        for stream in streams:
            assert stream.meta["content_type"] in ("text/html", "application/pdf", "application/octet-stream")
            if stream.meta["content_type"] == "text/html":
                assert "Haystack" in stream.data.decode("utf-8")
            elif stream.meta["content_type"] == "application/pdf":
                assert len(stream.data) > 0

    @pytest.mark.integration
    def test_link_content_fetcher_multiple_html_streams(self):
        """
        This test is to ensure that the fetcher can handle a list of URLs that contain different content types,
        and that we have two html streams.
        """

        fetcher = LinkContentFetcher()
        streams = fetcher.run([PDF_URL, HTML_URL, "https://google.com"])["streams"]
        assert len(streams) == 3
        for stream in streams:
            assert stream.meta["content_type"] in ("text/html", "application/pdf", "application/octet-stream")
            if stream.meta["content_type"] == "text/html":
                assert "Haystack" in stream.data.decode("utf-8") or "Google" in stream.data.decode("utf-8")
            elif stream.meta["content_type"] == "application/pdf":
                assert len(stream.data) > 0

    @pytest.mark.integration
    def test_mix_of_good_and_failed_requests(self):
        """
        This test is to ensure that the fetcher can handle a list of URLs that contain URLs that fail to be fetched.
        In such a case, the fetcher should return the content of the URLs that were successfully fetched and not raise
        an exception.
        """
        fetcher = LinkContentFetcher()
        result = fetcher.run(["https://non_existent_website_dot.com/", "https://www.google.com/"])
        assert len(result["streams"]) == 1
        first_stream = result["streams"][0]
        assert first_stream.meta["content_type"] == "text/html"

    @pytest.mark.integration
    def test_bad_request_exception_raised(self):
        """
        This test is to ensure that the fetcher raises an exception when a single bad request is made and it is configured to
        do so.
        """
        fetcher = LinkContentFetcher()
        with pytest.raises(requests.exceptions.ConnectionError):
            fetcher.run(["https://non_existent_website_dot.com/"])
