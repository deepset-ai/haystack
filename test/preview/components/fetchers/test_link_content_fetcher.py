import io
from unittest.mock import patch, Mock

import pytest

from haystack.preview.components.fetchers.link_content import (
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
    with patch("haystack.preview.components.fetchers.link_content.requests") as mock_run:
        mock_run.get.return_value = Mock(
            status_code=200, text="Example test response", headers={"Content-Type": "text/plain"}
        )
        yield mock_run


@pytest.fixture
def mock_get_link_content(test_files_path):
    with patch("haystack.preview.components.fetchers.link_content.requests") as mock_run:
        mock_run.get.return_value = Mock(
            status_code=200,
            content=open(test_files_path / "pdf" / "sample_pdf_1.pdf", "rb").read(),
            headers={"Content-Type": "application/pdf"},
        )
        yield mock_run


class TestLinkContentFetcher:
    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_init_with_params(self):
        fetcher = LinkContentFetcher(raise_on_failure=False, user_agents=["test"], retry_attempts=1, timeout=2)
        assert fetcher.raise_on_failure is False
        assert fetcher.user_agents == ["test"]
        assert fetcher.retry_attempts == 1
        assert fetcher.timeout == 2

    @pytest.mark.unit
    def test_to_dict(self):
        fetcher = LinkContentFetcher()
        assert fetcher.to_dict() == {
            "type": "LinkContentFetcher",
            "init_parameters": {
                "raise_on_failure": True,
                "user_agents": [DEFAULT_USER_AGENT],
                "retry_attempts": 2,
                "timeout": 3,
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_params(self):
        fetcher = LinkContentFetcher(raise_on_failure=False, user_agents=["test"], retry_attempts=1, timeout=2)
        assert fetcher.to_dict() == {
            "type": "LinkContentFetcher",
            "init_parameters": {"raise_on_failure": False, "user_agents": ["test"], "retry_attempts": 1, "timeout": 2},
        }

    @pytest.mark.unit
    def test_from_dict(self):
        fetcher = LinkContentFetcher.from_dict(
            {
                "type": "LinkContentFetcher",
                "init_parameters": {
                    "raise_on_failure": False,
                    "user_agents": ["test"],
                    "retry_attempts": 1,
                    "timeout": 2,
                },
            }
        )
        assert fetcher.raise_on_failure is False
        assert fetcher.user_agents == ["test"]
        assert fetcher.retry_attempts == 1

    @pytest.mark.unit
    def test_run_text(self):
        with patch("haystack.preview.components.fetchers.link_content.requests") as mock_run:
            mock_run.get.return_value = Mock(
                status_code=200, text="Example test response", headers={"Content-Type": "text/plain"}
            )
            fetcher = LinkContentFetcher()
            document = fetcher.run("https://www.example.com")["document"]
            assert document.text == "Example test response"
            assert document.metadata["url"] == "https://www.example.com"
            assert "timestamp" in document.metadata

    @pytest.mark.unit
    def test_run_html(self):
        with patch("haystack.preview.components.fetchers.link_content.requests") as mock_run:
            mock_run.get.return_value = Mock(
                status_code=200, text="<h1>Example test response</h1>", headers={"Content-Type": "text/html"}
            )
            fetcher = LinkContentFetcher()
            document = fetcher.run("https://www.example.com")["document"]
            assert document.text == "<h1>Example test response</h1>"
            assert document.metadata["url"] == "https://www.example.com"
            assert "timestamp" in document.metadata

    @pytest.mark.unit
    def test_run_binary(self, test_files_path):
        file_bytes = open(test_files_path / "pdf" / "sample_pdf_1.pdf", "rb").read()
        with patch("haystack.preview.components.fetchers.link_content.requests") as mock_run:
            mock_run.get.return_value = Mock(
                status_code=200, content=file_bytes, headers={"Content-Type": "application/pdf"}
            )
            fetcher = LinkContentFetcher()
            document = fetcher.run("https://www.example.com")["document"]
            # casting to list to make the blobs comparable
            assert list(document.blob) == list(io.BytesIO(file_bytes))
            assert document.metadata["url"] == "https://www.example.com"
            assert "timestamp" in document.metadata

    @pytest.mark.unit
    def test_run_bad_status_code(self):
        fetcher = LinkContentFetcher(raise_on_failure=False)
        mock_response = Mock(status_code=403)
        with patch("haystack.preview.components.fetchers.link_content.requests") as mock_run:
            mock_run.get.return_value = mock_response
            document = fetcher.run("https://www.example.com")["document"]
        assert document is None

    @pytest.mark.integration
    def test_link_content_fetcher_html(self):
        fetcher = LinkContentFetcher()
        document = fetcher.run(HTML_URL)["document"]
        assert document.mime_type == "text/html"
        assert "Introduction to Haystack" in document.text
        assert document.metadata["url"] == HTML_URL

    @pytest.mark.integration
    def test_link_content_fetcher_text(self):
        fetcher = LinkContentFetcher()
        document = fetcher.run(TEXT_URL)["document"]
        assert document.mime_type == "text/plain"
        assert "Haystack" in document.text
        assert document.metadata["url"] == TEXT_URL

    @pytest.mark.integration
    def test_link_content_fetcher_pdf(self):
        fetcher = LinkContentFetcher()
        document = fetcher.run(PDF_URL)["document"]
        assert document.mime_type == "application/octet-stream"  # FIXME Should be "application/pdf"?
        assert document.text is None
        assert document.blob is not None
        assert document.metadata["url"] == PDF_URL
