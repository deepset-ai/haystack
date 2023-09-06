import io
from unittest.mock import patch, Mock

import pytest

from haystack.preview.components.fetchers.link_content import (
    LinkContentFetcher,
    text_content_handler,
    binary_content_handler,
    DEFAULT_USER_AGENT,
)


@pytest.fixture
def mock_get_link_text_content():
    with patch("haystack.preview.components.fetchers.link_content.requests") as mock_run:
        mock_run.get.return_value = Mock(
            status_code=200, text="Example test response", headers={"Content-Type": "text/plain"}
        )
        yield mock_run


@pytest.fixture
def mock_get_link_binary_content(test_files_path):
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
    def test_run_text(self, mock_get_link_text_content):
        fetcher = LinkContentFetcher()
        document = fetcher.run("https://www.example.com")["document"]
        assert document.content == "Example test response"
        assert document.metadata["url"] == "https://www.example.com"
        assert "timestamp" in document.metadata

    @pytest.mark.unit
    def test_run_binary(self, mock_get_link_binary_content, test_files_path):
        fetcher = LinkContentFetcher()
        document = fetcher.run("https://www.example.com")["document"]
        assert document.content == io.BytesIO(open(test_files_path / "pdf" / "sample_pdf_1.pdf", "rb").read())
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
