from unittest.mock import patch, Mock

from haystack.preview.components.fetchers.link_content import LinkContentFetcher


class TestLinkContentFetcher:
    #  Tests that the fetch method retrieves content from a valid URL with HTTPStatus.OK status code and content.
    def test_fetch_valid_url_with_content(self, mock_get_link_content):
        # Create an instance of LinkContentFetcher
        fetcher = LinkContentFetcher()

        # Call the fetch method with a valid URL
        document = fetcher.run("https://www.example.com")

        # Assert that the document content is equal to the expected content
        assert document.content == "Example test response"

        # Assert that the document metadata contains the URL and timestamp
        assert document.metadata["url"] == "https://www.example.com"
        assert "timestamp" in document.metadata

    #  Tests that the fetch method returns an empty Document when fetching content from a valid URL but reqeust
    #  is blocked.
    def test_fetch_valid_url_blocked_status_code(self):
        # Create an instance of LinkContentFetcher
        fetcher = LinkContentFetcher(raise_on_failure=False)
        mock_response = Mock(status_code=403, text=None, headers={"Content-Type": "text/html"})
        # Call the fetch method with a valid URL
        with patch("haystack.preview.components.fetchers.link_content.requests") as mock_run:
            mock_run.get.return_value = mock_response
            document = fetcher.run("https://www.example.com")

        # Assert that the document content is an empty string
        assert document.content == ""

    #  Tests that the fetch method returns an empty Document when fetching content from a valid URL with
    #  HTTPStatus.INTERNAL_SERVER_ERROR status code.
    def test_fetch_valid_url_with_internal_server_error_status_code(self):
        # Create an instance of LinkContentFetcher
        fetcher = LinkContentFetcher(raise_on_failure=False)
        mock_response = Mock(status_code=500, text="Example test response", headers={"Content-Type": "text/html"})
        # Call the fetch method with a valid URL
        with patch("haystack.preview.components.fetchers.link_content.requests") as mock_run:
            mock_run.get.return_value = mock_response
            document = fetcher.run("https://www.example.com")

        # Assert that the document content is an empty string
        assert document.content == ""
