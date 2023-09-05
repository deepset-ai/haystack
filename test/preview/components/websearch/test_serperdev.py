from unittest.mock import patch

import pytest
from requests import Timeout, RequestException, HTTPError

from haystack.preview import Document
from haystack.preview.components.websearch.serper_dev import SerperDevSearchAPI


class TestSerperDev:
    @pytest.mark.unit
    def test_to_dict(self):
        component = SerperDevSearchAPI(api_key="test_key", top_k=10, allowed_domains=["test.com"])
        data = component.to_dict()
        assert data == {
            "type": "SerperDev",
            "init_parameters": {
                "api_key": "test_key",
                "top_k": 10,
                "allowed_domains": ["test.com"],
                "search_params": {},
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "SerperDev",
            "init_parameters": {
                "api_key": "test_key",
                "top_k": 10,
                "allowed_domains": ["test.com"],
                "search_params": {},
            },
        }
        component = SerperDevSearchAPI.from_dict(data)
        assert component.api_key == "test_key"
        assert component.top_k == 10
        assert component.allowed_domains == ["test.com"]
        assert component.search_params == {}

    @pytest.mark.unit
    @pytest.mark.parametrize("top_k", [1, 5, 7])
    def test_web_search_top_k(self, mock_web_search, top_k: int):
        ws = SerperDevSearchAPI(api_key="some_invalid_key", top_k=top_k)
        results = ws.run(query="Who is the boyfriend of Olivia Wilde?")
        assert len(results) == top_k
        assert all(isinstance(doc, Document) for doc in results)

    @pytest.mark.unit
    @patch("requests.post")
    def test_timeout_error(self, mock_post):
        mock_post.side_effect = Timeout
        ws = SerperDevSearchAPI(api_key="some_invalid_key")

        with pytest.raises(TimeoutError):
            ws.run(query="Who is the boyfriend of Olivia Wilde?")

    @pytest.mark.unit
    @patch("requests.post")
    def test_request_exception(self, mock_post):
        mock_post.side_effect = RequestException
        ws = SerperDevSearchAPI(api_key="some_invalid_key")

        with pytest.raises(Exception):
            ws.run(query="Who is the boyfriend of Olivia Wilde?")

    @pytest.mark.unit
    @patch("requests.post")
    def test_bad_response_code(self, mock_post):
        mock_response = mock_post.return_value
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError
        ws = SerperDevSearchAPI(api_key="some_invalid_key")

        with pytest.raises(Exception):
            ws.run(query="Who is the boyfriend of Olivia Wilde?")
