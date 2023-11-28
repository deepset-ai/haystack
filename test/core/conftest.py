from pathlib import Path

import pytest

from unittest.mock import patch, MagicMock


@pytest.fixture
def test_files():
    return Path(__file__).parent / "test_files"


@pytest.fixture(autouse=True)
def mock_mermaid_request(test_files):
    """
    Prevents real requests to https://mermaid.ink/
    """
    with patch("haystack.core.pipeline.draw.mermaid.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = open(test_files / "mermaid_mock" / "test_response.png", "rb").read()
        mock_get.return_value = mock_response
        yield
