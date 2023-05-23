from pathlib import Path

import pytest

from unittest.mock import patch, MagicMock


TEST_FILES = Path(__file__).parent / "test_files"


@pytest.fixture(autouse=True)
def mock_mermaid_request():
    """
    Prevents real requests to https://mermaid.ink/
    """
    with patch("canals.draw.mermaid.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = open(TEST_FILES / "mermaid_mock" / "test_response.png", "rb").read()
        mock_get.return_value = mock_response
        yield
