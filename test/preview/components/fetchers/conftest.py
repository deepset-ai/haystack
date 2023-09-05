from unittest.mock import patch, Mock

import pytest


@pytest.fixture
def mock_get_link_content():
    with patch("haystack.preview.components.fetchers.link_content.requests") as mock_run:
        mock_run.get.return_value = Mock(
            status_code=200, text="Example test response", headers={"Content-Type": "text/html"}
        )
        yield mock_run
