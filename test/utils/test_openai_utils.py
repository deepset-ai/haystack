from unittest.mock import patch

import pytest
from tenacity import wait_none

from haystack.errors import OpenAIError, OpenAIRateLimitError, OpenAIUnauthorizedError
from haystack.utils.openai_utils import openai_request


@pytest.mark.unit
@patch("haystack.utils.openai_utils.requests")
def test_openai_request_retries_generic_error(mock_requests):
    mock_requests.request.return_value.status_code = 418

    with pytest.raises(OpenAIError):
        # We need to use a custom wait amount otherwise the test would take forever to run
        # as the original wait time is exponential
        openai_request.retry_with(wait=wait_none())(url="some_url", headers={}, payload={}, read_response=False)

    assert mock_requests.request.call_count == 5


@pytest.mark.unit
@patch("haystack.utils.openai_utils.requests")
def test_openai_request_retries_on_rate_limit_error(mock_requests):
    mock_requests.request.return_value.status_code = 429

    with pytest.raises(OpenAIRateLimitError):
        # We need to use a custom wait amount otherwise the test would take forever to run
        # as the original wait time is exponential
        openai_request.retry_with(wait=wait_none())(url="some_url", headers={}, payload={}, read_response=False)

    assert mock_requests.request.call_count == 5


@pytest.mark.unit
@patch("haystack.utils.openai_utils.requests")
def test_openai_request_does_not_retry_on_unauthorized_error(mock_requests):
    mock_requests.request.return_value.status_code = 401

    with pytest.raises(OpenAIUnauthorizedError):
        # We need to use a custom wait amount otherwise the test would take forever to run
        # as the original wait time is exponential
        openai_request.retry_with(wait=wait_none())(url="some_url", headers={}, payload={}, read_response=False)

    assert mock_requests.request.call_count == 1


@pytest.mark.unit
@patch("haystack.utils.openai_utils.requests")
def test_openai_request_does_not_retry_on_success(mock_requests):
    mock_requests.request.return_value.status_code = 200
    # We need to use a custom wait amount otherwise the test would take forever to run
    # as the original wait time is exponential
    openai_request.retry_with(wait=wait_none())(url="some_url", headers={}, payload={}, read_response=False)

    assert mock_requests.request.call_count == 1
