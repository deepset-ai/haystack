from unittest.mock import patch, Mock

import pytest
import requests

from haystack.utils.requests_utils import request_with_retry


@pytest.mark.unit
@patch("haystack.utils.requests_utils.requests.request")
def test_request_with_retry_defaults_successfully(mock_request):
    # Make requests with default retry configuration
    request_with_retry(method="GET", url="https://example.com")

    # Verifies request has not been retried
    mock_request.assert_called_once_with(method="GET", url="https://example.com", timeout=10)


@pytest.mark.unit
@patch("haystack.utils.requests_utils.requests.request")
def test_request_with_retry_custom_timeout(mock_request):
    # Make requests with default retry configuration
    request_with_retry(method="GET", url="https://example.com", timeout=5)

    # Verifies request has not been retried
    mock_request.assert_called_once_with(method="GET", url="https://example.com", timeout=5)


@pytest.mark.unit
@patch("haystack.utils.requests_utils.requests.request")
def test_request_with_retry_failing_request_and_expected_status_code(mock_request):
    # Create fake failed response with status code that triggers retry
    fake_response = requests.Response()
    fake_response.status_code = 408
    mock_request.return_value = fake_response

    # Make request with expected status code and verify error is raised
    with pytest.raises(requests.HTTPError):
        request_with_retry(method="GET", url="https://example.com", timeout=1, attempts=2, status_codes_to_retry=[408])

    # Veries request has been retried the expected number of times
    assert mock_request.call_count == 2


@pytest.mark.unit
@patch("haystack.utils.requests_utils.requests.request")
def test_request_with_retry_failing_request_and_ignored_status_code(mock_request):
    # Create fake failed response with status code that doesn't trigger retry
    fake_response = requests.Response()
    fake_response.status_code = 500
    mock_request.return_value = fake_response

    # Make request with status code that won't trigger a retry and verify error is raised
    with pytest.raises(requests.HTTPError):
        request_with_retry(method="GET", url="https://example.com", timeout=1, status_codes_to_retry=[404])

    # Verify request has not been retried
    mock_request.assert_called_once()


@pytest.mark.unit
@patch("haystack.utils.requests_utils.requests.request")
def test_request_with_retry_timed_out_request(mock_request: Mock):
    # Make request fail cause of a timeout
    mock_request.side_effect = TimeoutError()

    # Make request and verifies it fails
    with pytest.raises(TimeoutError):
        request_with_retry(method="GET", url="https://example.com", timeout=1, attempts=2)

    # Verifies request has been retried the expected number of times
    assert mock_request.call_count == 2
