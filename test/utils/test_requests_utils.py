# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import httpx
import requests
from unittest.mock import patch, MagicMock

from haystack.utils.requests_utils import request_with_retry, async_request_with_retry


@pytest.fixture
def mock_requests_response():
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def mock_httpx_response():
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.raise_for_status.return_value = None
    return response


class TestRequestWithRetry:
    def test_request_with_retry_success(self, mock_requests_response):
        """Test that request_with_retry works with default parameters"""
        with patch("requests.request", return_value=mock_requests_response) as mock_request:
            response = request_with_retry(method="GET", url="https://example.com")

            assert response == mock_requests_response
            mock_request.assert_called_once_with(method="GET", url="https://example.com", timeout=10)

    def test_request_with_retry_custom_attempts(self, mock_requests_response):
        """Test that request_with_retry respects custom attempts parameter"""
        with patch("requests.request", return_value=mock_requests_response) as mock_request:
            response = request_with_retry(method="GET", url="https://example.com", attempts=5)

            assert response == mock_requests_response
            mock_request.assert_called_once_with(method="GET", url="https://example.com", timeout=10)

    def test_request_with_retry_custom_status_codes(self, mock_requests_response):
        """Test that request_with_retry respects custom status_codes_to_retry parameter"""
        with patch("requests.request", return_value=mock_requests_response) as mock_request:
            response = request_with_retry(method="GET", url="https://example.com", status_codes_to_retry=[500, 502])

            assert response == mock_requests_response
            mock_request.assert_called_once_with(method="GET", url="https://example.com", timeout=10)

    def test_request_with_retry_custom_timeout(self, mock_requests_response):
        """Test that request_with_retry respects custom timeout parameter"""
        with patch("requests.request", return_value=mock_requests_response) as mock_request:
            response = request_with_retry(method="GET", url="https://example.com", timeout=30)

            assert response == mock_requests_response
            mock_request.assert_called_once_with(method="GET", url="https://example.com", timeout=30)

    def test_request_with_retry_with_headers(self, mock_requests_response):
        """Test that request_with_retry passes headers correctly"""
        headers = {"Authorization": "Bearer token123"}
        with patch("requests.request", return_value=mock_requests_response) as mock_request:
            response = request_with_retry(method="GET", url="https://example.com", headers=headers)

            assert response == mock_requests_response
            mock_request.assert_called_once_with(method="GET", url="https://example.com", headers=headers, timeout=10)

    def test_request_with_retry_with_json(self, mock_requests_response):
        """Test that request_with_retry passes JSON data correctly"""
        json_data = {"key": "value"}
        with patch("requests.request", return_value=mock_requests_response) as mock_request:
            response = request_with_retry(method="POST", url="https://example.com", json=json_data)

            assert response == mock_requests_response
            mock_request.assert_called_once_with(method="POST", url="https://example.com", json=json_data, timeout=10)

    def test_request_with_retry_retries_on_error(self):
        """Test that request_with_retry retries on HTTP errors"""
        with patch("time.sleep") as mock_sleep:
            # Mock time.sleep used by tenacity to keep this test fast
            mock_sleep.return_value = None

            error_response = requests.Response()
            error_response.status_code = 503

            success_response = requests.Response()
            success_response.status_code = 200

            with patch("requests.request") as mock_request:
                # First call raises an error, second call succeeds
                mock_request.side_effect = [requests.exceptions.HTTPError("Server error"), success_response]

                response = request_with_retry(method="GET", url="https://example.com", attempts=2)

                assert response == success_response
                assert mock_request.call_count == 2
                mock_sleep.assert_called()

    def test_request_with_retry_retries_on_status_code(self):
        """Test that request_with_retry retries on specified status codes"""
        with patch("time.sleep") as mock_sleep:
            # Mock time.sleep used by tenacity to keep this test fast
            mock_sleep.return_value = None

            error_response = requests.Response()
            error_response.status_code = 503

            def raise_for_status():
                if error_response.status_code in [503]:
                    raise requests.exceptions.HTTPError("Service Unavailable")

            error_response.raise_for_status = raise_for_status

            success_response = requests.Response()
            success_response.status_code = 200
            success_response.raise_for_status = lambda: None

            with patch("requests.request") as mock_request:
                # First call returns error status code, second call succeeds
                mock_request.side_effect = [error_response, success_response]

                response = request_with_retry(
                    method="GET", url="https://example.com", attempts=2, status_codes_to_retry=[503]
                )

                assert response == success_response
                assert mock_request.call_count == 2
                mock_sleep.assert_called()


class TestAsyncRequestWithRetry:
    @pytest.mark.asyncio
    async def test_async_request_with_retry_success(self, mock_httpx_response):
        """Test that async_request_with_retry works with default parameters"""
        with patch("httpx.AsyncClient.request", return_value=mock_httpx_response) as mock_request:
            response = await async_request_with_retry(method="GET", url="https://example.com")

            assert response == mock_httpx_response
            mock_request.assert_called_once_with(method="GET", url="https://example.com", timeout=10)

    @pytest.mark.asyncio
    async def test_async_request_with_retry_custom_attempts(self, mock_httpx_response):
        """Test that async_request_with_retry respects custom attempts parameter"""
        with patch("httpx.AsyncClient.request", return_value=mock_httpx_response) as mock_request:
            response = await async_request_with_retry(method="GET", url="https://example.com", attempts=5)

            assert response == mock_httpx_response
            mock_request.assert_called_once_with(method="GET", url="https://example.com", timeout=10)

    @pytest.mark.asyncio
    async def test_async_request_with_retry_custom_status_codes(self, mock_httpx_response):
        """Test that async_request_with_retry respects custom status_codes_to_retry parameter"""
        with patch("httpx.AsyncClient.request", return_value=mock_httpx_response) as mock_request:
            response = await async_request_with_retry(
                method="GET", url="https://example.com", status_codes_to_retry=[500, 502]
            )

            assert response == mock_httpx_response
            mock_request.assert_called_once_with(method="GET", url="https://example.com", timeout=10)

    @pytest.mark.asyncio
    async def test_async_request_with_retry_custom_timeout(self, mock_httpx_response):
        """Test that async_request_with_retry respects custom timeout parameter"""
        with patch("httpx.AsyncClient.request", return_value=mock_httpx_response) as mock_request:
            response = await async_request_with_retry(method="GET", url="https://example.com", timeout=30)

            assert response == mock_httpx_response
            mock_request.assert_called_once_with(method="GET", url="https://example.com", timeout=30)

    @pytest.mark.asyncio
    async def test_async_request_with_retry_with_headers(self, mock_httpx_response):
        """Test that async_request_with_retry passes headers correctly"""
        headers = {"Authorization": "Bearer token123"}
        with patch("httpx.AsyncClient.request", return_value=mock_httpx_response) as mock_request:
            response = await async_request_with_retry(method="GET", url="https://example.com", headers=headers)

            assert response == mock_httpx_response
            mock_request.assert_called_once_with(method="GET", url="https://example.com", headers=headers, timeout=10)

    @pytest.mark.asyncio
    async def test_async_request_with_retry_with_json(self, mock_httpx_response):
        """Test that async_request_with_retry passes JSON data correctly"""
        json_data = {"key": "value"}
        with patch("httpx.AsyncClient.request", return_value=mock_httpx_response) as mock_request:
            response = await async_request_with_retry(method="POST", url="https://example.com", json=json_data)

            assert response == mock_httpx_response
            mock_request.assert_called_once_with(method="POST", url="https://example.com", json=json_data, timeout=10)

    @pytest.mark.asyncio
    async def test_async_request_with_retry_retries_on_error(self):
        """Test that async_request_with_retry retries on HTTP errors"""
        with patch("asyncio.sleep") as mock_sleep:
            # Mock asyncio.sleep used by tenacity to keep this test fast
            mock_sleep.return_value = None

            error_response = httpx.Response(status_code=503, request=httpx.Request("GET", "https://example.com"))
            success_response = httpx.Response(status_code=200, request=httpx.Request("GET", "https://example.com"))

            with patch("httpx.AsyncClient.request") as mock_request:
                # First call raises an error, second call succeeds
                mock_request.side_effect = [
                    httpx.RequestError("Server error", request=httpx.Request("GET", "https://example.com")),
                    success_response,
                ]

                response = await async_request_with_retry(method="GET", url="https://example.com", attempts=2)

                assert response == success_response
                assert mock_request.call_count == 2
                mock_sleep.assert_called()

    @pytest.mark.asyncio
    async def test_async_request_with_retry_retries_on_status_code(self):
        """Test that async_request_with_retry retries on specified status codes"""
        with patch("asyncio.sleep") as mock_sleep:
            # Mock asyncio.sleep used by tenacity to keep this test fast
            mock_sleep.return_value = None

            error_response = httpx.Response(status_code=503, request=httpx.Request("GET", "https://example.com"))

            def raise_for_status():
                if error_response.status_code in [503]:
                    raise httpx.HTTPStatusError(
                        "Service Unavailable", request=error_response.request, response=error_response
                    )

            error_response.raise_for_status = raise_for_status

            success_response = httpx.Response(status_code=200, request=httpx.Request("GET", "https://example.com"))
            success_response.raise_for_status = lambda: None

            with patch("httpx.AsyncClient.request") as mock_request:
                # First call returns error status code, second call succeeds
                mock_request.side_effect = [error_response, success_response]

                response = await async_request_with_retry(
                    method="GET", url="https://example.com", attempts=2, status_codes_to_retry=[503]
                )

                assert response == success_response
                assert mock_request.call_count == 2
                mock_sleep.assert_called()
