# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from haystack.utils.url_validation import _is_private_ip, is_safe_url, is_valid_http_url


def test_url_validation_with_valid_http_url():
    url = "http://example.com"
    assert is_valid_http_url(url)


def test_url_validation_with_valid_https_url():
    url = "https://example.com"
    assert is_valid_http_url(url)


def test_url_validation_with_invalid_scheme():
    url = "ftp://example.com"
    assert not is_valid_http_url(url)


def test_url_validation_with_no_scheme():
    url = "example.com"
    assert not is_valid_http_url(url)


def test_url_validation_with_no_netloc():
    url = "http://"
    assert not is_valid_http_url(url)


def test_url_validation_with_empty_string():
    url = ""
    assert not is_valid_http_url(url)


# Tests for _is_private_ip
class TestIsPrivateIP:
    """Tests for the _is_private_ip helper function."""

    def test_private_ip_10_range(self):
        """Test 10.0.0.0/8 private range."""
        assert _is_private_ip("10.0.0.1")
        assert _is_private_ip("10.255.255.255")

    def test_private_ip_172_range(self):
        """Test 172.16.0.0/12 private range."""
        assert _is_private_ip("172.16.0.1")
        assert _is_private_ip("172.31.255.255")

    def test_private_ip_192_range(self):
        """Test 192.168.0.0/16 private range."""
        assert _is_private_ip("192.168.0.1")
        assert _is_private_ip("192.168.255.255")

    def test_loopback_ipv4(self):
        """Test IPv4 loopback addresses."""
        assert _is_private_ip("127.0.0.1")
        assert _is_private_ip("127.0.0.255")

    def test_loopback_ipv6(self):
        """Test IPv6 loopback address."""
        assert _is_private_ip("::1")

    def test_link_local_ipv4(self):
        """Test IPv4 link-local addresses (cloud metadata endpoint)."""
        assert _is_private_ip("169.254.169.254")
        assert _is_private_ip("169.254.0.1")

    def test_public_ip_not_private(self):
        """Test that public IPs are not flagged as private."""
        assert not _is_private_ip("8.8.8.8")  # Google DNS
        assert not _is_private_ip("1.1.1.1")  # Cloudflare DNS
        assert not _is_private_ip("93.184.216.34")  # example.com

    def test_invalid_ip_returns_false(self):
        """Test that invalid IP strings return False."""
        assert not _is_private_ip("not-an-ip")
        assert not _is_private_ip("example.com")
        assert not _is_private_ip("")


# Tests for is_safe_url
class TestIsSafeUrl:
    """Tests for the is_safe_url function (SSRF protection)."""

    def test_public_url_is_safe(self):
        """Test that public URLs are considered safe."""
        assert is_safe_url("https://example.com")
        assert is_safe_url("https://www.google.com/search?q=test")
        assert is_safe_url("http://github.com/api/v1")

    def test_localhost_is_not_safe(self):
        """Test that localhost is blocked."""
        assert not is_safe_url("http://localhost/admin")
        assert not is_safe_url("http://localhost:8080/api")
        assert not is_safe_url("https://localhost/secret")

    def test_localhost_localdomain_is_not_safe(self):
        """Test that localhost.localdomain is blocked."""
        assert not is_safe_url("http://localhost.localdomain/admin")

    def test_127_0_0_1_is_not_safe(self):
        """Test that 127.0.0.1 is blocked."""
        assert not is_safe_url("http://127.0.0.1/")
        assert not is_safe_url("http://127.0.0.1:8080/api")

    def test_ipv6_loopback_is_not_safe(self):
        """Test that IPv6 loopback is blocked."""
        assert not is_safe_url("http://[::1]/admin")

    def test_0_0_0_0_is_not_safe(self):
        """Test that 0.0.0.0 is blocked."""
        assert not is_safe_url("http://0.0.0.0/")
        assert not is_safe_url("http://0.0.0.0:8080/")

    def test_private_ip_10_is_not_safe(self):
        """Test that 10.x.x.x private IPs are blocked."""
        assert not is_safe_url("http://10.0.0.1/internal")
        assert not is_safe_url("http://10.255.255.255/secret")

    def test_private_ip_172_is_not_safe(self):
        """Test that 172.16-31.x.x private IPs are blocked."""
        assert not is_safe_url("http://172.16.0.1/admin")
        assert not is_safe_url("http://172.31.255.255/")

    def test_private_ip_192_is_not_safe(self):
        """Test that 192.168.x.x private IPs are blocked."""
        assert not is_safe_url("http://192.168.0.1/router")
        assert not is_safe_url("http://192.168.1.1/config")

    def test_cloud_metadata_endpoint_is_not_safe(self):
        """Test that cloud metadata endpoint (169.254.169.254) is blocked."""
        assert not is_safe_url("http://169.254.169.254/")
        assert not is_safe_url("http://169.254.169.254/latest/meta-data/")
        assert not is_safe_url("http://169.254.169.254/latest/meta-data/iam/security-credentials/")

    def test_invalid_url_is_not_safe(self):
        """Test that invalid URLs are not safe."""
        assert not is_safe_url("not-a-url")
        assert not is_safe_url("ftp://example.com")
        assert not is_safe_url("")

    def test_url_with_port_public_is_safe(self):
        """Test that public URLs with ports are safe."""
        assert is_safe_url("https://example.com:443/")
        assert is_safe_url("http://example.com:8080/api")

    def test_dns_resolution_blocks_internal_hostname(self):
        """Test that hostnames resolving to private IPs are blocked."""
        # Mock socket.getaddrinfo to return a private IP
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            # Simulate a hostname that resolves to a private IP
            mock_getaddrinfo.return_value = [
                (2, 1, 6, "", ("10.0.0.1", 80)),  # AF_INET, SOCK_STREAM
            ]
            assert not is_safe_url("http://internal-server.company.com/api")

    def test_dns_resolution_allows_public_hostname(self):
        """Test that hostnames resolving to public IPs are allowed."""
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            # Simulate a hostname that resolves to a public IP
            mock_getaddrinfo.return_value = [
                (2, 1, 6, "", ("93.184.216.34", 80)),  # example.com IP
            ]
            assert is_safe_url("http://example.com/")

    def test_dns_failure_allows_url(self):
        """Test that DNS resolution failure doesn't block the URL."""
        import socket

        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.side_effect = socket.gaierror("Name resolution failed")
            # Should allow the URL even if DNS fails (the fetch will fail anyway)
            assert is_safe_url("http://nonexistent-domain-xyz123.com/")
