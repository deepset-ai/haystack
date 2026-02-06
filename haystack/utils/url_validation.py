# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import ipaddress
import socket
from urllib.parse import urlparse


def is_valid_http_url(url: str) -> bool:
    """Check if a URL is a valid HTTP/HTTPS URL."""
    r = urlparse(url)
    return all([r.scheme in ["http", "https"], r.netloc])


def _is_private_ip(ip_str: str) -> bool:
    """
    Check if an IP address is private, loopback, or otherwise internal.

    This includes:
    - Private ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
    - Loopback (127.0.0.0/8, ::1)
    - Link-local (169.254.0.0/16, fe80::/10) - includes cloud metadata endpoints
    - Reserved and unspecified addresses

    :param ip_str: IP address as a string.
    :returns: True if the IP is private/internal, False otherwise.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_unspecified
            or ip.is_multicast
        )
    except ValueError:
        # If it's not a valid IP, return False (hostname will be resolved separately)
        return False


def is_safe_url(url: str) -> bool:
    """
    Check if a URL is safe to fetch (not pointing to internal/private resources).

    This function provides SSRF (Server-Side Request Forgery) protection by blocking:
    - Private IP ranges (10.x.x.x, 172.16-31.x.x, 192.168.x.x)
    - Loopback addresses (127.0.0.1, localhost, ::1)
    - Link-local addresses (169.254.x.x) - commonly used for cloud metadata
    - Other reserved/internal IP ranges

    :param url: The URL to validate.
    :returns: True if the URL is safe to fetch, False if it points to internal resources.
    :raises ValueError: If the URL is malformed.
    """
    if not is_valid_http_url(url):
        return False

    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        return False

    # Check for common localhost aliases
    localhost_aliases = {"localhost", "localhost.localdomain", "127.0.0.1", "::1", "0.0.0.0"}
    if hostname.lower() in localhost_aliases:
        return False

    # Check if hostname is an IP address
    if _is_private_ip(hostname):
        return False

    # Resolve hostname to IP and check if it resolves to a private IP
    # This prevents DNS rebinding attacks using hostnames that resolve to internal IPs
    try:
        resolved_ips = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _, _, _, sockaddr in resolved_ips:
            ip_str = sockaddr[0]
            if _is_private_ip(ip_str):
                return False
    except socket.gaierror:
        # If DNS resolution fails, we allow it (the fetch will fail anyway)
        # This is more permissive but avoids blocking valid URLs during DNS issues
        pass

    return True
