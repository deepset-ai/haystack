# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import urlparse


def is_valid_http_url(url: str) -> bool:
    """Check if a URL is a valid HTTP/HTTPS URL."""
    r = urlparse(url)
    return all([r.scheme in ["http", "https"], r.netloc])
