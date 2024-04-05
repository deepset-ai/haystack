from urllib.parse import urlparse


def is_valid_http_url(url) -> bool:
    r = urlparse(url)
    return all([r.scheme in ["http", "https"], r.netloc])
