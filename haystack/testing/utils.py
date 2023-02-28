import pytest
import requests


def skip_if_down(url):
    """
    Usage:

    @skip_if_down("https://api.openai.com/v1/models")
    def test_foo():
        ...
    """
    try:
        r = requests.options(url, timeout=5)
        r.raise_for_status()
        return pytest.mark.skipif(False)

    except Exception as e:
        return pytest.mark.skipif(True, reason=f"Error accessing '{url}': {e}")
