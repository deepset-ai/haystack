from unittest.mock import patch

from ui.utils import haystack_is_ready


def test_haystack_is_ready():
    with patch("requests.get") as mocked_get:
        mocked_get.return_value.status_code = 200
        assert haystack_is_ready()


def test_haystack_is_ready_fail():
    with patch("requests.get") as mocked_get:
        mocked_get.return_value.status_code = 400
        assert not haystack_is_ready()
