from unittest import mock

import pytest
from requests import Response, HTTPError

from haystack.testing.utils import skip_if_down


@pytest.mark.unit
def test_skip_is_down_dont_skip():
    with mock.patch("haystack.testing.utils.requests") as r:
        r.options.return_value.status_code = 200

        mark_fn = skip_if_down("https://foo")
        assert mark_fn.args[0] == False
        assert mark_fn.kwargs.get("reason") is None


@pytest.mark.unit
def test_skip_is_down_unauthorized():
    with mock.patch("haystack.testing.utils.requests") as r:
        r.options.return_value.raise_for_status.side_effect = HTTPError("Something went wrong")

        mark_fn = skip_if_down("https://foo")
        assert mark_fn.args[0] == True
        assert mark_fn.kwargs.get("reason") == "Error accessing 'https://foo': Something went wrong"
