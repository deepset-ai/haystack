from unittest.mock import MagicMock

import pytest
from canals.testing import BaseTestComponent as CanalsBaseTestComponent


class BaseTestComponent(CanalsBaseTestComponent):
    """
    Base tests for Haystack components.
    """

    @pytest.fixture
    def request_mock(self, monkeypatch):
        request_mock = MagicMock()
        monkeypatch.setattr("requests.request", MagicMock())
        return request_mock
