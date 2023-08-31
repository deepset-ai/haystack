from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def tenacity_wait():
    """
    Mocks tenacity's wait function to speed up tests.
    """
    with patch("tenacity.nap.time"):
        yield
