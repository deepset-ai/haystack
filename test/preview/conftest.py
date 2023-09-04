from unittest.mock import Mock, patch
import pytest


@pytest.fixture()
def mock_tokenizer():
    """
    Tokenizes the string by splitting on spaces.
    """
    tokenizer = Mock()
    tokenizer.encode = lambda text: text.split()
    tokenizer.decode = lambda tokens: " ".join(tokens)
    return tokenizer


@pytest.fixture(autouse=True)
def tenacity_wait():
    """
    Mocks tenacity's wait function to speed up tests.
    """
    with patch("tenacity.nap.time"):
        yield
