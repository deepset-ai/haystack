from pathlib import Path
from unittest.mock import Mock
import pytest

from haystack.preview.testing.test_utils import set_all_seeds

set_all_seeds(0)


@pytest.fixture()
def mock_tokenizer():
    """
    Tokenizes the string by splitting on spaces.
    """
    tokenizer = Mock()
    tokenizer.encode = lambda text: text.split()
    tokenizer.decode = lambda tokens: " ".join(tokens)
    return tokenizer


@pytest.fixture()
def test_files_path():
    return Path(__file__).parent / "test_files"
