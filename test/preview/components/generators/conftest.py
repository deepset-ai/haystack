from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def mock_auto_tokenizer():
    """
    In the original mock_auto_tokenizer fixture, we were mocking the transformers.AutoTokenizer.from_pretrained
    method directly, but we were not providing a return value for this method. Therefore, when from_pretrained
    was called within HuggingFaceTGIChatGenerator, it returned None because that's the default behavior of a
    MagicMock object when a return value isn't specified.

    We will update the mock_auto_tokenizer fixture to return a MagicMock object when from_pretrained is called
    in another PR. For now, we will use this fixture to mock the AutoTokenizer.from_pretrained method.
    """

    with patch("transformers.AutoTokenizer.from_pretrained", autospec=True) as mock_from_pretrained:
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer
        yield mock_tokenizer
