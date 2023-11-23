from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture
def mock_openai_tokenizer():
    with patch("haystack.nodes.prompt.invocation_layer.open_ai.load_openai_tokenizer") as mock_tokenizer_func:
        mock_tokenizer = MagicMock()  # this will be our mock tokenizer
        # "This is a test for a mock openai tokenizer."
        mock_tokenizer.encode.return_value = [2028, 374, 264, 1296, 369, 264, 8018, 1825, 2192, 47058, 13]
        # Returning truncated prompt: [2028, 374, 264, 1296, 369, 264, 8018, 1825, 2192]
        mock_tokenizer.decode.return_value = "This is a test for a mock openai"
        mock_tokenizer_func.return_value = mock_tokenizer
        yield mock_tokenizer_func
