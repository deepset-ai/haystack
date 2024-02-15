from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def test_files():
    return Path(__file__).parent / "test_files"
