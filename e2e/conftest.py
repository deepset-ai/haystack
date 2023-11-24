from pathlib import Path

import pytest
from haystack.testing.test_utils import set_all_seeds

set_all_seeds(0)


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "samples"
