from typing import List, Tuple, Dict, Any

from pathlib import Path

import pytest

from haystack.preview.components import WhisperTranscriber

from test.preview.components.test_component_base import BaseTestComponent


SAMPLES_PATH = Path(__file__).parent.parent / "test_files"


class TestTranscriber(BaseTestComponent):
    """
    Tests for WhisperTranscriber.
    """

    @pytest.fixture
    def components(self):
        return [WhisperTranscriber()]

    def test_transcribe(self):
        pass
        # TODO mock model
