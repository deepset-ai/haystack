import os

import pytest

from haystack.preview.components import WhisperTranscriber

from test.preview.components.test_component_base import _BaseTestComponent


class TestTranscriber(_BaseTestComponent):
    @pytest.fixture
    def components(self):
        return [WhisperTranscriber()]

    def test_transcribe(self):
        pass
