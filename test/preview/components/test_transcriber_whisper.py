import os

from pathlib import Path

import pytest

from haystack.preview.components import WhisperTranscriber

from test.preview.components.test_component_base import _BaseTestComponent


SAMPLES_PATH = Path(__file__).parent / "test_files"


class TestTranscriber(_BaseTestComponent):
    @pytest.fixture
    def components(self):
        comps = [
            (
                WhisperTranscriber(),
                {
                    "data": [("audio", [SAMPLES_PATH / "audio" / "this is the content of the document.wav"])],
                    "parameters": {},
                },
            )
        ]
        for comp, _ in comps:
            comp.warm_up()
        return comp

    def test_transcribe(self):
        pass
