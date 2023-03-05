import os

import pytest

from haystack.nodes.audio import WhisperTranscriber
from ..conftest import SAMPLES_PATH


@pytest.mark.skipif(os.environ.get("OPENAI_API_KEY", "") == "", reason="OpenAI API key not found")
def test_whisper_transcribe():
    mt = WhisperTranscriber(api_key=os.environ.get("OPENAI_API_KEY", ""))
    with open(SAMPLES_PATH / "audio" / "answer.wav", mode="rb") as audio_file:
        transcript = mt.transcribe(audio=audio_file)
        assert "answer" in transcript["text"]
