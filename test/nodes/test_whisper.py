import os

import pytest

from haystack.nodes.audio import WhisperTranscriber
from haystack.utils.import_utils import is_whisper_available
from ..conftest import SAMPLES_PATH


@pytest.mark.skipif(os.environ.get("OPENAI_API_KEY", "") == "", reason="OpenAI API key not found")
@pytest.mark.integration
def test_whisper_api_transcribe():
    w = WhisperTranscriber(api_key=os.environ.get("OPENAI_API_KEY"))
    transcribe_test_helper(w)


@pytest.mark.integration
@pytest.mark.skipif(not is_whisper_available(), reason="Whisper is not installed")
def test_whisper_local_transcribe():
    w = WhisperTranscriber()
    transcribe_test_helper(w)


def transcribe_test_helper(w):
    # using resolved audio object
    with open(SAMPLES_PATH / "audio" / "answer.wav", mode="rb") as audio_file:
        transcript = w.transcribe(audio_file=audio_file)
        assert "answer" in transcript["text"].lower()

    # using path to audio file
    transcript = w.transcribe(audio_file=str(SAMPLES_PATH / "audio" / "answer.wav"))
    assert "answer" in transcript["text"].lower()
