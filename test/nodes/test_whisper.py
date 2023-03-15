import os

import pytest

from haystack.nodes.audio import WhisperTranscriber
from haystack.utils.import_utils import is_whisper_available
from ..conftest import SAMPLES_PATH


@pytest.mark.skipif(os.environ.get("OPENAI_API_KEY", "") == "", reason="OpenAI API key not found")
@pytest.mark.integration
def test_whisper_api_transcribe():
    w = WhisperTranscriber(api_key=os.environ.get("OPENAI_API_KEY"))
    audio_object_transcript, audio_path_transcript = transcribe_test_helper(w)
    assert "segments" not in audio_object_transcript and "segments" not in audio_path_transcript


@pytest.mark.skipif(os.environ.get("OPENAI_API_KEY", "") == "", reason="OpenAI API key not found")
@pytest.mark.integration
def test_whisper_api_transcribe_with_params():
    w = WhisperTranscriber(api_key=os.environ.get("OPENAI_API_KEY"))
    audio_object_transcript, audio_path_transcript = transcribe_test_helper(w)
    assert "segments" not in audio_object_transcript and "segments" not in audio_path_transcript


@pytest.mark.skip("Fails on CI cause it fills up memory")
@pytest.mark.integration
@pytest.mark.skipif(not is_whisper_available(), reason="Whisper is not installed")
def test_whisper_local_transcribe():
    w = WhisperTranscriber()
    audio_object_transcript, audio_path_transcript = transcribe_test_helper(w, language="en")
    assert "segments" not in audio_object_transcript and "segments" not in audio_path_transcript


@pytest.mark.skip("Fails on CI cause it fills up memory")
@pytest.mark.integration
@pytest.mark.skipif(not is_whisper_available(), reason="Whisper is not installed")
def test_whisper_local_transcribe_with_params():
    w = WhisperTranscriber()
    audio_object, audio_path = transcribe_test_helper(w, language="en", return_segments=True)
    assert len(audio_object["segments"]) == 1 and len(audio_path["segments"]) == 1


def transcribe_test_helper(whisper, **kwargs):
    # this file is 1 second long and contains the word "answer"
    file_path = str(SAMPLES_PATH / "audio" / "answer.wav")

    # using audio object
    with open(file_path, mode="rb") as audio_file:
        audio_object_transcript = whisper.transcribe(audio_file=audio_file, **kwargs)
        assert "answer" in audio_object_transcript["text"].lower()

    # using path to audio file
    audio_path_transcript = whisper.transcribe(audio_file=file_path, **kwargs)
    assert "answer" in audio_path_transcript["text"].lower()
    return audio_object_transcript, audio_path_transcript
