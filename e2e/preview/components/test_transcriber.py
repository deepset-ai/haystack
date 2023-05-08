import os
from pathlib import Path

from haystack.preview.components import WhisperTranscriber


SAMPLES_PATH = Path(__file__).parent.parent / "test_files"


def test_raw_transcribe_local():
    comp = WhisperTranscriber(model_name_or_path="tiny")
    output = comp.transcribe(audio_files=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"])
    assert "this is the content of the document" in output[0]["text"].lower()


def test_raw_transcribe_remote():
    comp = WhisperTranscriber(api_key=os.environ.get("OPENAI_API_KEY"))
    output = comp.transcribe(audio_files=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"])
    assert "this is the content of the document" in output[0]["text"].lower()
