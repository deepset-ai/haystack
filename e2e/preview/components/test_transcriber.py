from pathlib import Path

from haystack.preview.components import WhisperTranscriber

SAMPLES_PATH = Path(__file__).parent.parent / "test_files"


def test_raw_transcribe():
    comp = WhisperTranscriber()
    output = comp._transcribe(audio_files=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"])
    assert "this is the content of the document" in output[0]["text"].lower()


# Probably unnecessary, a mocked test is enough
def test_transcribe_to_documents():
    comp = WhisperTranscriber()
    output = comp.transcribe_to_documents(
        audio_files=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"]
    )
    assert "this is the content of the document" in output[0].content.lower()
    assert "this is the content of the document.wav" in str(output[0].metadata["audio_file"])
