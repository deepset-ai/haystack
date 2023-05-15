from pathlib import Path

from haystack.preview.components import LocalWhisperTranscriber


SAMPLES_PATH = Path(__file__).parent.parent / "test_files"


def test_whisperlocaltranscriber():
    comp = LocalWhisperTranscriber(model_name_or_path="tiny")
    docs = comp.transcribe(
        audio_files=[
            SAMPLES_PATH / "audio" / "this is the content of the document.wav",
            SAMPLES_PATH / "audio" / "the context for this answer is here.wav",
        ]
    )
    assert len(docs) == 2
    assert "this is the content of the document." == docs[0].content.strip().lower()
    assert SAMPLES_PATH / "audio" / "this is the content of the document.wav" == docs[0].metadata["audio_file"]
    assert "the context for this answer is here." == docs[1].content.strip().lower()
    assert SAMPLES_PATH / "audio" / "the context for this answer is here.wav" == docs[1].metadata["audio_file"]
