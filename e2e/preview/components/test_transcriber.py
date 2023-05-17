from pathlib import Path

from haystack.preview.components import LocalWhisperTranscriber


SAMPLES_PATH = Path(__file__).parent.parent / "test_files"


def test_whisperlocaltranscriber():
    comp = LocalWhisperTranscriber(model_name_or_path="tiny")
    docs = comp.transcribe(
        audio_files=[
            SAMPLES_PATH / "audio" / "this is the content of the document.wav",
            str((SAMPLES_PATH / "audio" / "the context for this answer is here.wav").absolute()),
            open(SAMPLES_PATH / "audio" / "answer.wav", "rb"),
        ]
    )
    assert len(docs) == 3

    assert "this is the content of the document." == docs[0].content.strip().lower()
    assert SAMPLES_PATH / "audio" / "this is the content of the document.wav" == docs[0].metadata["audio_file"]

    assert "the context for this answer is here." == docs[1].content.strip().lower()
    assert (
        str((SAMPLES_PATH / "audio" / "the context for this answer is here.wav").absolute())
        == docs[1].metadata["audio_file"]
    )

    assert "answer." == docs[2].content.strip().lower()
    assert "<<binary stream>>" == docs[2].metadata["audio_file"]
