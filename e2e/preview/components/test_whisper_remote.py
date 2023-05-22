import os

import pytest

from haystack.preview.components.audio.whisper_remote import RemoteWhisperTranscriber


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_whisper_remote_transcriber(preview_samples_path):
    comp = RemoteWhisperTranscriber(api_key=os.environ.get("OPENAI_API_KEY"))

    output = comp.run(
        RemoteWhisperTranscriber.Input(
            audio_files=[
                preview_samples_path / "audio" / "this is the content of the document.wav",
                str((preview_samples_path / "audio" / "the context for this answer is here.wav").absolute()),
                open(preview_samples_path / "audio" / "answer.wav", "rb"),
            ]
        )
    )
    docs = output.documents
    assert len(docs) == 3

    assert "this is the content of the document." == docs[0].content.strip().lower()
    assert preview_samples_path / "audio" / "this is the content of the document.wav" == docs[0].metadata["audio_file"]

    assert "the context for this answer is here." == docs[1].content.strip().lower()
    assert (
        str((preview_samples_path / "audio" / "the context for this answer is here.wav").absolute())
        == docs[1].metadata["audio_file"]
    )

    assert "answer." == docs[2].content.strip().lower()
    assert "<<binary stream>>" == docs[2].metadata["audio_file"]
