from haystack.preview.components.audio.whisper_local import LocalWhisperTranscriber


def test_whisper_local_transcriber(preview_samples_path):
    comp = LocalWhisperTranscriber(model_name_or_path="tiny")
    docs = comp.transcribe(
        audio_files=[
            preview_samples_path / "audio" / "this is the content of the document.wav",
            str((preview_samples_path / "audio" / "the context for this answer is here.wav").absolute()),
            open(preview_samples_path / "audio" / "answer.wav", "rb"),
        ]
    )
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
