import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from haystack.preview.dataclasses import Document
from haystack.preview.components.audio import LocalWhisperTranscriber


SAMPLES_PATH = Path(__file__).parent.parent.parent / "test_files"


class TestLocalWhisperTranscriber:
    @pytest.mark.unit
    def test_init(self):
        transcriber = LocalWhisperTranscriber(
            model_name_or_path="large-v2"
        )  # Doesn't matter if it's huge, the model is not loaded in init.
        assert transcriber.model_name == "large-v2"
        assert transcriber.device == torch.device("cpu")
        assert transcriber._model is None

    @pytest.mark.unit
    def test_init_wrong_model(self):
        with pytest.raises(ValueError, match="Model name 'whisper-1' not recognized"):
            LocalWhisperTranscriber(model_name_or_path="whisper-1")

    @pytest.mark.unit
    def test_to_dict(self):
        transcriber = LocalWhisperTranscriber()
        data = transcriber.to_dict()
        assert data == {
            "type": "haystack.preview.components.audio.whisper_local.LocalWhisperTranscriber",
            "init_parameters": {"model_name_or_path": "large", "device": "cpu", "whisper_params": {}},
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        transcriber = LocalWhisperTranscriber(
            model_name_or_path="tiny",
            device="cuda",
            whisper_params={"return_segments": True, "temperature": [0.1, 0.6, 0.8]},
        )
        data = transcriber.to_dict()
        assert data == {
            "type": "haystack.preview.components.audio.whisper_local.LocalWhisperTranscriber",
            "init_parameters": {
                "model_name_or_path": "tiny",
                "device": "cuda",
                "whisper_params": {"return_segments": True, "temperature": [0.1, 0.6, 0.8]},
            },
        }

    @pytest.mark.unit
    def test_warmup(self):
        with patch("haystack.preview.components.audio.whisper_local.whisper") as mocked_whisper:
            transcriber = LocalWhisperTranscriber(model_name_or_path="large-v2")
            mocked_whisper.load_model.assert_not_called()
            transcriber.warm_up()
            mocked_whisper.load_model.assert_called_once_with("large-v2", device=torch.device(type="cpu"))

    @pytest.mark.unit
    def test_warmup_doesnt_reload(self):
        with patch("haystack.preview.components.audio.whisper_local.whisper") as mocked_whisper:
            transcriber = LocalWhisperTranscriber(model_name_or_path="large-v2")
            transcriber.warm_up()
            transcriber.warm_up()
            mocked_whisper.load_model.assert_called_once()

    @pytest.mark.unit
    def test_run_with_path(self):
        comp = LocalWhisperTranscriber(model_name_or_path="large-v2")
        comp._model = MagicMock()
        comp._model.transcribe.return_value = {
            "text": "test transcription",
            "other_metadata": ["other", "meta", "data"],
        }
        results = comp.run(audio_files=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"])
        expected = Document(
            content="test transcription",
            meta={
                "audio_file": SAMPLES_PATH / "audio" / "this is the content of the document.wav",
                "other_metadata": ["other", "meta", "data"],
            },
        )
        assert results["documents"] == [expected]

    @pytest.mark.unit
    def test_run_with_str(self):
        comp = LocalWhisperTranscriber(model_name_or_path="large-v2")
        comp._model = MagicMock()
        comp._model.transcribe.return_value = {
            "text": "test transcription",
            "other_metadata": ["other", "meta", "data"],
        }
        results = comp.run(
            audio_files=[str((SAMPLES_PATH / "audio" / "this is the content of the document.wav").absolute())]
        )
        expected = Document(
            content="test transcription",
            meta={
                "audio_file": str((SAMPLES_PATH / "audio" / "this is the content of the document.wav").absolute()),
                "other_metadata": ["other", "meta", "data"],
            },
        )
        assert results["documents"] == [expected]

    @pytest.mark.unit
    def test_transcribe(self):
        comp = LocalWhisperTranscriber(model_name_or_path="large-v2")
        comp._model = MagicMock()
        comp._model.transcribe.return_value = {
            "text": "test transcription",
            "other_metadata": ["other", "meta", "data"],
        }
        results = comp.transcribe(audio_files=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"])
        expected = Document(
            content="test transcription",
            meta={
                "audio_file": SAMPLES_PATH / "audio" / "this is the content of the document.wav",
                "other_metadata": ["other", "meta", "data"],
            },
        )
        assert results == [expected]

    @pytest.mark.unit
    def test_transcribe_stream(self):
        comp = LocalWhisperTranscriber(model_name_or_path="large-v2")
        comp._model = MagicMock()
        comp._model.transcribe.return_value = {
            "text": "test transcription",
            "other_metadata": ["other", "meta", "data"],
        }
        results = comp.transcribe(
            audio_files=[open(SAMPLES_PATH / "audio" / "this is the content of the document.wav", "rb")]
        )
        expected = Document(
            content="test transcription",
            meta={"audio_file": "<<binary stream>>", "other_metadata": ["other", "meta", "data"]},
        )
        assert results == [expected]

    @pytest.mark.integration
    @pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="ffmpeg not installed on Windows CI")
    def test_whisper_local_transcriber(self, preview_samples_path):
        comp = LocalWhisperTranscriber(model_name_or_path="medium", whisper_params={"language": "english"})
        comp.warm_up()
        output = comp.run(
            audio_files=[
                preview_samples_path / "audio" / "this is the content of the document.wav",
                str((preview_samples_path / "audio" / "the context for this answer is here.wav").absolute()),
                open(preview_samples_path / "audio" / "answer.wav", "rb"),
            ]
        )
        docs = output["documents"]
        assert len(docs) == 3

        assert docs[0].content.strip().lower() == "this is the content of the document."
        assert preview_samples_path / "audio" / "this is the content of the document.wav" == docs[0].meta["audio_file"]

        assert docs[1].content.strip().lower() == "the context for this answer is here."
        assert (
            str((preview_samples_path / "audio" / "the context for this answer is here.wav").absolute())
            == docs[1].meta["audio_file"]
        )

        assert docs[2].content.strip().lower() == "answer."
        assert docs[2].meta["audio_file"] == "<<binary stream>>"
