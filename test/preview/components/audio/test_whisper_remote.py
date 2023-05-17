import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import whisper
from generalimport import FakeModule

from haystack.preview.dataclasses import Document
from haystack.preview.components import RemoteWhisperTranscriber

from test.preview.components.test_component_base import BaseTestComponent


SAMPLES_PATH = Path(__file__).parent.parent / "test_files"


class TestRemoteWhisperTranscriber(BaseTestComponent):
    """
    Tests for RemoteWhisperTranscriber.
    """

    @pytest.fixture
    def components(self):
        return [RemoteWhisperTranscriber(api_key="just a test")]

    @pytest.fixture
    def mock_models(self, monkeypatch):
        def mock_transcribe(_, audio_file, **kwargs):
            return {
                "text": "test transcription",
                "other_metadata": ["other", "meta", "data"],
                "kwargs received": kwargs,
            }

        monkeypatch.setattr(RemoteWhisperTranscriber, "_transcribe_with_api", mock_transcribe)

    @pytest.mark.unit
    def test_init_remote_unknown_model(self):
        with pytest.raises(ValueError, match="not recognized"):
            RemoteWhisperTranscriber(model_name_or_path="anything")

    @pytest.mark.unit
    def test_init_default_remote_missing_key(self):
        with pytest.raises(ValueError, match="API key"):
            RemoteWhisperTranscriber()

    @pytest.mark.unit
    def test_init_explicit_remote_missing_key(self):
        with pytest.raises(ValueError, match="API key"):
            RemoteWhisperTranscriber(model_name_or_path="whisper-1")

    @pytest.mark.unit
    def test_init_remote(self):
        transcriber = RemoteWhisperTranscriber(api_key="just a test")
        assert transcriber.model_name == "whisper-1"
        assert not transcriber.use_local_whisper
        assert not hasattr(transcriber, "device")
        assert hasattr(transcriber, "_model") and transcriber._model is None

    @pytest.mark.unit
    def test_init_local(self):
        transcriber = RemoteWhisperTranscriber(model_name_or_path="large-v2")
        assert transcriber.model_name == "large-v2"  # Doesn't matter if it's huge, the model is not loaded in init.
        assert transcriber.use_local_whisper
        assert hasattr(transcriber, "device") and transcriber.device == torch.device("cpu")
        assert hasattr(transcriber, "_model") and transcriber._model is None

    @pytest.mark.unit
    def test_init_local_with_api_key(self):
        transcriber = RemoteWhisperTranscriber(model_name_or_path="large-v2")
        assert transcriber.model_name == "large-v2"  # Doesn't matter if it's huge, the model is not loaded in init.
        assert transcriber.use_local_whisper
        assert hasattr(transcriber, "device") and transcriber.device == torch.device("cpu")
        assert hasattr(transcriber, "_model") and transcriber._model is None

    @pytest.mark.unit
    def test_init_missing_whisper_lib_local_model(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "whisper", FakeModule(spec=MagicMock(), message="test"))
        with pytest.raises(ValueError, match="audio extra"):
            RemoteWhisperTranscriber(model_name_or_path="large-v2")

    @pytest.mark.unit
    def test_init_missing_whisper_lib_remote_model(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "whisper", FakeModule(spec=MagicMock(), message="test"))
        # Should not fail if the lib is missing and we're using API
        RemoteWhisperTranscriber(model_name_or_path="whisper-1", api_key="doesn't matter")

    @pytest.mark.unit
    def test_warmup_remote_model(self, monkeypatch):
        load_model = MagicMock()
        monkeypatch.setattr(whisper, "load_model", load_model)
        component = RemoteWhisperTranscriber(model_name_or_path="whisper-1", api_key="doesn't matter")
        component.warm_up()
        assert not load_model.called

    @pytest.mark.unit
    def test_warmup_local_model(self, monkeypatch):
        load_model = MagicMock()
        load_model.side_effect = ["FAKE MODEL"]
        monkeypatch.setattr(whisper, "load_model", load_model)

        component = RemoteWhisperTranscriber(model_name_or_path="large-v2")
        component.warm_up()

        assert hasattr(component, "_model")
        assert component._model == "FAKE MODEL"
        load_model.assert_called_with("large-v2", device=torch.device(type="cpu"))

    @pytest.mark.unit
    def test_warmup_local_model_doesnt_reload(self, monkeypatch):
        load_model = MagicMock()
        monkeypatch.setattr(whisper, "load_model", load_model)
        component = RemoteWhisperTranscriber(model_name_or_path="large-v2")
        component.warm_up()
        component.warm_up()
        load_model.assert_called_once()

    @pytest.mark.unit
    def test_transcribe_to_documents(self, mock_models):
        comp = RemoteWhisperTranscriber(model_name_or_path="large-v2")
        output = comp.transcribe(audio_files=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"])
        assert output == [
            Document(
                content="test transcription",
                metadata={
                    "audio_file": SAMPLES_PATH / "audio" / "this is the content of the document.wav",
                    "other_metadata": ["other", "meta", "data"],
                    "kwargs received": {},
                },
            )
        ]
