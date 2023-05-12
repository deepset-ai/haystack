import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import whisper
from generalimport import FakeModule

from haystack.preview.dataclasses import Document
from haystack.preview.components import LocalWhisperTranscriber

from test.preview.components.test_component_base import BaseTestComponent


SAMPLES_PATH = Path(__file__).parent.parent / "test_files"


class TestTranscriber(BaseTestComponent):
    """
    Tests for WhisperTranscriber.
    """

    @pytest.fixture
    def components(self):
        return [LocalWhisperTranscriber(model_name_or_path="large-v2")]

    @pytest.fixture
    def mock_models(self, monkeypatch):
        def mock_transcribe(_, audio_file, **kwargs):
            return {
                "text": "test transcription",
                "other_metadata": ["other", "meta", "data"],
                "kwargs received": kwargs,
            }

        monkeypatch.setattr(LocalWhisperTranscriber, "_transcribe_with_api", mock_transcribe)
        monkeypatch.setattr(LocalWhisperTranscriber, "_transcribe_locally", mock_transcribe)
        monkeypatch.setattr(LocalWhisperTranscriber, "warm_up", lambda self: None)

    @pytest.mark.unit
    def test_init(self):
        transcriber = LocalWhisperTranscriber(model_name_or_path="large-v2")
        assert transcriber.model_name == "large-v2"  # Doesn't matter if it's huge, the model is not loaded in init.
        assert transcriber.use_local_whisper
        assert hasattr(transcriber, "device") and transcriber.device == torch.device("cpu")
        assert hasattr(transcriber, "_model") and transcriber._model is None

    @pytest.mark.unit
    def test_init_missing_whisper_lib(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "whisper", FakeModule(spec=MagicMock(), message="test"))
        with pytest.raises(ValueError, match="audio extra"):
            LocalWhisperTranscriber(model_name_or_path="large-v2")

    @pytest.mark.unit
    def test_warmup(self, monkeypatch):
        load_model = MagicMock()
        load_model.side_effect = ["FAKE MODEL"]
        monkeypatch.setattr(whisper, "load_model", load_model)

        component = LocalWhisperTranscriber(model_name_or_path="large-v2")
        component.warm_up()

        assert hasattr(component, "_model")
        assert component._model == "FAKE MODEL"
        load_model.assert_called_with("large-v2", device=torch.device(type="cpu"))

    @pytest.mark.unit
    def test_warmup_doesnt_reload(self, monkeypatch):
        load_model = MagicMock()
        monkeypatch.setattr(whisper, "load_model", load_model)
        component = LocalWhisperTranscriber(model_name_or_path="large-v2")
        component.warm_up()
        component.warm_up()
        load_model.assert_called_once()

    @pytest.mark.unit
    def test_transcribe_to_documents(self, mock_models):
        comp = LocalWhisperTranscriber(model_name_or_path="large-v2")
        output = comp.transcribe_to_documents(
            audio_files=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"]
        )
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
