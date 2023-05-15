import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import whisper
from generalimport import FakeModule, MissingOptionalDependency

from haystack import is_imported
from haystack.preview.dataclasses import Document
from haystack.preview.components import LocalWhisperTranscriber

from test.preview.components.base import BaseTestComponent


SAMPLES_PATH = Path(__file__).parent.parent / "test_files"


class FakeWhisperModel(MagicMock):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)

    def transcribe(_, audio_file, **kwargs):
        return {"text": "test transcription", "other_metadata": ["other", "meta", "data"], "kwargs received": kwargs}


class Test_LocalWhisperTranscriber(BaseTestComponent):
    @pytest.fixture
    def components(self):
        return [LocalWhisperTranscriber(model_name_or_path="large-v2")]

    @pytest.fixture(autouse=True)
    def mock_model(self, monkeypatch):
        load_model = MagicMock()
        load_model.side_effect = [FakeWhisperModel()]
        monkeypatch.setattr(whisper, "load_model", load_model)
        return load_model

    @pytest.mark.unit
    def test_init(self):
        transcriber = LocalWhisperTranscriber(
            model_name_or_path="large-v2"
        )  # Doesn't matter if it's huge, the model is not loaded in init.
        assert transcriber.model_name == "large-v2"
        assert hasattr(transcriber, "device") and transcriber.device == torch.device("cpu")
        assert hasattr(transcriber, "_model") and transcriber._model is None

    @pytest.mark.unit
    def test_warmup(self, mock_model):
        component = LocalWhisperTranscriber(model_name_or_path="large-v2")
        assert hasattr(component, "_model")
        assert not isinstance(component._model, FakeWhisperModel)
        mock_model.assert_not_called()

        component.warm_up()
        assert hasattr(component, "_model")
        assert isinstance(component._model, FakeWhisperModel)
        mock_model.assert_called_with("large-v2", device=torch.device(type="cpu"))

    @pytest.mark.unit
    def test_warmup_doesnt_reload(self, mock_model):
        component = LocalWhisperTranscriber(model_name_or_path="large-v2")
        component.warm_up()
        component.warm_up()
        mock_model.assert_called_once()

    @pytest.mark.unit
    def test_transcribe_to_documents(self):
        comp = LocalWhisperTranscriber(model_name_or_path="large-v2")
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
