# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from haystack import Pipeline
from haystack.components.fetchers import LinkContentFetcher
from haystack.dataclasses import Document, ByteStream
from haystack.components.audio import LocalWhisperTranscriber
from haystack.utils.device import ComponentDevice, Device


SAMPLES_PATH = Path(__file__).parent.parent.parent / "test_files"


class TestLocalWhisperTranscriber:
    def test_init(self):
        transcriber = LocalWhisperTranscriber(
            model="large-v2"
        )  # Doesn't matter if it's huge, the model is not loaded in init.
        assert transcriber.model == "large-v2"
        assert transcriber.device == ComponentDevice.resolve_device(None)
        assert transcriber._model is None

    def test_init_wrong_model(self):
        with pytest.raises(ValueError, match="Model name 'whisper-1' not recognized"):
            LocalWhisperTranscriber(model="whisper-1")

    def test_to_dict(self):
        transcriber = LocalWhisperTranscriber()
        data = transcriber.to_dict()
        assert data == {
            "type": "haystack.components.audio.whisper_local.LocalWhisperTranscriber",
            "init_parameters": {
                "model": "large",
                "device": ComponentDevice.resolve_device(None).to_dict(),
                "whisper_params": {},
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        transcriber = LocalWhisperTranscriber(
            model="tiny",
            device=ComponentDevice.from_str("cuda:0"),
            whisper_params={"return_segments": True, "temperature": [0.1, 0.6, 0.8]},
        )
        data = transcriber.to_dict()
        assert data == {
            "type": "haystack.components.audio.whisper_local.LocalWhisperTranscriber",
            "init_parameters": {
                "model": "tiny",
                "device": ComponentDevice.from_str("cuda:0").to_dict(),
                "whisper_params": {"return_segments": True, "temperature": [0.1, 0.6, 0.8]},
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.audio.whisper_local.LocalWhisperTranscriber",
            "init_parameters": {
                "model": "tiny",
                "device": ComponentDevice.from_single(Device.cpu()).to_dict(),
                "whisper_params": {},
            },
        }
        transcriber = LocalWhisperTranscriber.from_dict(data)
        assert transcriber.model == "tiny"
        assert transcriber.device == ComponentDevice.from_single(Device.cpu())
        assert transcriber.whisper_params == {}
        assert transcriber._model is None

    def test_from_dict_no_default_parameters(self):
        data = {"type": "haystack.components.audio.whisper_local.LocalWhisperTranscriber", "init_parameters": {}}
        transcriber = LocalWhisperTranscriber.from_dict(data)
        assert transcriber.model == "large"
        assert transcriber.device == ComponentDevice.resolve_device(None)
        assert transcriber.whisper_params == {}

    def test_from_dict_none_device(self):
        data = {
            "type": "haystack.components.audio.whisper_local.LocalWhisperTranscriber",
            "init_parameters": {"model": "tiny", "device": None, "whisper_params": {}},
        }
        transcriber = LocalWhisperTranscriber.from_dict(data)
        assert transcriber.model == "tiny"
        assert transcriber.device == ComponentDevice.resolve_device(None)
        assert transcriber.whisper_params == {}
        assert transcriber._model is None

    def test_warmup(self):
        with patch("haystack.components.audio.whisper_local.whisper") as mocked_whisper:
            transcriber = LocalWhisperTranscriber(model="large-v2", device=ComponentDevice.from_str("cpu"))
            mocked_whisper.load_model.assert_not_called()
            transcriber.warm_up()
            mocked_whisper.load_model.assert_called_once_with("large-v2", device=torch.device(type="cpu"))

    def test_warmup_doesnt_reload(self):
        with patch("haystack.components.audio.whisper_local.whisper") as mocked_whisper:
            transcriber = LocalWhisperTranscriber(model="large-v2")
            transcriber.warm_up()
            transcriber.warm_up()
            mocked_whisper.load_model.assert_called_once()

    def test_run_with_path(self):
        comp = LocalWhisperTranscriber(model="large-v2")
        comp._model = MagicMock()
        comp._model.transcribe.return_value = {
            "text": "test transcription",
            "other_metadata": ["other", "meta", "data"],
        }
        results = comp.run(sources=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"])
        expected = Document(
            content="test transcription",
            meta={
                "audio_file": SAMPLES_PATH / "audio" / "this is the content of the document.wav",
                "other_metadata": ["other", "meta", "data"],
            },
        )
        assert results["documents"] == [expected]

    def test_run_with_str(self):
        comp = LocalWhisperTranscriber(model="large-v2")
        comp._model = MagicMock()
        comp._model.transcribe.return_value = {
            "text": "test transcription",
            "other_metadata": ["other", "meta", "data"],
        }
        results = comp.run(
            sources=[str((SAMPLES_PATH / "audio" / "this is the content of the document.wav").absolute())]
        )
        expected = Document(
            content="test transcription",
            meta={
                "audio_file": (SAMPLES_PATH / "audio" / "this is the content of the document.wav").absolute(),
                "other_metadata": ["other", "meta", "data"],
            },
        )
        assert results["documents"] == [expected]

    def test_transcribe(self):
        comp = LocalWhisperTranscriber(model="large-v2")
        comp._model = MagicMock()
        comp._model.transcribe.return_value = {
            "text": "test transcription",
            "other_metadata": ["other", "meta", "data"],
        }
        results = comp.transcribe(sources=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"])
        expected = Document(
            content="test transcription",
            meta={
                "audio_file": SAMPLES_PATH / "audio" / "this is the content of the document.wav",
                "other_metadata": ["other", "meta", "data"],
            },
        )
        assert results == [expected]

    def test_transcribe_stream(self):
        comp = LocalWhisperTranscriber(model="large-v2")
        comp._model = MagicMock()
        comp._model.transcribe.return_value = {
            "text": "test transcription",
            "other_metadata": ["other", "meta", "data"],
        }
        path = SAMPLES_PATH / "audio" / "this is the content of the document.wav"
        bs = ByteStream.from_file_path(path)
        bs.meta["file_path"] = path
        results = comp.transcribe(sources=[bs])
        expected = Document(
            content="test transcription", meta={"audio_file": path, "other_metadata": ["other", "meta", "data"]}
        )
        assert results == [expected]

    @pytest.mark.integration
    @pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="ffmpeg not installed on Windows CI")
    def test_whisper_local_transcriber(self, test_files_path):
        comp = LocalWhisperTranscriber(model="tiny", whisper_params={"language": "english"})
        comp.warm_up()
        output = comp.run(
            sources=[
                test_files_path / "audio" / "this is the content of the document.wav",
                str((test_files_path / "audio" / "the context for this answer is here.wav").absolute()),
                ByteStream.from_file_path(test_files_path / "audio" / "answer.wav", "rb"),
            ]
        )
        docs = output["documents"]
        assert len(docs) == 3

        assert all(
            word in docs[0].content.strip().lower() for word in {"content", "the", "document"}
        ), f"Expected words not found in: {docs[0].content.strip().lower()}"
        assert test_files_path / "audio" / "this is the content of the document.wav" == docs[0].meta["audio_file"]

        assert all(
            word in docs[1].content.strip().lower() for word in {"context", "answer"}
        ), f"Expected words not found in: {docs[1].content.strip().lower()}"
        path = test_files_path / "audio" / "the context for this answer is here.wav"
        assert path.absolute() == docs[1].meta["audio_file"]

        assert docs[2].content.strip().lower() == "answer."
        # meta.audio_file should contain the temp path where we dumped the audio bytes
        assert docs[2].meta["audio_file"]

    @pytest.mark.integration
    @pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="ffmpeg not installed on Windows CI")
    def test_whisper_local_transcriber_pipeline_and_url_source(self):
        pipe = Pipeline()
        pipe.add_component("fetcher", LinkContentFetcher())
        pipe.add_component("transcriber", LocalWhisperTranscriber(model="tiny"))

        pipe.connect("fetcher", "transcriber")
        result = pipe.run(
            data={
                "fetcher": {
                    "urls": [
                        "https://github.com/deepset-ai/haystack/raw/refs/heads/main/test/test_files/audio/MLK_Something_happening.mp3"  # noqa: E501
                    ]
                }
            }
        )
        assert "masses of people" in result["transcriber"]["documents"][0].content
