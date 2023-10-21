from typing import Union, BinaryIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from copy import deepcopy


import pytest
import openai
from openai.util import convert_to_openai_object


from haystack.preview.dataclasses import Document
from haystack.preview.components.audio.whisper_remote import RemoteWhisperTranscriber


def mock_openai_response(
    file: Union[str, Path, BinaryIO], model: str = "whisper-1", response_format="json", **kwargs
) -> openai.openai_object.OpenAIObject:
    if isinstance(file, (str, Path)):
        file_path = str(file)
    else:
        file_path = file.name
    if response_format == "json":
        dict_response = {"text": f"model: {model}, file: str{file_path}, test transcription"}
    else:
        dict_response = {}

    return convert_to_openai_object(dict_response)


class TestRemoteWhisperTranscriber:
    @pytest.mark.unit
    def test_init_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        error_msg = "RemoteWhisperTranscriber expects an OpenAI API key."
        with pytest.raises(ValueError, match=error_msg):
            RemoteWhisperTranscriber(api_key=None)

    @pytest.mark.unit
    def test_init_default(self):
        transcriber = RemoteWhisperTranscriber(api_key="test_api_key")

        assert transcriber.api_key == "test_api_key"
        assert transcriber.model_name == "whisper-1"
        assert transcriber.organization is None
        assert transcriber.api_base_url == "https://api.openai.com/v1"
        assert transcriber.whisper_params == {"response_format": "json"}

    @pytest.mark.unit
    def test_init_custom_parameters(self):
        transcriber = RemoteWhisperTranscriber(
            api_key="test_api_key",
            model_name="whisper-1",
            organization="test-org",
            api_base_url="test_api_url",
            language="en",
            prompt="test-prompt",
            response_format="json",
            temperature="0.5",
        )

        assert transcriber.api_key == "test_api_key"
        assert transcriber.model_name == "whisper-1"
        assert transcriber.organization == "test-org"
        assert transcriber.api_base_url == "test_api_url"
        assert transcriber.whisper_params == {
            "language": "en",
            "prompt": "test-prompt",
            "response_format": "json",
            "temperature": "0.5",
        }

    @pytest.mark.unit
    def test_to_dict_default_parameters(self):
        transcriber = RemoteWhisperTranscriber(api_key="test_api_key")
        data = transcriber.to_dict()
        assert data == {
            "type": "RemoteWhisperTranscriber",
            "init_parameters": {
                "model_name": "whisper-1",
                "api_base_url": "https://api.openai.com/v1",
                "organization": None,
                "response_format": "json",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        transcriber = RemoteWhisperTranscriber(
            api_key="test_api_key",
            model_name="whisper-1",
            organization="test-org",
            api_base_url="test_api_url",
            language="en",
            prompt="test-prompt",
            response_format="json",
            temperature="0.5",
        )
        data = transcriber.to_dict()
        assert data == {
            "type": "RemoteWhisperTranscriber",
            "init_parameters": {
                "model_name": "whisper-1",
                "organization": "test-org",
                "api_base_url": "test_api_url",
                "language": "en",
                "prompt": "test-prompt",
                "response_format": "json",
                "temperature": "0.5",
            },
        }

    def test_from_dict_with_defualt_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")

        data = {
            "type": "RemoteWhisperTranscriber",
            "init_parameters": {
                "model_name": "whisper-1",
                "api_base_url": "https://api.openai.com/v1",
                "organization": None,
                "response_format": "json",
            },
        }

        transcriber = RemoteWhisperTranscriber.from_dict(data)

        assert transcriber.api_key == "test_api_key"
        assert transcriber.model_name == "whisper-1"
        assert transcriber.organization is None
        assert transcriber.api_base_url == "https://api.openai.com/v1"
        assert transcriber.whisper_params == {"response_format": "json"}

    def test_from_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")

        data = {
            "type": "RemoteWhisperTranscriber",
            "init_parameters": {
                "model_name": "whisper-1",
                "organization": "test-org",
                "api_base_url": "test_api_url",
                "language": "en",
                "prompt": "test-prompt",
                "response_format": "json",
                "temperature": "0.5",
            },
        }
        transcriber = RemoteWhisperTranscriber.from_dict(data)

        assert transcriber.api_key == "test_api_key"
        assert transcriber.model_name == "whisper-1"
        assert transcriber.organization == "test-org"
        assert transcriber.api_base_url == "test_api_url"
        assert transcriber.whisper_params == {
            "language": "en",
            "prompt": "test-prompt",
            "response_format": "json",
            "temperature": "0.5",
        }

    def test_from_dict_with_defualt_parameters_no_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        data = {
            "type": "RemoteWhisperTranscriber",
            "init_parameters": {
                "model_name": "whisper-1",
                "api_base_url": "https://api.openai.com/v1",
                "organization": None,
                "response_format": "json",
            },
        }

        with pytest.raises(ValueError, match="RemoteWhisperTranscriber expects an OpenAI API key."):
            RemoteWhisperTranscriber.from_dict(data)

    @pytest.mark.unit
    def test_run_with_path(self, preview_samples_path):
        model = "whisper-1"
        file_path = preview_samples_path / "audio" / "this is the content of the document.wav"
        with patch("haystack.preview.components.audio.whisper_remote.openai.Audio") as openai_audio_patch:
            openai_audio_patch.transcribe.side_effect = mock_openai_response

            transcriber = RemoteWhisperTranscriber(api_key="test_api_key", model_name=model, response_format="json")

            result = transcriber.run(audio_files=[file_path])

            assert result["documents"][0].text == f"model: {model}, file: str{file_path}, test transcription"

            open_file = open(file_path, "rb")
            openai_audio_patch.transcribe.assert_called_once_with(file=open_file, model=model, response_format="json")
