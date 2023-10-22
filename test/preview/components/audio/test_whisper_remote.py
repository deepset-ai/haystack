from unittest.mock import patch
import os


import pytest
import openai
from openai.util import convert_to_openai_object


from haystack.preview.dataclasses import Document
from haystack.preview.components.audio.whisper_remote import RemoteWhisperTranscriber


def mock_openai_response(response_format="json", **kwargs) -> openai.openai_object.OpenAIObject:
    if response_format == "json":
        dict_response = {"text": "test transcription"}
    # Currently only "json" is supported.
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

        assert openai.api_key == "test_api_key"
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

        assert openai.api_key == "test_api_key"
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

        assert openai.api_key == "test_api_key"
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

        assert openai.api_key == "test_api_key"
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
    def test_run(self, preview_samples_path):
        with patch("haystack.preview.components.audio.whisper_remote.openai.Audio") as openai_audio_patch:
            model = "whisper-1"
            openai_audio_patch.transcribe.side_effect = mock_openai_response

            transcriber = RemoteWhisperTranscriber(api_key="test_api_key", model_name=model, response_format="json")
            with open(preview_samples_path / "audio" / "this is the content of the document.wav", "rb") as audio_stream:
                result = transcriber.run(audio_files=[audio_stream])

                assert result["documents"][0].text == "test transcription"

                openai_audio_patch.transcribe.assert_called_once_with(
                    file=audio_stream, model=model, response_format="json"
                )

    @pytest.mark.unit
    def test_run_with_path(self, preview_samples_path):
        with patch("haystack.preview.components.audio.whisper_remote.openai.Audio") as openai_audio_patch:
            model = "whisper-1"
            openai_audio_patch.transcribe.side_effect = mock_openai_response

            transcriber = RemoteWhisperTranscriber(api_key="test_api_key", model_name=model, response_format="json")

            result = transcriber.run(
                audio_files=[preview_samples_path / "audio" / "this is the content of the document.wav"]
            )

            expected = Document(
                text="test transcription",
                metadata={
                    "audio_file": str(
                        (preview_samples_path / "audio" / "this is the content of the document.wav").absolute()
                    )
                },
            )
            assert result["documents"][0].text == expected.text
            assert result["documents"][0].metadata == expected.metadata

    @pytest.mark.unit
    def test_run_with_str(self, preview_samples_path):
        with patch("haystack.preview.components.audio.whisper_remote.openai.Audio") as openai_audio_patch:
            model = "whisper-1"
            openai_audio_patch.transcribe.side_effect = mock_openai_response

            transcriber = RemoteWhisperTranscriber(api_key="test_api_key", model_name=model, response_format="json")

            result = transcriber.run(
                audio_files=[
                    str((preview_samples_path / "audio" / "this is the content of the document.wav").absolute())
                ]
            )

            expected = Document(
                text="test transcription",
                metadata={
                    "audio_file": str(
                        (preview_samples_path / "audio" / "this is the content of the document.wav").absolute()
                    )
                },
            )
            assert result["documents"][0].text == expected.text
            assert result["documents"][0].metadata == expected.metadata

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_whisper_remote_transcriber(self, preview_samples_path):
        comp = RemoteWhisperTranscriber(api_key=os.environ.get("OPENAI_API_KEY"))

        output = comp.run(
            audio_files=[
                preview_samples_path / "audio" / "this is the content of the document.wav",
                str((preview_samples_path / "audio" / "the context for this answer is here.wav").absolute()),
                open(preview_samples_path / "audio" / "answer.wav", "rb"),
            ]
        )
        docs = output["documents"]
        assert len(docs) == 3

        assert docs[0].text.strip().lower() == "this is the content of the document."
        assert (
            str((preview_samples_path / "audio" / "this is the content of the document.wav").absolute())
            == docs[0].metadata["audio_file"]
        )

        assert docs[1].text.strip().lower() == "the context for this answer is here."
        assert (
            str((preview_samples_path / "audio" / "the context for this answer is here.wav").absolute())
            == docs[1].metadata["audio_file"]
        )

        assert docs[2].text.strip().lower() == "answer."
        assert docs[2].metadata["audio_file"] == "<<binary stream>>"
