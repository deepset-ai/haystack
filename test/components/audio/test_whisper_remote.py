# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
import pytest

from haystack import Pipeline
from haystack.components.audio.whisper_remote import RemoteWhisperTranscriber
from haystack.components.fetchers import LinkContentFetcher
from haystack.dataclasses import ByteStream
from haystack.utils import Secret


class TestRemoteWhisperTranscriber:
    def test_init_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            RemoteWhisperTranscriber()

    def test_init_key_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        t = RemoteWhisperTranscriber()
        assert t.client.api_key == "test_api_key"

    def test_init_key_module_env_and_global_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key_2")
        t = RemoteWhisperTranscriber()
        assert t.client.api_key == "test_api_key_2"

    def test_init_default(self):
        transcriber = RemoteWhisperTranscriber(api_key=Secret.from_token("test_api_key"))
        assert transcriber.client.api_key == "test_api_key"
        assert transcriber.model == "whisper-1"
        assert transcriber.organization is None
        assert transcriber.whisper_params == {"response_format": "json"}

    def test_init_custom_parameters(self):
        transcriber = RemoteWhisperTranscriber(
            api_key=Secret.from_token("test_api_key"),
            model="whisper-1",
            organization="test-org",
            api_base_url="test_api_url",
            language="en",
            prompt="test-prompt",
            response_format="json",
            temperature="0.5",
        )

        assert transcriber.model == "whisper-1"
        assert transcriber.api_key == Secret.from_token("test_api_key")
        assert transcriber.organization == "test-org"
        assert transcriber.api_base_url == "test_api_url"
        assert transcriber.whisper_params == {
            "language": "en",
            "prompt": "test-prompt",
            "response_format": "json",
            "temperature": "0.5",
        }

    def test_to_dict_default_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        transcriber = RemoteWhisperTranscriber()
        data = transcriber.to_dict()
        assert data == {
            "type": "haystack.components.audio.whisper_remote.RemoteWhisperTranscriber",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "whisper-1",
                "api_base_url": None,
                "organization": None,
                "response_format": "json",
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        transcriber = RemoteWhisperTranscriber(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="whisper-1",
            organization="test-org",
            api_base_url="test_api_url",
            language="en",
            prompt="test-prompt",
            response_format="json",
            temperature="0.5",
        )
        data = transcriber.to_dict()
        assert data == {
            "type": "haystack.components.audio.whisper_remote.RemoteWhisperTranscriber",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "whisper-1",
                "organization": "test-org",
                "api_base_url": "test_api_url",
                "language": "en",
                "prompt": "test-prompt",
                "response_format": "json",
                "temperature": "0.5",
            },
        }

    def test_from_dict_with_default_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")

        data = {
            "type": "haystack.components.audio.whisper_remote.RemoteWhisperTranscriber",
            "init_parameters": {
                "model": "whisper-1",
                "api_base_url": "https://api.openai.com/v1",
                "organization": None,
                "response_format": "json",
            },
        }

        transcriber = RemoteWhisperTranscriber.from_dict(data)

        assert transcriber.model == "whisper-1"
        assert transcriber.organization is None
        assert transcriber.api_base_url == "https://api.openai.com/v1"
        assert transcriber.whisper_params == {"response_format": "json"}

    def test_from_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")

        data = {
            "type": "haystack.components.audio.whisper_remote.RemoteWhisperTranscriber",
            "init_parameters": {
                "model": "whisper-1",
                "organization": "test-org",
                "api_base_url": "test_api_url",
                "language": "en",
                "prompt": "test-prompt",
                "response_format": "json",
                "temperature": "0.5",
            },
        }
        transcriber = RemoteWhisperTranscriber.from_dict(data)

        assert transcriber.model == "whisper-1"
        assert transcriber.organization == "test-org"
        assert transcriber.api_base_url == "test_api_url"
        assert transcriber.whisper_params == {
            "language": "en",
            "prompt": "test-prompt",
            "response_format": "json",
            "temperature": "0.5",
        }

    def test_from_dict_with_default_parameters_no_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        data = {
            "type": "haystack.components.audio.whisper_remote.RemoteWhisperTranscriber",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "whisper-1",
                "api_base_url": "https://api.openai.com/v1",
                "organization": None,
                "response_format": "json",
            },
        }

        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            RemoteWhisperTranscriber.from_dict(data)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_whisper_remote_transcriber(self, test_files_path):
        transcriber = RemoteWhisperTranscriber()

        paths = [
            test_files_path / "audio" / "this is the content of the document.wav",
            str(test_files_path / "audio" / "the context for this answer is here.wav"),
            ByteStream.from_file_path(test_files_path / "audio" / "answer.wav"),
        ]

        output = transcriber.run(sources=paths)

        docs = output["documents"]
        assert len(docs) == 3
        assert docs[0].content.strip().lower() == "this is the content of the document."
        assert test_files_path / "audio" / "this is the content of the document.wav" == docs[0].meta["file_path"]

        assert docs[1].content.strip().lower() == "the context for this answer is here."
        assert str(test_files_path / "audio" / "the context for this answer is here.wav") == docs[1].meta["file_path"]

        assert docs[2].content.strip().lower() == "answer."

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_whisper_remote_transcriber_pipeline_and_url_source(self):
        pipe = Pipeline()
        pipe.add_component("fetcher", LinkContentFetcher())
        pipe.add_component("transcriber", RemoteWhisperTranscriber())

        pipe.connect("fetcher", "transcriber")
        result = pipe.run(
            data={
                "fetcher": {
                    "urls": [
                        "https://github.com/deepset-ai/haystack/raw/refs/heads/main/test/test_files/audio/MLK_Something_happening.mp3"
                    ]  # noqa: E501
                }
            }
        )
        assert "masses of people" in result["transcriber"]["documents"][0].content
