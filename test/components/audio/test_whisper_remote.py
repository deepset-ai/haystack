# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

import haystack.components.audio.whisper_remote as whisper_remote_module
from haystack.components.audio.whisper_remote import RemoteWhisperTranscriber
from haystack.dataclasses import ByteStream
from haystack.utils import Secret


class TestRemoteWhisperTranscriber:
    def test_init_default(self):
        transcriber = RemoteWhisperTranscriber(api_key=Secret.from_token("test_api_key"))
        assert transcriber.api_key == Secret.from_token("test_api_key")
        assert transcriber.model == "whisper-1"
        assert transcriber.organization is None
        assert transcriber.whisper_params == {"response_format": "json"}
        assert transcriber.client is None
        assert transcriber.async_client is None

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
        assert transcriber.client is None
        assert transcriber.async_client is None

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
                "http_client_kwargs": None,
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
            http_client_kwargs={"proxy": "http://localhost:8080"},
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
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
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
                "http_client_kwargs": None,
                "response_format": "json",
            },
        }

        transcriber = RemoteWhisperTranscriber.from_dict(data)

        assert transcriber.model == "whisper-1"
        assert transcriber.organization is None
        assert transcriber.api_base_url == "https://api.openai.com/v1"
        assert transcriber.whisper_params == {"response_format": "json"}
        assert transcriber.http_client_kwargs is None

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

        transcriber = RemoteWhisperTranscriber.from_dict(data)
        assert transcriber.model == "whisper-1"
        assert transcriber.client is None
        assert transcriber.async_client is None

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


class TestRemoteWhisperTranscriberAsync:
    @pytest.mark.asyncio
    async def test_run_async_with_path(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        transcriber = RemoteWhisperTranscriber()

        mock_client = Mock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=Mock(text="this is the content of the document.")
        )
        transcriber.async_client = mock_client

        output = await transcriber.run_async(sources=["test/test_files/audio/this is the content of the document.wav"])

        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "this is the content of the document."
        assert docs[0].meta["file_path"] == "test/test_files/audio/this is the content of the document.wav"
        mock_client.audio.transcriptions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_async_with_bytestream(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        transcriber = RemoteWhisperTranscriber()

        mock_client = Mock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=Mock(text="answer."))
        transcriber.async_client = mock_client

        source = ByteStream(data=b"fake audio bytes")
        source.meta["file_path"] = "answer.wav"
        output = await transcriber.run_async(sources=[source])

        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "answer."
        assert docs[0].meta["file_path"] == "answer.wav"
        mock_client.audio.transcriptions.create.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    async def test_whisper_remote_transcriber_async(self, test_files_path):
        transcriber = RemoteWhisperTranscriber()

        paths = [
            test_files_path / "audio" / "this is the content of the document.wav",
            str(test_files_path / "audio" / "the context for this answer is here.wav"),
            ByteStream.from_file_path(test_files_path / "audio" / "answer.wav"),
        ]

        output = await transcriber.run_async(sources=paths)

        docs = output["documents"]
        assert len(docs) == 3
        assert docs[0].content.strip().lower() == "this is the content of the document."
        assert test_files_path / "audio" / "this is the content of the document.wav" == docs[0].meta["file_path"]

        assert docs[1].content.strip().lower() == "the context for this answer is here."
        assert str(test_files_path / "audio" / "the context for this answer is here.wav") == docs[1].meta["file_path"]

        assert docs[2].content.strip().lower() == "answer."


@pytest.fixture
def mock_openai_clients(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    sync_cls = MagicMock(name="OpenAI")
    async_cls = MagicMock(name="AsyncOpenAI")
    async_cls.return_value.close = AsyncMock()
    monkeypatch.setattr(whisper_remote_module, "OpenAI", sync_cls)
    monkeypatch.setattr(whisper_remote_module, "AsyncOpenAI", async_cls)
    return sync_cls, async_cls


class TestComponentLifecycle:
    def test_warm_up_resolves_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        transcriber = RemoteWhisperTranscriber()
        transcriber.warm_up()
        assert transcriber.client.api_key == "test_api_key"

    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        transcriber = RemoteWhisperTranscriber()
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            transcriber.warm_up()

    def test_sync_lifecycle(self, mock_openai_clients):
        sync_cls, _ = mock_openai_clients
        transcriber = RemoteWhisperTranscriber()
        assert transcriber.client is None
        assert transcriber.async_client is None

        transcriber.warm_up()
        assert transcriber.client is sync_cls.return_value
        assert transcriber.async_client is None

        transcriber.close()
        sync_cls.return_value.close.assert_called_once()
        assert transcriber.client is None

    async def test_async_lifecycle(self, mock_openai_clients):
        _, async_cls = mock_openai_clients
        transcriber = RemoteWhisperTranscriber()

        await transcriber.warm_up_async()
        assert transcriber.async_client is async_cls.return_value
        assert transcriber.client is None

        await transcriber.close_async()
        async_cls.return_value.close.assert_awaited_once()
        assert transcriber.async_client is None

    async def test_close_is_safe_without_warm_up(self, mock_openai_clients):
        transcriber = RemoteWhisperTranscriber()
        transcriber.close()
        await transcriber.close_async()
        assert transcriber.client is None
        assert transcriber.async_client is None

    async def test_close_and_close_async_are_independent(self, mock_openai_clients):
        transcriber = RemoteWhisperTranscriber()
        transcriber.warm_up()
        await transcriber.warm_up_async()

        transcriber.close()
        assert transcriber.client is None
        assert transcriber.async_client is not None

        await transcriber.close_async()
        assert transcriber.async_client is None
