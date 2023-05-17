from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from haystack.preview.dataclasses import Document
from haystack.preview.components.audio.whisper_remote import RemoteWhisperTranscriber, OPENAI_TIMEOUT

from test.preview.components.base import BaseTestComponent


SAMPLES_PATH = Path(__file__).parent.parent.parent / "test_files"


class TestRemoteWhisperTranscriber(BaseTestComponent):
    """
    Tests for RemoteWhisperTranscriber.
    """

    @pytest.fixture
    def components(self):
        return [RemoteWhisperTranscriber(api_key="just a test")]

    @pytest.mark.unit
    def test_init_unknown_model(self):
        with pytest.raises(ValueError, match="not recognized"):
            RemoteWhisperTranscriber(model_name="anything", api_key="something")

    @pytest.mark.unit
    def test_init_default(self):
        transcriber = RemoteWhisperTranscriber(api_key="just a test")
        assert transcriber.model_name == "whisper-1"
        assert transcriber.api_key == "just a test"

    @pytest.mark.unit
    def test_init_no_key(self):
        with pytest.raises(ValueError, match="API key is None"):
            RemoteWhisperTranscriber(api_key=None)

    @pytest.mark.unit
    def test_run_with_path(self):
        with patch("haystack.preview.components.audio.whisper_remote.request_with_retry") as mocked_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = '{"text": "test transcription", "other_metadata": ["other", "meta", "data"]}'
            mocked_requests.post.return_value = mock_response
            comp = RemoteWhisperTranscriber(api_key="whatever")

            result = comp.run(audio_files=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"])
            expected = Document(
                content="test transcription",
                metadata={
                    "audio_file": SAMPLES_PATH / "audio" / "this is the content of the document.wav",
                    "other_metadata": ["other", "meta", "data"],
                },
            )
            assert result.documents == [expected]

    @pytest.mark.unit
    def test_run_with_str(self):
        with patch("haystack.preview.components.audio.whisper_remote.request_with_retry") as mocked_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = '{"text": "test transcription", "other_metadata": ["other", "meta", "data"]}'
            mocked_requests.post.return_value = mock_response
            comp = RemoteWhisperTranscriber(api_key="whatever")

            result = comp.run(
                audio_files=[str((SAMPLES_PATH / "audio" / "this is the content of the document.wav").absolute())]
            )
            expected = Document(
                content="test transcription",
                metadata={
                    "audio_file": str((SAMPLES_PATH / "audio" / "this is the content of the document.wav").absolute()),
                    "other_metadata": ["other", "meta", "data"],
                },
            )
            assert result.documents == [expected]

    @pytest.mark.unit
    def test_transcribe_with_stream(self):
        with patch("haystack.preview.components.audio.whisper_remote.request_with_retry") as mocked_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = '{"text": "test transcription", "other_metadata": ["other", "meta", "data"]}'
            mocked_requests.post.return_value = mock_response
            comp = RemoteWhisperTranscriber(api_key="whatever")

            with open(SAMPLES_PATH / "audio" / "this is the content of the document.wav", "rb") as audio_stream:
                result = comp.transcribe(audio_files=[audio_stream])
                expected = Document(
                    content="test transcription",
                    metadata={"audio_file": "<<binary stream>>", "other_metadata": ["other", "meta", "data"]},
                )
                assert result == [expected]

    @pytest.mark.unit
    def test_api_transcription(self):
        with patch("haystack.preview.components.audio.whisper_remote.request_with_retry") as mocked_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = '{"text": "test transcription", "other_metadata": ["other", "meta", "data"]}'
            mocked_requests.post.return_value = mock_response
            comp = RemoteWhisperTranscriber(api_key="whatever")

            comp.run(audio_files=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"])

            requests_params = mocked_requests.post.call_args.kwargs
            requests_params.pop("files")
            assert requests_params == {
                "url": "https://api.openai.com/v1/audio/transcriptions",
                "data": {"model": "whisper-1"},
                "headers": {"Authorization": f"Bearer whatever"},
                "timeout": OPENAI_TIMEOUT,
            }

    @pytest.mark.unit
    def test_api_translation(self):
        with patch("haystack.preview.components.audio.whisper_remote.request_with_retry") as mocked_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = '{"text": "test transcription", "other_metadata": ["other", "meta", "data"]}'
            mocked_requests.post.return_value = mock_response
            comp = RemoteWhisperTranscriber(api_key="whatever")

            comp.run(
                audio_files=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"],
                whisper_params={"translate": True},
            )

            requests_params = mocked_requests.post.call_args.kwargs
            requests_params.pop("files")
            assert requests_params == {
                "url": "https://api.openai.com/v1/audio/translations",
                "data": {"model": "whisper-1"},
                "headers": {"Authorization": f"Bearer whatever"},
                "timeout": OPENAI_TIMEOUT,
            }

    @pytest.mark.unit
    def test_api_fails(self):
        with patch("haystack.preview.components.audio.whisper_remote.request_with_retry") as mocked_requests:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.content = '{"error": "something went wrong on our end!"}'
            mocked_requests.post.return_value = mock_response
            comp = RemoteWhisperTranscriber(api_key="whatever")

            with pytest.raises(requests.HTTPError):
                comp.run(audio_files=[SAMPLES_PATH / "audio" / "this is the content of the document.wav"])
