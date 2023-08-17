from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from haystack.preview.dataclasses import Document
from haystack.preview.components.audio.whisper_remote import RemoteWhisperTranscriber, OPENAI_TIMEOUT

from test.preview.components.base import BaseTestComponent


class TestRemoteWhisperTranscriber(BaseTestComponent):
    """
    Tests for RemoteWhisperTranscriber.
    """

    @pytest.mark.unit
    def test_save_load(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(RemoteWhisperTranscriber(api_key="just a test"), tmp_path)

    @pytest.mark.unit
    def test_init_unknown_model(self):
        with pytest.raises(ValueError, match="not recognized"):
            RemoteWhisperTranscriber(model_name="anything", api_key="something")

    @pytest.mark.unit
    def test_init_default(self):
        transcriber = RemoteWhisperTranscriber(api_key="just a test")
        assert transcriber.model_name == "whisper-1"
        assert transcriber.api_key == "just a test"
        assert transcriber.api_base == "https://api.openai.com/v1"

    @pytest.mark.unit
    def test_init_no_key(self):
        with pytest.raises(ValueError, match="API key is None"):
            RemoteWhisperTranscriber(api_key=None)

    @pytest.mark.unit
    def test_run_with_path(self, preview_samples_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = '{"text": "test transcription", "other_metadata": ["other", "meta", "data"]}'
        comp = RemoteWhisperTranscriber(api_key="whatever")

        with patch("haystack.preview.utils.requests_utils.requests") as mocked_requests:
            mocked_requests.request.return_value = mock_response

            result = comp.run(audio_files=[preview_samples_path / "audio" / "this is the content of the document.wav"])
            expected = Document(
                content="test transcription",
                metadata={
                    "audio_file": preview_samples_path / "audio" / "this is the content of the document.wav",
                    "other_metadata": ["other", "meta", "data"],
                },
            )
            assert result["documents"] == [expected]

    @pytest.mark.unit
    def test_run_with_str(self, preview_samples_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = '{"text": "test transcription", "other_metadata": ["other", "meta", "data"]}'
        comp = RemoteWhisperTranscriber(api_key="whatever")

        with patch("haystack.preview.utils.requests_utils.requests") as mocked_requests:
            mocked_requests.request.return_value = mock_response

            result = comp.run(
                audio_files=[
                    str((preview_samples_path / "audio" / "this is the content of the document.wav").absolute())
                ]
            )
            expected = Document(
                content="test transcription",
                metadata={
                    "audio_file": str(
                        (preview_samples_path / "audio" / "this is the content of the document.wav").absolute()
                    ),
                    "other_metadata": ["other", "meta", "data"],
                },
            )
            assert result["documents"] == [expected]

    @pytest.mark.unit
    def test_transcribe_with_stream(self, preview_samples_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = '{"text": "test transcription", "other_metadata": ["other", "meta", "data"]}'
        comp = RemoteWhisperTranscriber(api_key="whatever")

        with patch("haystack.preview.utils.requests_utils.requests") as mocked_requests:
            mocked_requests.request.return_value = mock_response

            with open(preview_samples_path / "audio" / "this is the content of the document.wav", "rb") as audio_stream:
                result = comp.transcribe(audio_files=[audio_stream])
                expected = Document(
                    content="test transcription",
                    metadata={"audio_file": "<<binary stream>>", "other_metadata": ["other", "meta", "data"]},
                )
                assert result == [expected]

    @pytest.mark.unit
    def test_api_transcription(self, preview_samples_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = '{"text": "test transcription", "other_metadata": ["other", "meta", "data"]}'
        comp = RemoteWhisperTranscriber(api_key="whatever")

        with patch("haystack.preview.utils.requests_utils.requests") as mocked_requests:
            mocked_requests.request.return_value = mock_response

            comp.run(audio_files=[preview_samples_path / "audio" / "this is the content of the document.wav"])
            requests_params = mocked_requests.request.call_args.kwargs
            requests_params.pop("files")
            assert requests_params == {
                "method": "post",
                "url": "https://api.openai.com/v1/audio/transcriptions",
                "data": {"model": "whisper-1"},
                "headers": {"Authorization": f"Bearer whatever"},
                "timeout": OPENAI_TIMEOUT,
            }

    @pytest.mark.unit
    def test_api_translation(self, preview_samples_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = '{"text": "test transcription", "other_metadata": ["other", "meta", "data"]}'
        comp = RemoteWhisperTranscriber(api_key="whatever")

        with patch("haystack.preview.utils.requests_utils.requests") as mocked_requests:
            mocked_requests.request.return_value = mock_response

            comp.run(
                audio_files=[preview_samples_path / "audio" / "this is the content of the document.wav"],
                whisper_params={"translate": True},
            )
            requests_params = mocked_requests.request.call_args.kwargs
            requests_params.pop("files")
            assert requests_params == {
                "method": "post",
                "url": "https://api.openai.com/v1/audio/translations",
                "data": {"model": "whisper-1"},
                "headers": {"Authorization": f"Bearer whatever"},
                "timeout": OPENAI_TIMEOUT,
            }

    @pytest.mark.unit
    @patch("haystack.preview.components.audio.whisper_remote.request_with_retry")
    def test_default_api_base(self, mock_request, preview_samples_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = '{"text": "test transcription", "other_metadata": ["other", "meta", "data"]}'
        mock_request.return_value = mock_response

        transcriber = RemoteWhisperTranscriber(api_key="just a test")
        assert transcriber.api_base == "https://api.openai.com/v1"

        transcriber.transcribe(audio_files=[preview_samples_path / "audio" / "this is the content of the document.wav"])
        assert mock_request.call_args.kwargs["url"] == "https://api.openai.com/v1/audio/transcriptions"

    @pytest.mark.unit
    @patch("haystack.preview.components.audio.whisper_remote.request_with_retry")
    def test_custom_api_base(self, mock_request, preview_samples_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = '{"text": "test transcription", "other_metadata": ["other", "meta", "data"]}'
        mock_request.return_value = mock_response

        transcriber = RemoteWhisperTranscriber(api_key="just a test", api_base="https://fake_api_base.com")
        assert transcriber.api_base == "https://fake_api_base.com"

        transcriber.transcribe(audio_files=[preview_samples_path / "audio" / "this is the content of the document.wav"])
        assert mock_request.call_args.kwargs["url"] == "https://fake_api_base.com/audio/transcriptions"
