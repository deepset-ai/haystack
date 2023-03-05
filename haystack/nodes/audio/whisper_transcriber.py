import json
import tempfile

from typing import List, Optional, Dict, Any, Union, BinaryIO, Literal

import requests
import torch
from requests import PreparedRequest

from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.nodes.base import BaseComponent
from haystack.utils.import_utils import is_whisper_available


WhisperModel = Literal["tiny", "small", "medium", "large", "large-v2"]


class WhisperTranscriber(BaseComponent):
    # If it's not a decision component, there is only one outgoing edge
    outgoing_edges = 1

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name_or_path: WhisperModel = "medium",
        language: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self._language = language

        self.use_local_whisper = is_whisper_available() and self.api_key is None

        if self.use_local_whisper:
            import whisper

            self._model = whisper.load_model(model_name_or_path, device=device)
        else:
            if api_key is None:
                raise ValueError(
                    "Please provide a valid api_key for OpenAI API. Alternatively, "
                    "install OpenAI whisper (see https://github.com/openai/whisper for more details)."
                )

    def transcribe(
        self,
        audio: BinaryIO,
        language: Optional[str] = None,
        return_segments: bool = False,
        translate: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        transcript: Dict[str, Any] = {}

        new_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        language = language or self._language
        if language is not None:
            new_kwargs["language"] = language

        if self.use_local_whisper:
            new_kwargs["return_segments"] = return_segments
            transcript = self._invoke_local(audio, translate, **new_kwargs)
        elif self.api_key:
            transcript = self._invoke_api(audio, translate, **new_kwargs)
        return transcript

    def _invoke_api(self, audio_file: BinaryIO, translate: Optional[bool] = False, **kwargs) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        request = PreparedRequest()
        url: str = (
            "https://api.openai.com/v1/audio/transcriptions"
            if not translate
            else "https://api.openai.com/v1/audio/translations"
        )

        request.prepare(
            method="POST",
            url=url,
            headers=headers,
            data={"model": "whisper-1", **kwargs},
            files=[("file", (audio_file.name, audio_file, "application/octet-stream"))],
        )
        response = requests.post(url, data=request.body, headers=request.headers, timeout=600)

        if response.status_code != 200:
            openai_error: OpenAIError
            if response.status_code == 429:
                openai_error = OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
            else:
                openai_error = OpenAIError(
                    f"OpenAI returned an error.\n"
                    f"Status code: {response.status_code}\n"
                    f"Response body: {response.text}",
                    status_code=response.status_code,
                )
            raise openai_error

        return json.loads(response.content)

    def _invoke_local(self, audio_file: BinaryIO, translate: Optional[bool] = False, **kwargs) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile() as fl:
            fl.write(audio_file.read())
            fl.flush()

            return_segments = kwargs.pop("return_segments", None)
            kwargs["task"] = "translate" if translate else "transcribe"
            transcription = self._model.transcribe(fl.name, **kwargs)
            if not return_segments:
                transcription.pop("segments", None)

            return transcription

    def run(self, audio: BinaryIO, source_language: Optional[str] = None, return_segments: bool = False, translate: bool = False):  # type: ignore
        document = self.transcribe(
            audio=audio, source_language=source_language, return_segments=return_segments, translate=translate
        )

        output = {"documents": [document]}

        return output, "output_1"

    def run_batch(self, audios: List[BinaryIO], source_language: Optional[str] = None, return_segments: bool = False, translate: bool = False):  # type: ignore
        documents = []
        for audio in audios:
            document = self.transcribe(
                audio=audio, source_language=source_language, return_segments=return_segments, translate=translate
            )
            documents.append(document)

        output = {"documents": documents}

        return output, "output_1"
