from typing import List, Optional, Dict, Any, Union, BinaryIO, Literal, get_args, Sequence

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass

import requests
from tenacity import retry, wait_exponential, retry_if_not_result

from haystack.preview import component, Document
from haystack.errors import OpenAIError, OpenAIRateLimitError


logger = logging.getLogger(__name__)


OPENAI_TIMEOUT = float(os.environ.get("HAYSTACK_OPENAI_TIMEOUT_SEC", 600))


WhisperRemoteModel = Literal["whisper-1"]


@component
class RemoteWhisperTranscriber:
    """
    Transcribes audio files using OpenAI's Whisper using OpenAI API. Requires an API key. See the
      [OpenAI blog post](https://beta.openai.com/docs/api-reference/whisper for more details.
      You can get one by signing up for an [OpenAI account](https://beta.openai.com/).

    For the supported audio formats, languages, and other parameters, see the
    [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text)
    """

    @dataclass
    class Output:
        documents: List[Document]

    def __init__(self, model_name_or_path: WhisperRemoteModel = "whisper-1", api_key: Optional[str] = None):
        """
        Transcribes a list of audio files into a list of Documents.

        :param api_key: OpenAI API key.
        :param model_name_or_path: Name of the model to use. It now accepts only `whisper-1`.
        """
        if model_name_or_path not in get_args(WhisperRemoteModel):
            raise ValueError(
                f"Model name not recognized. Choose one among: " f"{', '.join(get_args(WhisperRemoteModel))}."
            )
        if not api_key:
            raise ValueError("Provide a valid API key for OpenAI API.")

        self.api_key = api_key
        self.model_name = model_name_or_path

    def run(self, audios: List[Path], whisper_params: Dict[str, Any]) -> Output:
        documents = self.transcribe_to_documents(audios, **whisper_params)
        return RemoteWhisperTranscriber.Output(documents)

    def transcribe_to_documents(self, audio_files: Sequence[Union[str, Path, BinaryIO]], **kwargs) -> List[Document]:
        """
        Transcribe the given audio files. Returns a list of Documents.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files: a list of paths or binary streams to transcribe
        :returns: a list of transcriptions.
        """
        transcriptions = self.transcribe(audio_files=audio_files, **kwargs)
        return [
            Document(content=transcript.pop("text"), metadata={"audio_file": audio, **transcript})
            for audio, transcript in zip(audio_files, transcriptions)
        ]

    def transcribe(self, audio_files: Sequence[Union[str, Path, BinaryIO]], **kwargs) -> List[Dict[str, Any]]:
        """
        Transcribe the given audio files. Returns a list of strings.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files: a list of paths or binary streams to transcribe
        :returns: a list of transcriptions.
        """
        transcriptions = []
        for audio_file in audio_files:
            if isinstance(audio_file, (str, Path)):
                audio_file = open(audio_file, "rb")
                transcription = self._transcribe(audio_file, **kwargs)

            transcriptions.append(transcription)
        return transcriptions

    @retry(retry=retry_if_not_result(bool), wait=wait_exponential(min=1, max=10))
    def _transcribe(self, audio_file: BinaryIO, **kwargs) -> Dict[str, Any]:
        """
        Calls a remote Whisper model through OpenAI Whisper API.
        """
        translate = kwargs.pop("translate", False)

        response = requests.post(
            url=f"https://api.openai.com/v1/audio/{'translations' if translate else 'transcriptions'}",
            data={"model": "whisper-1", **kwargs},
            headers={"Authorization": f"Bearer {self.api_key}"},
            files=[("file", (audio_file.name, audio_file, "application/octet-stream"))],
            timeout=OPENAI_TIMEOUT,
        )

        if response.status_code != 200:
            if response.status_code == 429:
                raise OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
            raise OpenAIError(
                f"OpenAI returned an error.\n"
                f"Status code: {response.status_code}\n"
                f"Response body: {response.text}",
                status_code=response.status_code,
            )

        return json.loads(response.content)
