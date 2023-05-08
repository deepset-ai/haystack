from typing import List, Optional, Dict, Any, Union, BinaryIO, Literal, Tuple

import os
import json
from pathlib import Path
from dataclasses import dataclass

import requests
import torch
import whisper
from tenacity import retry, wait_exponential, retry_if_not_result

from haystack.preview import component, Document
from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.preview.utils.import_utils import is_imported


OPENAI_TIMEOUT = float(os.environ.get("HAYSTACK_OPENAI_TIMEOUT_SEC", 30))


WhisperModel = Literal["tiny", "small", "medium", "large", "large-v2"]


@component
class WhisperTranscriber:
    """
    Transcribes audio files using OpenAI's Whisper. This class supports two underlying implementations:

    - API (default): Uses the OpenAI API and requires an API key. See the
      [OpenAI blog post](https://beta.openai.com/docs/api-reference/whisper for more details.

    - Local (requires installing Whisper): Uses the local installation
      of [Whisper](https://github.com/openai/whisper).

    To use Whisper locally, install it following the instructions on the Whisper
    [GitHub repo](https://github.com/openai/whisper) and omit the `api_key` parameter.

    To use the API implementation, provide an API key. You can get one by signing up for an
    [OpenAI account](https://beta.openai.com/).

    For the supported audio formats, languages, and other parameters, see the
    [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
    [github repo](https://github.com/openai/whisper).
    """

    @dataclass
    class Output:
        documents: List[Document]

    def __init__(self, api_key: Optional[str] = None, model_name_or_path: WhisperModel = "medium", device: str = None):
        """
        Transcribes a list of audio files into a list of Documents.

        :param api_key: OpenAI API key. If None, a local installation of Whisper is used.
        :param model_name_or_path: Name of the model to use. If using a local installation of Whisper, set this to one
            of the following values:
                - `tiny`
                - `small`
                - `medium`
                - `large`
                - `large-v2`
            If using the API, set this value to:
                - `whisper-1` (default)
        :param device: Device to use for inference. Only used if you're using a local installation of Whisper.
            If None, CPU is used.
        """
        self.api_key = api_key
        self.model_name = model_name_or_path
        self.device = torch.device(device) or torch.device("cpu")
        self.use_local_whisper = is_imported("whisper") and self.api_key is None

        self._model = None
        if not self.use_local_whisper and api_key is None:
            raise ValueError(
                "Provide a valid api_key for OpenAI API. Alternatively, install OpenAI Whisper (see "
                "[Whisper](https://github.com/openai/whisper) for more details)."
            )

    def warm_up(self):
        """
        If we're using a local model, load it here.
        """
        if self.use_local_whisper and not self._model:
            self._model = whisper.load_model(self.model_name, device=self.device)

    def run(self, audios: List[Path], whisper_params: Dict[str, Any]) -> Output:
        self.warm_up()
        documents = self.transcribe_to_documents(audios, **whisper_params)
        return WhisperTranscriber.Output(documents)

    def transcribe_to_documents(self, audio_files: List[Union[str, BinaryIO]], **kwargs) -> Dict[str, Any]:
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

    def transcribe(self, audio_files: List[Union[str, BinaryIO]], **kwargs) -> Dict[str, Any]:
        """
        Transcribe the given audio files. Returns a list of strings.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files: a list of paths or binary streams to transcribe
        :returns: a list of transcriptions.
        """
        self.warm_up()
        transcriptions = []
        for audio_file in audio_files:
            if isinstance(audio_file, (str, Path)):
                audio_file = open(audio_file, "rb")

            if self.use_local_whisper:
                transcription = self._transcribe_locally(audio_file, **kwargs)
            else:
                transcription = self._transcribe_with_api(audio_file, **kwargs)

            transcriptions.append(transcription)
        return transcriptions

    @retry(retry=retry_if_not_result(bool), wait=wait_exponential(min=1, max=10))
    def _transcribe_with_api(self, audio_file: BinaryIO, **kwargs) -> Dict[str, Any]:
        """
        Calls a remote Whisper model through OpenAI Whisper API.
        """
        translate = kwargs.pop("translate", False)

        response = requests.post(
            url=f"https://api.openai.com/v1/audio/{'translations' if translate else 'transcriptions'}",
            data={"model": "whisper-1", **kwargs},
            headers={"Authorization": f"Bearer {self.api_key}"},
            files=[("file", (audio_file.name, audio_file, "application/octet-stream"))],
            timeout=600,
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

    def _transcribe_locally(self, audio_file: BinaryIO, **kwargs) -> Dict[str, Any]:
        """
        Calls a local Whisper model.
        """
        return_segments = kwargs.pop("return_segments", None)
        transcription = self._model.transcribe(audio_file.name, **kwargs)
        if not return_segments:
            transcription.pop("segments", None)
        return transcription
