import json

from typing import List, Optional, Dict, Any, Union, BinaryIO, Literal

import requests
import torch
from requests import PreparedRequest

from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.nodes.base import BaseComponent
from haystack.utils.import_utils import is_whisper_available


WhisperModel = Literal["tiny", "small", "medium", "large", "large-v2"]


class WhisperTranscriber(BaseComponent):
    """
    Transcribes audio files using OpenAI's Whisper. This class supports two underlying implementations:

    - API (default): Uses the OpenAI API and requires an API key. See blog
    [post](https://beta.openai.com/docs/api-reference/whisper for more details.) for more details.
    - Local (requires installation of whisper): Uses the local installation
    of [whisper](https://github.com/openai/whisper).

    If you are using local installation of whisper, install whisper following the instructions available on
    the Whisper [github repo](https://github.com/openai/whisper) and omit the api_key parameter.

    If you are using the API implementation, you need to provide an api_key. You can get one by signing up
    for an OpenAI account [here](https://beta.openai.com/).
    """

    # If it's not a decision component, there is only one outgoing edge
    outgoing_edges = 1

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name_or_path: WhisperModel = "medium",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Creates a WhisperTranscriber instance.

        :param api_key: OpenAI API key. If None, local installation of whisper is used.
        :param model_name_or_path: Name of the model to use. If using local installation of whisper, this
        value has to be one of the following: "tiny", "small", "medium", "large", "large-v2". If using
        the API, this value has to be "whisper-1" (default).
        :param device: Device to use for inference. This parameter is only used if you are using local
        installation of whisper. If None, the device is automatically selected.
        """
        super().__init__()
        self.api_key = api_key

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
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        return_segments: bool = False,
        translate: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file.

        :param audio_file: Path to audio file or a binary file-like object.
        :param language: Language of the audio file. If None, the language is automatically detected.
        :param return_segments: If True, returns the transcription for each segment of the audio file.
        :param translate: If True, translates the transcription to English.

        """
        transcript: Dict[str, Any] = {}

        new_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if language is not None:
            new_kwargs["language"] = language

        if self.use_local_whisper:
            new_kwargs["return_segments"] = return_segments
            transcript = self._invoke_local(audio_file, translate, **new_kwargs)
        elif self.api_key:
            transcript = self._invoke_api(audio_file, translate, **new_kwargs)
        return transcript

    def _invoke_api(
        self, audio_file: Union[str, BinaryIO], translate: Optional[bool] = False, **kwargs
    ) -> Dict[str, Any]:
        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                return self._invoke_api(f, translate, **kwargs)
        else:
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

    def _invoke_local(
        self, audio_file: Union[str, BinaryIO], translate: Optional[bool] = False, **kwargs
    ) -> Dict[str, Any]:
        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                return self._invoke_local(f, translate, **kwargs)
        else:
            return_segments = kwargs.pop("return_segments", None)
            kwargs["task"] = "translate" if translate else "transcribe"
            transcription = self._model.transcribe(audio_file.name, **kwargs)
            if not return_segments:
                transcription.pop("segments", None)

            return transcription

    def run(self, audio_file: Union[str, BinaryIO], language: Optional[str] = None, return_segments: bool = False, translate: bool = False):  # type: ignore
        """
        Transcribe audio file.

        :param audio_file: Path to audio file or a binary file-like object.
        :param language: Language of the audio file. If None, the language is automatically detected.
        :param return_segments: If True, returns the transcription for each segment of the audio file.
        :param translate: If True, translates the transcription to English.
        """
        document = self.transcribe(audio_file, language, return_segments, translate)

        output = {"documents": [document]}

        return output, "output_1"

    def run_batch(self, audio_files: List[Union[str, BinaryIO]], language: Optional[str] = None, return_segments: bool = False, translate: bool = False):  # type: ignore
        """
        Transcribe audio files.

        :param audio_files: List of paths to audio files or binary file-like objects.
        :param language: Language of the audio files. If None, the language is automatically detected.
        :param return_segments: If True, returns the transcription for each segment of the audio files.
        :param translate: If True, translates the transcription to English.
        """
        documents = []
        for audio in audio_files:
            document = self.transcribe(audio, language, return_segments, translate)
            documents.append(document)

        output = {"documents": documents}

        return output, "output_1"
