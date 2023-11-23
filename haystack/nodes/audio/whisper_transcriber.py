import json
from typing import List, Optional, Dict, Any, Union, BinaryIO, Literal

import requests
from requests import PreparedRequest

from haystack import MultiLabel, Document
from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.nodes.base import BaseComponent
from haystack.utils.import_utils import is_whisper_available
from haystack.lazy_imports import LazyImport


with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_import:
    import torch


WhisperModel = Literal["tiny", "small", "medium", "large", "large-v2"]


class WhisperTranscriber(BaseComponent):
    """
    Transcribes audio files using OpenAI's Whisper. This class supports two underlying implementations:

    - API (default): Uses the OpenAI API and requires an API key. See the [OpenAI blog post](https://beta.openai.com/docs/api-reference/whisper for more details.
    - Local (requires installing Whisper): Uses the local installation
    of [Whisper](https://github.com/openai/whisper).

    To use Whisper locally, install it following the instructions on
    the Whisper [GitHub repo](https://github.com/openai/whisper) and omit the `api_key` parameter.

    To use the API implementation, provide an api_key. You can get one by signing up
    for an [OpenAI account](https://beta.openai.com/).

    For the supported audio formats, languages, and other parameters, see the
    [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
    [github repo](https://github.com/openai/whisper).
    """

    # If it's not a decision component, there is only one outgoing edge
    outgoing_edges = 1

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name_or_path: WhisperModel = "medium",
        device: Optional[Union[str, "torch.device"]] = None,
        api_base: str = "https://api.openai.com/v1",
    ) -> None:
        """
        Creates a WhisperTranscriber instance.

        :param api_key: OpenAI API key. If None, a local installation of Whisper is used.
        :param model_name_or_path: Name of the model to use. If using a local installation of Whisper, set this to one of the following values: "tiny", "small", "medium", "large", "large-v2". If using
        the API, set this value to: "whisper-1" (default).
        :param device: Device to use for inference. Only used if you're using a local
        installation of Whisper. If None, the device is automatically selected.
        :param api_base: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        """
        super().__init__()
        self.api_key = api_key
        self.api_base = api_base
        self.use_local_whisper = is_whisper_available() and self.api_key is None

        if self.use_local_whisper:
            import whisper

            self._model = whisper.load_model(model_name_or_path, device=device)
        else:
            if api_key is None:
                raise ValueError(
                    "Provide a valid api_key for OpenAI API. Alternatively, "
                    "install OpenAI Whisper (see [Whisper](https://github.com/openai/whisper) for more details)."
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
        Transcribe an audio file.

        :param audio_file: Path to the audio file or a binary file-like object.
        :param language: Language of the audio file. If None, the language is automatically detected.
        :param return_segments: If True, returns the transcription for each segment of the audio file. Supported with
        local installation of whisper only.
        :param translate: If True, translates the transcription to English.
        :return: A dictionary containing the transcription text and metadata like timings, segments etc.

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
                f"{self.api_base}/audio/transcriptions" if not translate else f"{self.api_base}/audio/translations"
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
        torch_import.check()

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

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ):  # type: ignore
        """
        Transcribe audio files.

        :param query: Ignored
        :param file_paths: List of paths to audio files.
        :param labels: Ignored
        :param documents: Ignored
        :param meta: Ignored
        :return: A dictionary containing a list of Document objects, one for each input file.

        """
        transcribed_documents: List[Document] = []
        if file_paths:
            for file_path in file_paths:
                transcription = self.transcribe(file_path)
                d = Document.from_dict(transcription, field_map={"text": "content"})
                transcribed_documents.append(d)

        output = {"documents": transcribed_documents}
        return output, "output_1"

    def run_batch(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):  # type: ignore
        """
        Transcribe audio files.

        :param queries: Ignored
        :param file_paths: List of paths to audio files.
        :param labels: Ignored
        :param documents: Ignored
        :param meta: Ignored
        :param params: Ignored
        :param debug: Ignored
        """
        if file_paths and isinstance(file_paths[0], list):
            all_files = []
            for files_list in file_paths:
                all_files += files_list
            return self.run(file_paths=all_files)
        return self.run(file_paths=file_paths)
