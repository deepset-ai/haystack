from typing import List, Optional, Dict, Any, Union, BinaryIO

import os
import logging
from pathlib import Path

import openai

from haystack.preview import component, Document, default_to_dict, default_from_dict


logger = logging.getLogger(__name__)


API_BASE_URL = "https://api.openai.com/v1"


@component
class RemoteWhisperTranscriber:
    """
    Transcribes audio files using OpenAI's Whisper using OpenAI API. Requires an API key. See the
    [OpenAI blog post](https://beta.openai.com/docs/api-reference/whisper for more details.
    You can get one by signing up for an [OpenAI account](https://beta.openai.com/).

    For the supported audio formats, languages, and other parameters, see the
    [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "whisper-1",
        organization: Optional[str] = None,
        api_base_url: str = API_BASE_URL,
        **kwargs,
    ):
        """
        Transcribes a list of audio files into a list of Documents.

        :param api_key: OpenAI API key.
        :param model_name: Name of the model to use. It now accepts only `whisper-1`.
        :param organization: The OpenAI-Organization ID, defaults to `None`. For more details, see OpenAI
        [documentation](https://platform.openai.com/docs/api-reference/requesting-organization).
        :param api_base: OpenAI base URL, defaults to `"https://api.openai.com/v1"`.
        :param kwargs: Other parameters to use for the model. These parameters are all sent directly to the OpenAI
            endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/audio) for more details.
            Some of the supported parameters:
            - `language`: The language of the input audio.
            Supplying the input language in ISO-639-1 format
              will improve accuracy and latency.
            - `prompt`: An optional text to guide the model's
              style or continue a previous audio segment.
              The prompt should match the audio language.
            - `response_format`: The format of the transcript
              output, in one of these options: json, text, srt,
               verbose_json, or vtt. Defaults to "json". Currently only "json" is supported.
            - `temperature`: The sampling temperature, between 0
            and 1. Higher values like 0.8 will make the output more
            random, while lower values like 0.2 will make it more
            focused and deterministic. If set to 0, the model will
            use log probability to automatically increase the
            temperature until certain thresholds are hit.
        """

        if api_key is None:
            try:
                api_key = os.environ["OPENAI_API_KEY"]
            except KeyError as e:
                raise ValueError(
                    "RemoteWhisperTranscriber expects an OpenAI API key. "
                    "Set the OPENAI_API_KEY environment variable (recommended) or pass it explicitly."
                ) from e

        self.organization = organization
        self.model_name = model_name
        self.api_base_url = api_base_url

        # Only response_format = "json" is supported
        whisper_params = kwargs
        whisper_params["response_format"] = "json"
        self.whisper_params = whisper_params

        openai.api_key = api_key
        if organization is not None:
            openai.organization = organization

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        This method overrides the default serializer in order to
        avoid leaking the `api_key` value passed to the constructor.
        """
        return default_to_dict(
            self,
            model_name=self.model_name,
            organization=self.organization,
            api_base_url=self.api_base_url,
            **self.whisper_params,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemoteWhisperTranscriber":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, audio_files: List[Union[str, Path, BinaryIO]]):
        """
        Transcribe the audio files into a list of Documents, one for each input file.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files: a list of paths or binary streams to transcribe
        :returns: a list of Documents, one for each file. The content of the
        document is the transcription text, while the document's metadata
        contains a key called `audio_file`, which contains the path to the
        audio file used for the transcription.
        """
        documents = []

        for audio_file in audio_files:
            if isinstance(audio_file, (str, Path)):
                if isinstance(audio_file, str):
                    file_name = audio_file
                else:
                    file_name = str(audio_file.absolute())
                with open(audio_file, "rb") as file:
                    content = openai.Audio.transcribe(file=file, model=self.model_name, **self.whisper_params)
            else:
                file_name = "<<binary stream>>"
                content = openai.Audio.transcribe(file=audio_file, model=self.model_name, **self.whisper_params)

            doc = Document(text=content["text"], metadata={"audio_file": file_name})
            documents.append(doc)

        return {"documents": documents}
