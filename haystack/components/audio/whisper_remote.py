# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


@component
class RemoteWhisperTranscriber:
    """
    Transcribes audio files using the OpenAI's Whisper API.

    The component requires an OpenAI API key, see the
    [OpenAI documentation](https://platform.openai.com/docs/api-reference/authentication) for more details.
    For the supported audio formats, languages, and other parameters, see the
    [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text).

    ### Usage example

    ```python
    from haystack.components.audio import RemoteWhisperTranscriber

    whisper = RemoteWhisperTranscriber(api_key=Secret.from_token("<your-api-key>"), model="tiny")
    transcription = whisper.run(sources=["path/to/audio/file"])
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "whisper-1",
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs,
    ):
        """
        Creates an instance of the RemoteWhisperTranscriber component.

        :param api_key:
            OpenAI API key.
            You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
            during initialization.
        :param model:
            Name of the model to use. Currently accepts only `whisper-1`.
        :param organization:
            Your OpenAI organization ID. See OpenAI's documentation on
            [Setting Up Your Organization](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization).
        :param api_base:
            An optional URL to use as the API base. For details, see the
            OpenAI [documentation](https://platform.openai.com/docs/api-reference/audio).
        :param kwargs:
            Other optional parameters for the model. These are sent directly to the OpenAI
            endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/audio) for more details.
            Some of the supported parameters are:
            - `language`: The language of the input audio.
              Provide the input language in ISO-639-1 format
              to improve transcription accuracy and latency.
            - `prompt`: An optional text to guide the model's
              style or continue a previous audio segment.
              The prompt should match the audio language.
            - `response_format`: The format of the transcript
              output. This component only supports `json`.
            - `temperature`: The sampling temperature, between 0
            and 1. Higher values like 0.8 make the output more
            random, while lower values like 0.2 make it more
            focused and deterministic. If set to 0, the model
            uses log probability to automatically increase the
            temperature until certain thresholds are hit.
        """

        self.organization = organization
        self.model = model
        self.api_base_url = api_base_url
        self.api_key = api_key

        # Only response_format = "json" is supported
        whisper_params = kwargs
        response_format = whisper_params.get("response_format", "json")
        if response_format != "json":
            logger.warning(
                "RemoteWhisperTranscriber only supports 'response_format: json'. This parameter will be overwritten."
            )
        whisper_params["response_format"] = "json"
        self.whisper_params = whisper_params
        self.client = OpenAI(api_key=api_key.resolve_value(), organization=organization, base_url=api_base_url)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            organization=self.organization,
            api_base_url=self.api_base_url,
            **self.whisper_params,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemoteWhisperTranscriber":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]):
        """
        Transcribes the list of audio files into a list of documents.

        :param sources:
            A list of file paths or `ByteStream` objects containing the audio files to transcribe.

        :returns: A dictionary with the following keys:
            - `documents`: A list of documents, one document for each file.
                The content of each document is the transcribed text.
        """
        documents = []

        for source in sources:
            if not isinstance(source, ByteStream):
                path = source
                source = ByteStream.from_file_path(Path(source))
                source.meta["file_path"] = path

            file = io.BytesIO(source.data)
            file.name = str(source.meta["file_path"]) if "file_path" in source.meta else "__fallback__.wav"

            content = self.client.audio.transcriptions.create(file=file, model=self.model, **self.whisper_params)
            doc = Document(content=content.text, meta=source.meta)
            documents.append(doc)

        return {"documents": documents}
