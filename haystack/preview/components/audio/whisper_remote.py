from typing import List, Optional, Dict, Any, Union, BinaryIO, Literal, get_args, Sequence

import os
import json
import logging
from pathlib import Path
import openai


from haystack.preview.utils import request_with_retry
from haystack.preview import component, Document, default_to_dict

logger = logging.getLogger(__name__)

OPENAI_TIMEOUT = float(os.environ.get("HAYSTACK_OPENAI_TIMEOUT_SEC", 600))

WhisperRemoteModel = Literal["whisper-1"]

@component
class RemoteWhisperTranscriber:
    """
    Transcribes audio files using OpenAI's Whisper using OpenAI API.

    For the supported audio formats, languages, and other parameters, see the
    [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text).

    :param api_key: OpenAI API key.
    :param whisper_params: Optional parameters for Whisper API.
    """

    def __init__(
        self,
        api_key: str,
        model_name: WhisperRemoteModel = "whisper-1",
        whisper_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the RemoteWhisperTranscriber.

        :param api_key: OpenAI API key.
        :param whisper_params: Optional parameters for the Whisper API.
        :param model_name: Name of the model to use. It now accepts only `whisper-1`.
        """

        if model_name not in get_args(WhisperRemoteModel):
            raise ValueError(
                f"Model name not recognized. Choose one among: " f"{', '.join(get_args(WhisperRemoteModel))}."
            )
        if not api_key:
            raise ValueError("API key is None.")
        
        self.model_name = model_name
        self.api_key = api_key
        self.whisper_params = whisper_params or {}

    @component.output_types(documents=List[Document])
    def run(self, audio_files: List[Path], whisper_params: Optional[Dict[str, Any]] = None):
        """
        Transcribe the audio files into a list of Documents.

        :param audio_files: a list of paths or binary streams to transcribe
        :returns: a list of Documents, one for each file. The content of the document is the transcription text,
            while the document's metadata contains all the other values returned by the Whisper model, such as the
            alignment data. Another key called `audio_file` contains the path to the audio file used for the
            transcription.
        """
        if whisper_params is None:
            whisper_params = self.whisper_params

        documents = self.transcribe(audio_files, **whisper_params)
        return {"documents": documents}

    def transcribe(self, audio_files: List[Path], **kwargs) -> List[Document]:
        """
        Transcribe the audio files into a list of Documents.

        :param audio_files: A list of paths to audio files.
        :returns: a list of transcriptions.
        """
        transcriptions = self._raw_transcribe(audio_files, **kwargs)
        documents = []
        for audio, transcript in zip(audio_files, transcriptions):
            content = transcript.pop("text")
            doc = Document(text=content, metadata={"audio_file": audio, **transcript})
            documents.append(doc)
        return documents

    def _raw_transcribe(self, audio_files: List[Path], **kwargs) -> List[Dict[str, Any]]:
        """
        Transcribe the given audio files. Returns a list of strings.

        :param audio_files: A list of paths to audio files.
        :param kwargs: Additional parameters for the Whisper API.
        """
        transcriptions = []
        for audio_file in audio_files:
            with audio_file.open("rb") as f:
                response = openai.Audio.transcribe(
                    f.read(),
                    model=self.model_name,
                    language="en-US",
                    **kwargs
                )
                transcription = json.loads(response)
                transcriptions.append(transcription)
        return transcriptions

    def to_dict(self) -> Dict[str, Any]:
        """
        This method overrides the default serializer in order to avoid leaking the `api_key` value passed
        to the constructor.
        """
        return default_to_dict(
            self, model_name=self.model_name, whisper_params=self.whisper_params
        )