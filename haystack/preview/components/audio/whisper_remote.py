from typing import List, Optional, Dict, Any, Union, BinaryIO, Literal, get_args, Sequence

import os
import json
import logging
from pathlib import Path

from haystack.preview.utils import request_with_retry
from haystack.preview import component, Document, default_to_dict, default_from_dict

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

    def __init__(
        self,
        api_key: str,
        model_name: WhisperRemoteModel = "whisper-1",
        api_base: str = "https://api.openai.com/v1",
        whisper_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Transcribes a list of audio files into a list of Documents.

        :param api_key: OpenAI API key.
        :param model_name: Name of the model to use. It now accepts only `whisper-1`.
        :param api_base: OpenAI base URL, defaults to `"https://api.openai.com/v1"`.
        """
        if model_name not in get_args(WhisperRemoteModel):
            raise ValueError(
                f"Model name not recognized. Choose one among: " f"{', '.join(get_args(WhisperRemoteModel))}."
            )
        if not api_key:
            raise ValueError("API key is None.")

        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.whisper_params = whisper_params or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            model_name=self.model_name,
            api_key=self.api_key,
            api_base=self.api_base,
            whisper_params=self.whisper_params,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemoteWhisperTranscriber":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, audio_files: List[Path], whisper_params: Optional[Dict[str, Any]] = None):
        """
        Transcribe the audio files into a list of Documents, one for each input file.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

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

    def transcribe(self, audio_files: Sequence[Union[str, Path, BinaryIO]], **kwargs) -> List[Document]:
        """
        Transcribe the audio files into a list of Documents, one for each input file.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files: a list of paths or binary streams to transcribe
        :returns: a list of transcriptions.
        """
        transcriptions = self._raw_transcribe(audio_files=audio_files, **kwargs)
        documents = []
        for audio, transcript in zip(audio_files, transcriptions):
            content = transcript.pop("text")
            if not isinstance(audio, (str, Path)):
                audio = "<<binary stream>>"
            doc = Document(content=content, metadata={"audio_file": audio, **transcript})
            documents.append(doc)
        return documents

    def _raw_transcribe(self, audio_files: Sequence[Union[str, Path, BinaryIO]], **kwargs) -> List[Dict[str, Any]]:
        """
        Transcribe the given audio files. Returns a list of strings.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files: a list of paths or binary streams to transcribe.
        :param kwargs: any other parameters that Whisper API can understand.
        :returns: a list of transcriptions as they are produced by the Whisper API (JSON).
        """
        translate = kwargs.pop("translate", False)
        url = f"{self.api_base}/audio/{'translations' if translate else 'transcriptions'}"
        data = {"model": self.model_name, **kwargs}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        transcriptions = []
        for audio_file in audio_files:
            if isinstance(audio_file, (str, Path)):
                audio_file = open(audio_file, "rb")

            request_files = ("file", (audio_file.name, audio_file, "application/octet-stream"))
            response = request_with_retry(
                method="post", url=url, data=data, headers=headers, files=[request_files], timeout=OPENAI_TIMEOUT
            )
            transcription = json.loads(response.content)

            transcriptions.append(transcription)
        return transcriptions
