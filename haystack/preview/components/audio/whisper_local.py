from typing import List, Optional, Dict, Any, Union, BinaryIO, Literal, get_args, Sequence

import logging
from pathlib import Path

import torch
import whisper

from haystack.preview import component, Document, default_to_dict, default_from_dict


logger = logging.getLogger(__name__)
WhisperLocalModel = Literal["tiny", "small", "medium", "large", "large-v2"]


@component
class LocalWhisperTranscriber:
    """
    Transcribes audio files using OpenAI's Whisper's model on your local machine.

    For the supported audio formats, languages, and other parameters, see the
    [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
    [github repo](https://github.com/openai/whisper).
    """

    def __init__(
        self,
        model_name_or_path: WhisperLocalModel = "large",
        device: Optional[str] = None,
        whisper_params: Optional[Dict[str, Any]] = None,
    ):
        """
        :param model_name_or_path: Name of the model to use. Set it to one of the following values:
            - `tiny`
            - `small`
            - `medium`
            - `large`
            - `large-v2`
        :param device: Name of the torch device to use for inference. If None, CPU is used.
        """
        if model_name_or_path not in get_args(WhisperLocalModel):
            raise ValueError(
                f"Model name '{model_name_or_path}' not recognized. Choose one among: "
                f"{', '.join(get_args(WhisperLocalModel))}."
            )
        self.model_name = model_name_or_path
        self.whisper_params = whisper_params or {}
        self.device = torch.device(device) if device else torch.device("cpu")
        self._model = None

    def warm_up(self) -> None:
        """
        Loads the model.
        """
        if not self._model:
            self._model = whisper.load_model(self.model_name, device=self.device)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self, model_name_or_path=self.model_name, device=str(self.device), whisper_params=self.whisper_params
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalWhisperTranscriber":
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
        :returns: a list of Documents, one for each file. The content of the document is the transcription text,
            while the document's metadata contains all the other values returned by the Whisper model, such as the
            alignment data. Another key called `audio_file` contains the path to the audio file used for the
            transcription.
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
        Transcribe the given audio files. Returns the output of the model, a dictionary, for each input file.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files: a list of paths or binary streams to transcribe
        :returns: a list of transcriptions.
        """
        return_segments = kwargs.pop("return_segments", False)
        transcriptions = []
        for audio_file in audio_files:
            if isinstance(audio_file, (str, Path)):
                audio_file = open(audio_file, "rb")

            # mypy compains that _model is not guaranteed to be not None. It is: check self.warm_up()
            transcription = self._model.transcribe(audio_file.name, **kwargs)  # type: ignore
            if not return_segments:
                transcription.pop("segments", None)
            transcriptions.append(transcription)

        return transcriptions
