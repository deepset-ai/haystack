from typing import List, Optional, Dict, Any, Union, BinaryIO, Literal, get_args, Sequence

import logging
from pathlib import Path
from dataclasses import dataclass

import torch
import haystack.preview.components.audio.whisper_local as whisper_local

from haystack.preview import component, Document
from haystack import is_imported


logger = logging.getLogger(__name__)
WhisperLocalModel = Literal["tiny", "small", "medium", "large", "large-v2"]


@component
class LocalWhisperTranscriber:
    """
    Transcribes audio files using OpenAI's Whisper's on your local machine.

    To use Whisper locally, install it following the instructions on the Whisper
    [GitHub repo](https://github.com/openai/whisper) and omit the `api_key` parameter.

    For the supported audio formats, languages, and other parameters, see the
    [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
    [github repo](https://github.com/openai/whisper).
    """

    @dataclass
    class Output:
        documents: List[Document]

    def __init__(self, model_name_or_path: WhisperLocalModel = "large", device: Optional[str] = None):
        """
        Transcribes a list of audio files into a list of Documents.

        :param model_name_or_path: Name of the model to use. If using a local installation of Whisper, set this to one
            of the following values:
                - `tiny`
                - `small`
                - `medium`
                - `large`
                - `large-v2`
        :param device: Device to use for inference. Only used if you're using a local installation of Whisper.
            If None, CPU is used.
        """
        if model_name_or_path not in get_args(WhisperLocalModel):
            raise ValueError(
                f"Model name not recognized. Choose one among: " f"{', '.join(get_args(WhisperLocalModel))}."
            )

        if not is_imported("whisper"):
            raise ValueError(
                "To use a local Whisper model, install Haystack's audio extras as `pip install farm-haystack[audio]` "
                "or install Whisper yourself with `pip install openai-whisper`. You will need ffmpeg on your system "
                "in either case, see: https://github.com/openai/whisper."
            )

        self.model_name = model_name_or_path
        self.device = torch.device(device) if device else torch.device("cpu")
        self._model = None

    def warm_up(self):
        """
        If we're using a local model, load it here.
        """
        if not self._model:
            self._model = whisper_local.load_model(self.model_name, device=self.device)

    def run(self, audios: List[Path], whisper_params: Dict[str, Any]) -> Output:
        documents = self.transcribe_to_documents(audios, **whisper_params)
        return LocalWhisperTranscriber.Output(documents)

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
        self.warm_up()
        if not self._model:
            raise ValueError("WhisperTranscriber._transcribe_locally() can't work without a local model.")

        return_segments = kwargs.pop("return_segments", None)
        transcriptions = []
        for audio_file in audio_files:
            if isinstance(audio_file, (str, Path)):
                audio_file = open(audio_file, "rb")

            transcription = self._model.transcribe(audio_file.name, **kwargs)
            if not return_segments:
                transcription.pop("segments", None)

            transcriptions.append(transcription)
        return transcriptions
