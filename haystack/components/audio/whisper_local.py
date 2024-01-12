from typing import List, Optional, Dict, Any, Union, Literal, get_args

import logging
import tempfile
from pathlib import Path

from haystack import component, Document, default_to_dict, ComponentError
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport(
    "Run 'pip install transformers[torch]' to install torch and "
    "'pip install \"openai-whisper>=20231106\"' to install whisper."
) as whisper_import:
    import torch
    import whisper


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
        model: WhisperLocalModel = "large",
        device: Optional[str] = None,
        whisper_params: Optional[Dict[str, Any]] = None,
    ):
        """
        :param model: Name of the model to use. Set it to one of the following values:
        :type model: Literal["tiny", "small", "medium", "large", "large-v2"]
        :param device: Name of the torch device to use for inference. If None, CPU is used.
        :type device: Optional[str]
        """
        whisper_import.check()
        if model not in get_args(WhisperLocalModel):
            raise ValueError(
                f"Model name '{model}' not recognized. Choose one among: " f"{', '.join(get_args(WhisperLocalModel))}."
            )
        self.model = model
        self.whisper_params = whisper_params or {}
        self.device = torch.device(device) if device else torch.device("cpu")
        self._model = None

    def warm_up(self) -> None:
        """
        Loads the model.
        """
        if not self._model:
            self._model = whisper.load_model(self.model, device=self.device)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, model=self.model, device=str(self.device), whisper_params=self.whisper_params)

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]], whisper_params: Optional[Dict[str, Any]] = None):
        """
        Transcribe the audio files into a list of Documents, one for each input file.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files: A list of paths or binary streams to transcribe.
        :returns: A list of Documents, one for each file. The content of the document is the transcription text,
            while the document's metadata contains all the other values returned by the Whisper model, such as the
            alignment data. Another key called `audio_file` contains the path to the audio file used for the
            transcription.
        """
        if self._model is None:
            raise ComponentError("The component was not warmed up. Run 'warm_up()' before calling 'run()'.")

        if whisper_params is None:
            whisper_params = self.whisper_params

        documents = self.transcribe(sources, **whisper_params)
        return {"documents": documents}

    def transcribe(self, sources: List[Union[str, Path, ByteStream]], **kwargs) -> List[Document]:
        """
        Transcribe the audio files into a list of Documents, one for each input file.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files: A list of paths or binary streams to transcribe.
        :returns: A list of Documents, one for each file. The content of the document is the transcription text,
            while the document's metadata contains all the other values returned by the Whisper model, such as the
            alignment data. Another key called `audio_file` contains the path to the audio file used for the
            transcription.
        """
        transcriptions = self._raw_transcribe(sources, **kwargs)
        documents = []
        for path, transcript in transcriptions.items():
            content = transcript.pop("text")
            doc = Document(content=content, meta={"audio_file": path, **transcript})
            documents.append(doc)
        return documents

    def _raw_transcribe(self, sources: List[Union[str, Path, ByteStream]], **kwargs) -> Dict[Path, Any]:
        """
        Transcribe the given audio files. Returns the output of the model, a dictionary, for each input file.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files: A list of paths or binary streams to transcribe.
        :returns: A dictionary of  file_path -> transcription.
        """
        if self._model is None:
            raise ComponentError("Model is not loaded, please run 'warm_up()' before calling 'run()'")

        return_segments = kwargs.pop("return_segments", False)
        transcriptions: Dict[Path, Any] = {}
        for source in sources:
            if not isinstance(source, ByteStream):
                path = Path(source)
                source = ByteStream.from_file_path(path)
                source.meta["file_path"] = path
            else:
                # If we received a ByteStream instance that doesn't have the "file_path" metadata set,
                # we dump the bytes into a temporary file.
                path = source.meta.get("file_path")
                if path is None:
                    fp = tempfile.NamedTemporaryFile(delete=False)
                    path = Path(fp.name)
                    source.to_file(path)
                    source.meta["file_path"] = path

            transcription = self._model.transcribe(str(path), **kwargs)
            if not return_segments:
                transcription.pop("segments", None)
            transcriptions[path] = transcription

        return transcriptions
