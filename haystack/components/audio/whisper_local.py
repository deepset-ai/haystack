# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, get_args

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice

with LazyImport("Run 'pip install \"openai-whisper>=20231106\"' to install whisper.") as whisper_import:
    import whisper


logger = logging.getLogger(__name__)
WhisperLocalModel = Literal["tiny", "small", "medium", "large", "large-v2"]


@component
class LocalWhisperTranscriber:
    """
    Transcribes audio files using OpenAI's Whisper model in your local machine.

    For the supported audio formats, languages, and other parameters, see the
    [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
    [github repository](https://github.com/openai/whisper).

    Usage example:
    ```python
    from haystack.components.audio import LocalWhisperTranscriber

    whisper = LocalWhisperTranscriber(model="small")
    whisper.warm_up()
    transcription = whisper.run(audio_files=["path/to/audio/file"])
    ```
    """

    def __init__(
        self,
        model: WhisperLocalModel = "large",
        device: Optional[ComponentDevice] = None,
        whisper_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of the LocalWhisperTranscriber component.

        :param model:
            Name of the model to use. Set it to one of the following values:
        :type model:
            Literal["tiny", "small", "medium", "large", "large-v2"]
        :param device:
            The device on which the model is loaded. If `None`, the default device is automatically selected.
        """
        whisper_import.check()
        if model not in get_args(WhisperLocalModel):
            raise ValueError(
                f"Model name '{model}' not recognized. Choose one among: " f"{', '.join(get_args(WhisperLocalModel))}."
            )
        self.model = model
        self.whisper_params = whisper_params or {}
        self.device = ComponentDevice.resolve_device(device)
        self._model = None

    def warm_up(self) -> None:
        """
        Loads the model in memory.
        """
        if not self._model:
            self._model = whisper.load_model(self.model, device=self.device.to_torch())

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, model=self.model, device=self.device.to_dict(), whisper_params=self.whisper_params)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalWhisperTranscriber":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data["init_parameters"]
        if init_params["device"] is not None:
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]], whisper_params: Optional[Dict[str, Any]] = None):
        """
        Transcribes the audio files into a list of Documents, one for each input file.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files:
            A list of paths or binary streams to transcribe.

        :returns: A dictionary with the following keys:
            - `documents`: A list of Documents, one for each file. The content of the document is the transcription text,
                while the document's metadata contains the values returned by the Whisper model, such as the
                alignment data and the path to the audio file used for the transcription.
        """
        if self._model is None:
            raise RuntimeError(
                "The component LocalWhisperTranscriber was not warmed up. Run 'warm_up()' before calling 'run()'."
            )

        if whisper_params is None:
            whisper_params = self.whisper_params

        documents = self.transcribe(sources, **whisper_params)
        return {"documents": documents}

    def transcribe(self, sources: List[Union[str, Path, ByteStream]], **kwargs) -> List[Document]:
        """
        Transcribes the audio files into a list of Documents, one for each input file.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files:
            A list of paths or binary streams to transcribe.
        :returns:
            A list of Documents, one for each file.
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
        Transcribes the given audio files. Returns the output of the model, a dictionary, for each input file.

        For the supported audio formats, languages, and other parameters, see the
        [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
        [github repo](https://github.com/openai/whisper).

        :param audio_files:
            A list of paths or binary streams to transcribe.
        :returns:
            A dictionary mapping 'file_path' to 'transcription'.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded, please run 'warm_up()' before calling 'run()'")

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
