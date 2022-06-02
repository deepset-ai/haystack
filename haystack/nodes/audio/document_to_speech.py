import logging
from typing import Union, Callable, Optional, List, Dict, Tuple, Any

import hashlib
from pathlib import Path

from haystack.nodes import BaseComponent
from haystack.schema import Document, AudioDocument, GeneratedAudioDocument
from haystack.nodes.audio._text_to_speech import TextToSpeech


class DocumentToSpeech(BaseComponent):
    """
    This node converts text-based Documents into AudioDocuments, where the content is
    read out into an audio file.
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "espnet/kan-bayashi_ljspeech_vits",
        generated_audio_dir: Path = Path(__file__).parent / "generated_audio_documents",
        audio_format: str = "wav",
        subtype: str = "PCM_16",
        audio_naming_function: Callable = lambda text: hashlib.md5(text.encode("utf-8")).hexdigest(),
        transformers_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Convert an input Document into an audio file containing the document's content read out loud.

        :param model_name_or_path: the text to speech model, for example `espnet/kan-bayashi_ljspeech_vits`
        :param generated_audio_dir: folder to save the audio file to
        :param audio_format: the format to save the audio into (wav, mp3, ...)
        :param subtype: see soundfile.write()
        :param audio_naming_function: function mapping the input text into the audio file name.
                By default, the audio file gets the name from the MD5 sum of the input text.
        :param transformers_params: parameters to pass over to the Text2Speech.from_pretrained() call.
        """
        super().__init__()
        self.converter = TextToSpeech(model_name_or_path=model_name_or_path, transformers_params=transformers_params)
        self.params: Dict[str, Any] = {
            "generated_audio_dir": generated_audio_dir,
            "audio_format": audio_format,
            "subtype": subtype,
            "audio_naming_function": audio_naming_function,
        }

    def run(self, documents: List[Document]) -> Tuple[Dict[str, List[AudioDocument]], str]:  # type: ignore
        audio_documents = []
        for doc in documents:

            logging.info(f"Processing document '{doc.id}'...")
            content_audio = self.converter.text_to_audio_file(text=doc.content, **self.params)

            audio_document = GeneratedAudioDocument.from_text_document(
                document_object=doc,
                generated_audio_content=content_audio,
                additional_meta={"audio_format": self.params["audio_format"], "sample_rate": self.converter.model.fs},
            )
            audio_document.type = "generative"
            audio_documents.append(audio_document)

        return {"documents": audio_documents}, "output_1"

    def run_batch(self, documents: List[List[Document]]) -> Tuple[Dict[str, List[List[AudioDocument]]], str]:  # type: ignore
        results = {"documents": []}
        for docs_list in documents:
            results["documents"].append(self.run(docs_list))

        return results, "output_1"
