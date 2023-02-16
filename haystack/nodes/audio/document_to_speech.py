from typing import Union, Optional, List, Dict, Tuple, Any

from pathlib import Path
from tqdm.auto import tqdm

from haystack.nodes import BaseComponent
from haystack.schema import Document, SpeechDocument
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
        generated_audio_dir: Path = Path("./generated_audio_documents"),
        audio_params: Optional[Dict[str, Any]] = None,
        transformers_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Convert an input Document into an audio file containing the document's content read out loud.

        :param model_name_or_path: The text to speech model, for example `espnet/kan-bayashi_ljspeech_vits`.
        :param generated_audio_dir: The folder to save the audio file to.
        :param audio_params: Additional parameters for the audio file. See `TextToSpeech` for details.
            The allowed parameters are:
            - audio_format: The format to save the audio into (wav, mp3, ...). Defaults to `wav`.
                Supported formats:
                - Uncompressed formats thanks to `soundfile` (see https://libsndfile.github.io/libsndfile/api.html)
                    for a list of supported formats).
                - Compressed formats thanks to `pydub`
                    (uses FFMPEG: run `ffmpeg -formats` in your terminal to see the list of supported formats).
            - subtype: Used only for uncompressed formats. See https://libsndfile.github.io/libsndfile/api.html
                for the complete list of available subtypes. Defaults to `PCM_16`.
            - sample_width: Used only for compressed formats. The sample width of your audio. Defaults to 2.
            - channels count: Used only for compressed formats. The number of channels your audio file has:
                1 for mono, 2 for stereo. Depends on the model, but it's often mono so it defaults to 1.
            - bitrate: Used only for compressed formats. The desired bitrate of your compressed audio. Defaults to '320k'.
            - normalized: Used only for compressed formats. Normalizes the audio before compression (range 2^15)
                or leaves it untouched.
            - audio_naming_function: The function mapping the input text into the audio file name.
                By default, the audio file gets the name from the MD5 sum of the input text.
        :param transformers_params: The parameters to pass over to the `Text2Speech.from_pretrained()` call.
        """
        super().__init__()
        self.converter = TextToSpeech(model_name_or_path=model_name_or_path, transformers_params=transformers_params)
        self.generated_audio_dir = generated_audio_dir
        self.params: Dict[str, Any] = audio_params or {}

    def run(self, documents: List[Document]) -> Tuple[Dict[str, List[Document]], str]:  # type: ignore
        audio_documents = []
        for doc in tqdm(documents):
            content_audio = self.converter.text_to_audio_file(
                text=doc.content, generated_audio_dir=self.generated_audio_dir, **self.params
            )
            audio_document = SpeechDocument.from_text_document(
                document_object=doc,
                audio_content=content_audio,
                additional_meta={
                    "audio_format": self.params.get("audio_format", content_audio.suffix.replace(".", "")),
                    "sample_rate": self.converter.model.fs,
                },
            )
            audio_document.type = "generative"
            audio_documents.append(audio_document)

        return {"documents": audio_documents}, "output_1"

    def run_batch(self, documents: List[List[Document]]) -> Tuple[Dict[str, List[List[Document]]], str]:  # type: ignore
        results: Dict[str, List[List[Document]]] = {"documents": []}
        for docs_list in documents:
            results["documents"].append(self.run(docs_list)[0]["documents"])

        return results, "output_1"
