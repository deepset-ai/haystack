from typing import Union, Optional, Callable, List, Dict, Tuple, Any

import os
import hashlib
import logging
from pathlib import Path

from tqdm import tqdm
from pydub import AudioSegment

from haystack.nodes import BaseComponent
from haystack.schema import AudioAlignment, Document, SpeechDocument, Answer, SpeechAnswer, Span
from haystack.nodes.audio.utils import TextToSpeech
from haystack.errors import AudioNodeError


logger = logging.getLogger(__name__)


class AnswerToSpeech(BaseComponent):
    """
    This node converts text-based Answers into AudioAnswers, where the answer and its context are
    read out into an audio file.
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = "espnet/kan-bayashi_ljspeech_vits",
        audio_answers_dir: Path = Path("./audio_answers"),
        audio_params: Optional[Dict[str, Any]] = None,
        audio_naming_function: Callable = lambda text: hashlib.md5(text.encode("utf-8")).hexdigest(),
        transformers_params: Optional[Dict[str, Any]] = None,
        progress_bar: bool = True,
    ):
        """
        Convert an input Answer into an audio file containing the answer and its context read out loud.

        If the Answer comes from a regular Document, the answer's audio is generated with a text-to-speech model.
        If the original document contains alignment data instead, the answer is extracted from the original
        audio using the alignment data.

        :param model_name_or_path: The text-to-speech model, for example `espnet/kan-bayashi_ljspeech_vits`.
        :param generated_audio_dir: The folder to save the audio file to.
        :param audio_naming_function: A function mapping the input text into the audio file name.
            By default, the audio file gets the name from the MD5 sum of the input text.
        :param audio_params: Additional parameters for the audio file. See `TextToSpeech` for details.
            The allowed parameters are:
            - audio_format: The format to save the audio into (wav, mp3, ...). Defaults to `wav`.
                Supported formats:
                - Uncompressed formats thanks to `soundfile` (see `libsndfile documentation <https://libsndfile.github.io/libsndfile/api.html>`_
                    for a list of supported formats).
                - Compressed formats thanks to `pydub`
                    (uses FFMPEG: run `ffmpeg -formats` in your terminal to see the list of supported formats).
            - subtype: Used only for uncompressed formats. See `libsndfile documentation <https://libsndfile.github.io/libsndfile/api.html>`_
                for the complete list of available subtypes. Defaults to `PCM_16`.
            - sample_width: Used only for compressed formats. The sample width of your audio. Defaults to 2.
            - channels count: Used only for compressed formats. The number of channels your audio file has:
                1 for mono, 2 for stereo. Depends on the model, but it's often mono so it defaults to 1.
            - bitrate: Used only for compressed formats. The desired bitrate of your compressed audio. Defaults to '320k'.
            - normalized: Used only for compressed formats. Normalizes the audio before compression (range 2^15)
                or leaves it untouched.
        :param transformers_params: The parameters to pass over to the `Text2Speech.from_pretrained()` call.
        :param progress_bar: Whether to show a progress bar while converting the text to audio.
        """
        super().__init__()

        if model_name_or_path:
            self.converter = TextToSpeech(model_name_or_path=model_name_or_path, transformers_params=transformers_params)
        else:
            logger.warning(
                "No text-to-speech model given to AnswerToSpeech. "
                "Answers coming from documents without alignment data will be ignored."
            )
        self.audio_answers_dir = audio_answers_dir
        self.audio_naming_function = audio_naming_function
        self.params: Dict[str, Any] = audio_params or {}
        self.progress_bar = progress_bar

        if not os.path.exists(self.audio_answers_dir):
            logger.warning(f"The directory {self.audio_answers_dir} seems not to exist. Creating it.")
            os.makedirs(self.audio_answers_dir)

    def _extract_audio_snippet(self, audio_data: AudioSegment, audio_span: Span, filename: str, format="wav") -> Path:
        """
        Extract a snippet corresponding to a certain position in text from its source audio,
        using the provided alignment data.

        :param audio_data: the audio to take the snippet from
        :param audio_span: the result of _get_milliseconds_from_chars() 
        :param filename: name of the audio snippet file (only filename, no extension nor complete path)
        :param format: the format to use to save the audio snippet.
        :return: the path the snippet was saved at.
        """
        path = self.audio_answers_dir / f"{filename}.{format}"
        answer_audio = audio_data[audio_span.start : audio_span.end]
        answer_audio.export(path, format=format)
        return path

    def _extract_audio_answer(self, answer: Answer, document: SpeechDocument):
        """
        Returns a SpeechAnswer with extracted audio.
        """
        audio_format = document.content_audio.suffix.replace(".", "")
        document_audio = AudioSegment.from_file(document.content_audio, format=audio_format)

        audio_answer_span = _get_milliseconds_from_chars(
            text_span=answer.offsets_in_document[0],
            alignment_data=document.alignment_data,
        )
        audio_answer_path = self._extract_audio_snippet(
            audio_data=document_audio,
            audio_span=audio_answer_span,
            filename=self.audio_naming_function(answer.answer)
        )

        context_position = document.content.index(answer.context)   # Hopefully there won't be duplicate contexts in the document.
        audio_context_span = _get_milliseconds_from_chars(
            text_span=Span(context_position, context_position+len(answer.context)),
            alignment_data=document.alignment_data,
        )
        audio_context_path = self._extract_audio_snippet(
            audio_data=document_audio,
            audio_span = audio_context_span,
            filename=self.audio_naming_function(answer.context)
        )
        audio_answer = SpeechAnswer.from_text_answer(
            answer_object=answer,
            audio_answer=audio_answer_path,
            audio_context=audio_context_path,
            offset_in_audio=audio_answer_span,
            additional_meta={
                "audio_format": self.params.get("audio_format", audio_answer_path.suffix.replace(".", "")),
                "sample_rate": document_audio.frame_rate
            },
        )
        audio_answer.type = "extractive"
        return audio_answer

    def _generate_audio_answer(self, answer: Answer):
        """
        Returns a SpeechAnswer with generated audio.
        """
        answer_audio = self.converter.text_to_audio_file(
            text=answer.answer, 
            generated_audio_dir=self.audio_answers_dir, 
            audio_naming_function=self.audio_naming_function, 
            **self.params
        )
        if isinstance(answer.context, str):  # Can be a table!
            context_audio = self.converter.text_to_audio_file(
                text=answer.context, 
                generated_audio_dir=self.audio_answers_dir, 
                audio_naming_function=self.audio_naming_function, 
                **self.params
            )
        else:
            logger.warning(
                f"The context for answer '{answer.answer}' "
                f"is not readable (type is {type(answer.context)}). Skipping it."
            )

        audio_answer = SpeechAnswer.from_text_answer(
            answer_object=answer,
            audio_answer=answer_audio,
            audio_context=context_audio,
            additional_meta={
                "audio_format": self.params.get("audio_format", answer_audio.suffix.replace(".", "")),
                "sample_rate": self.converter.model.fs,
            },
        )
        audio_answer.type = "generative"
        return audio_answer

    def run(self, answers: List[Answer], documents: List[Document]) -> Tuple[Dict[str, List[Answer]], str]:  # type: ignore
        audio_answers = []
        for answer in tqdm(answers):

            source_docs = [doc for doc in documents if doc.id == answer.document_id]
            if not source_docs:
                logger.error(f"Source document for answer '{answer.answer}' not found! The audio will be generated.")
            source_doc = source_docs[0]

            if isinstance(source_doc, SpeechDocument) and source_doc.alignment_data:
                audio_answers.append(self._extract_audio_answer(answer, source_doc))
            else:
                audio_answers.append(self._generate_audio_answer(answer))

        return {"answers": audio_answers}, "output_1"

    def run_batch(self, answers: List[List[Answer]],  documents: List[List[Document]]) -> Tuple[Dict[str, List[List[Answer]]], str]:  # type: ignore
        results: Dict[str, List[List[Answer]]] = {"answers": []}

        for answers_list, documents_list in zip(answers, documents):
            results["answers"].append(self.run(answers_list, documents_list)[0]["answers"])

        return results, "output_1"



def _get_milliseconds_from_chars(text_span: Span, alignment_data: List[AudioAlignment]) -> Span:
    """
    Converts a position in text into a position in audio using the provided alignment data.
    """
    candidate_start = [align for align in alignment_data if text_span.start in align.offset_text]
    candidate_end = [align for align in alignment_data if text_span.end in align.offset_text]

    if not candidate_start or not candidate_end:
        raise AudioNodeError("Could not find matching alignments in the alignment data.")
    if len(candidate_start) > 1 or len(candidate_end) > 1:
        raise AudioNodeError("The alignment data is ambiguous: there are several alignments for the same position.")

    print("----> Start: ", text_span.start, candidate_start[0])
    print("----> End: ", text_span.end, candidate_end[0])
    
    return Span(candidate_start[0].offset_audio.start, candidate_end[0].offset_audio.end)