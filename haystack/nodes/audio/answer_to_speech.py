import logging
from typing import Any, Union, Callable, Optional, List, Dict, Tuple

import hashlib
from pathlib import Path

from haystack.nodes import BaseComponent
from haystack.schema import Answer, AudioAnswer, GeneratedAudioAnswer
from haystack.nodes.audio._text_to_speech import TextToSpeech


class AnswerToSpeech(BaseComponent):
    """
    This node converts text-based Answers into AudioAnswers, where the answer and its context are
    read out into an audio file.
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "espnet/kan-bayashi_ljspeech_vits",
        generated_audio_path: Path = Path(__file__).parent / "generated_audio_answers",
        audio_format: str = "wav",
        subtype: str = "PCM_16",
        audio_naming_function: Callable = lambda text: hashlib.md5(text.encode("utf-8")).hexdigest(),
        transformers_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Convert an input Answer into an audio file containing the answer's answer and context read out loud.

        :param model_name_or_path: the text to speech model, for example `espnet/kan-bayashi_ljspeech_vits`
        :param generated_audio_path: folder to save the audio file to
        :param audio_format: the format to save the audio into (wav, mp3, ...)
        :param subtype: see soundfile.write()
        :param audio_naming_function: function mapping the input text into the audio file name.
                By default, the audio file gets the name from the MD5 sum of the input text.
        :param transformers_params: parameters to pass over to the Text2Speech.from_pretrained() call.
        """
        super().__init__()
        self.converter = TextToSpeech(model_name_or_path=model_name_or_path, transformers_params=transformers_params)
        self.generated_audio_path = generated_audio_path
        self.audio_format = audio_format
        self.subtype = subtype
        self.audio_naming_function = audio_naming_function

    def run(
        self,
        answers: List[Answer],
        generated_audio_path: Optional[Path] = None,
        audio_format: Optional[str] = None,
        subtype: Optional[str] = None,
        audio_naming_function: Optional[Callable] = None,
    ) -> Tuple[Dict[str, AudioAnswer], str]:

        params = {
            "generated_audio_path": generated_audio_path or self.generated_audio_path,
            "audio_format": audio_format or self.audio_format,
            "subtype": subtype or self.subtype,
            "audio_naming_function": audio_naming_function or self.audio_naming_function,
        }
        audio_answers = []
        for answer in answers:

            logging.info(f"Processing answer '{answer.answer}' and its context...")
            answer_audio = self.converter.text_to_audio_file(text=answer.answer, **params)
            context_audio = self.converter.text_to_audio_file(answer.context, **params)

            audio_answer = GeneratedAudioAnswer.from_text_answer(
                answer_object=answer,
                generated_audio_answer=answer_audio,
                generated_audio_context=context_audio,
                additional_meta={
                    "audio_format": params["audio_format"],
                    "subtype": params["subtype"],
                    "sample_rate": self.converter.model.fs,
                },
            )
            audio_answer.type = "generative"
            audio_answers.append(audio_answer)

        return {"answers": audio_answers}, "output_1"

    def run_batch(self, answers: List[List[Answer]]) -> Tuple[Dict[str, AudioAnswer], str]:
        results = {"answers": []}
        for answers_list in answers:
            results["answers"].append(self.run(answers_list))

        return results, "output_1"
