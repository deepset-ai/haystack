import logging
from typing import Union, List, Dict, Any, Tuple

import os
import hashlib
from pathlib import Path

from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf

from haystack.nodes import BaseComponent
from haystack.schema import Answer, AudioAnswer, GeneratedAudioAnswer


class AnswerToSpeech(BaseComponent):

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "espnet/kan-bayashi_ljspeech_vits",
        generated_audio_path: Path = Path(__file__).parent / "generated_audio_answers",
    ):
        super().__init__()
        self.model = Text2Speech.from_pretrained(model_name_or_path)
        self.generated_audio_path = generated_audio_path

        if not os.path.exists(self.generated_audio_path):
            os.mkdir(self.generated_audio_path)

    def text_to_speech(self, text: str) -> Any:
        filename = hashlib.md5(text.encode("utf-8")).hexdigest()
        path = self.generated_audio_path / f"{filename}.wav"

        # Duplicate answers might be in the list, in this case we save time by not regenerating.
        if not os.path.exists(path):
            output = self.model(text)["wav"]
            sf.write(path, output.numpy(), self.model.fs, "PCM_16")

        return path

    def run(self, answers: List[Answer]) -> Tuple[Dict[str, AudioAnswer], str]:

        audio_answers = []
        for answer in answers:

            logging.info(f"Processing answer '{answer.answer}' and its context...")
            answer_audio = self.text_to_speech(answer.answer)
            context_audio = self.text_to_speech(answer.context)

            audio_answer = GeneratedAudioAnswer.from_text_answer(
                answer_object=answer, generated_audio_answer=answer_audio, generated_audio_context=context_audio
            )
            audio_answer.type = "generative"
            audio_answers.append(audio_answer)

        return {"answers": audio_answers}, "output_1"

    def run_batch(self, answers: List[Answer]) -> Tuple[Dict[str, AudioAnswer], str]:
        return self.run(answers)
