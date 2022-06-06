import logging
from typing import Any, Union, Optional, List, Dict, Tuple

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
        generated_audio_dir: Path = Path("./generated_audio_answers"),
        audio_params: Optional[Dict[str, Any]] = None,
        transformers_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Convert an input Answer into an audio file containing the answer's answer and context read out loud.

        :param model_name_or_path: the text to speech model, for example `espnet/kan-bayashi_ljspeech_vits`
        :param generated_audio_dir: folder to save the audio file to
        :param audio_params: additional parameter for the audio file. See `TextToSpeech` for details.
            The allowed parameters are:
            - audio_format: the format to save the audio into (wav, mp3, ...). Defaults to `wav`.
                Formats supported:
                - Uncompressed formats thanks to `soundfile` (see https://libsndfile.github.io/libsndfile/api.html) 
                    for a list of supported formats)
                - Compressed formats thanks to `pydub` 
                    (uses FFMPEG: run `ffmpeg -formats` in your terminal to see the list of supported formats)
            - subtype: Used only for uncompressed formats. See https://libsndfile.github.io/libsndfile/api.html 
                for the complete list of available subtypes. Defaults to `PCM_16`.
            - sample_width: Used only for compressed formats. The sample width of your audio. Defaults to 2
            - channels count: Used only for compressed formats. How many channels your audio file has: 
                1 for mono, 2 for stereo. Depends on the model, but it's often mono so it defaults to 1.
            - bitrate: Used only for compressed formats. The desired bitrate of your compressed audio. Default to '320k'
            - normalized: Used only for compressed formats. Whether to normalize the audio before compression (range 2^15) 
                or leave it untouched
            - audio_naming_function: function mapping the input text into the audio file name.
                By default, the audio file gets the name from the MD5 sum of the input text.
        :param transformers_params: parameters to pass over to the `Text2Speech.from_pretrained()` call.
        """
        super().__init__()
        self.converter = TextToSpeech(model_name_or_path=model_name_or_path, transformers_params=transformers_params)
        self.generated_audio_dir = generated_audio_dir
        self.params: Dict[str, Any] = audio_params or {}

    def run(self, answers: List[Answer]) -> Tuple[Dict[str, List[AudioAnswer]], str]:  # type: ignore
        audio_answers = []
        for answer in answers:

            logging.info(f"Processing answer '{answer.answer}' and its context")
            answer_audio = self.converter.text_to_audio_file(text=answer.answer, generated_audio_dir=self.generated_audio_dir, **self.params)
            if isinstance(answer.context, str):
                context_audio = self.converter.text_to_audio_file(text=answer.context, generated_audio_dir=self.generated_audio_dir, **self.params)

            audio_answer = GeneratedAudioAnswer.from_text_answer(
                answer_object=answer,
                generated_audio_answer=answer_audio,
                generated_audio_context=context_audio,
                additional_meta={"audio_format": self.params.get("audio_format", answer_audio.suffix.replace(".", "")), "sample_rate": self.converter.model.fs},
            )
            audio_answer.type = "generative"
            audio_answers.append(audio_answer)

        return {"answers": audio_answers}, "output_1"

    def run_batch(self, answers: List[List[Answer]]) -> Tuple[Dict[str, List[AudioAnswer]], str]:  # type: ignore
        results: Dict[str, List[List[AudioAnswer]]] = {"answers": []}
        for answers_list in answers:
            results["answers"].append(self.run(answers_list)[0]["answers"])

        return results, "output_1"
