from abc import abstractmethod, ABC
from typing import Union, Any

import logging
import datetime
from pathlib import Path

import numpy as np
import librosa
import torch
from deepmultilingualpunctuation import PunctuationModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer



def _denoise_speech(self, audio_data: Any, sample_rate: int):

    S_full, phase = librosa.magphase(librosa.stft(audio_data))
    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=sample_rate)))
    S_filter = np.minimum(S_full, S_filter)

    margin_i = 2
    margin_v = 10
    power = 2

    mask_i = librosa.util.softmask(S_filter, margin_i* (S_full - S_filter), power=power)
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
    S_foreground = mask_v * S_full
    #S_background = mask_i * S_full

    return S_foreground


class BaseSpeechTranscriber(ABC):
    """
    Converts audio containing speech into its trascription.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "facebook/wav2vec2-base-960h",
    ):
        super().__init__()
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name_or_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path)

    @abstractmethod
    def transcribe(self, audio_file: Path, sample_rate=16000):
        pass


class Wav2VecTranscriber(BaseSpeechTranscriber):
    """
    Converts audio containing speech into its trascription using HF models.
    Tested with `facebook/wav2vec-base-960` 
    (TODO try models that predict punctuation like `boris/xlsr-en-punctuation`)

    Returns the transcript.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        restore_punctuation: bool = True
    ):
        super().__init__()
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name_or_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path)

    def transcribe(self, audio_file: Path, sample_rate=16000, denoise=False):
        input_audio, _ = librosa.load(audio_file, sr=sample_rate)
        if denoise:
            input_audio = _denoise_speech(input_audio, sample_rate=sample_rate)

        input_values = self.tokenizer(input_audio, return_tensors="pt").input_values
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]

        return transcription
