from typing import Union, Callable, Any, Optional, Dict, List

import os
import logging
import hashlib
from pathlib import Path

import numpy as np
import torch

try:
    import soundfile as sf
    from espnet2.bin.tts_inference import Text2Speech as _Text2SpeechModel

except OSError as ose:
    logging.exception(
        "`libsndfile` not found, it's probably not installed. The node will most likely crash. "
        "Please install soundfile's dependencies (https://python-soundfile.readthedocs.io/en/latest/)"
    )
from pydub import AudioSegment

from haystack.errors import AudioNodeError
from haystack.modeling.utils import initialize_device_settings

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    This class converts text into audio using text-to-speech models.

    NOTE: This is NOT a node. Use AnswerToSpeech or DocumentToSpeech.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        use_gpu: bool = True,
        transformers_params: Optional[Dict[str, Any]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        """
        :param model_name_or_path: The text to speech model, for example `espnet/kan-bayashi_ljspeech_vits`.
        :param use_gpu: Whether to use GPU (if available). Defaults to True.
        :param transformers_params: Parameters to pass over to the `Text2Speech.from_pretrained()` call.
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        super().__init__()

        resolved_devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(resolved_devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )

        self.model = _Text2SpeechModel.from_pretrained(
            str(model_name_or_path), device=resolved_devices[0].type, **(transformers_params or {})
        )

    def text_to_audio_file(
        self,
        text: str,
        generated_audio_dir: Path,
        audio_format: str = "wav",
        subtype: str = "PCM_16",
        sample_width: int = 2,
        channels_count: int = 1,
        bitrate: str = "320k",
        normalized=True,
        audio_naming_function: Callable = lambda text: hashlib.md5(text.encode("utf-8")).hexdigest(),
    ) -> Path:
        """
        Convert an input string into an audio file containing the same string read out loud.

        :param text: The text to convert into audio.
        :param generated_audio_dir: The folder to save the audio file to.
        :param audio_format: The format to save the audio into (wav, mp3, ...).
            Supported formats:
             - Uncompressed formats thanks to `soundfile` (see `libsndfile documentation <https://libsndfile.github.io/libsndfile/api.html>`_ for a list of supported formats)
             - Compressed formats thanks to `pydub` (uses FFMPEG: run `ffmpeg -formats` in your terminal to see the list of supported formats).
        :param subtype: Used only for uncompressed formats. See https://libsndfile.github.io/libsndfile/api.html for the complete list of available subtypes.
        :param sample_width: Used only for compressed formats. The sample width of your audio. Defaults to 2.
        :param channels count: Used only for compressed formats. THe number of channels your audio file has: 1 for mono, 2 for stereo. Depends on the model, but it's often mono so it defaults to 1.
        :param bitrate: Used only for compressed formats. The desired bitrate of your compressed audio. Defaults to '320k'.
        :param normalized: Used only for compressed formats. Normalizes the audio before compression (range 2^15) or leaves it untouched.
        :param audio_naming_function: A function mapping the input text into the audio file name.
                By default, the audio file gets the name from the MD5 sum of the input text.
        :return: The path to the generated file.
        """
        if not os.path.exists(generated_audio_dir):
            os.mkdir(generated_audio_dir)

        filename = audio_naming_function(text)
        file_path = generated_audio_dir / f"{filename}.{audio_format}"

        # To save time, we avoid regenerating if a file with the same name is already in the folder.
        # This happens rather often in text from AnswerToSpeech.
        if not os.path.exists(file_path):
            audio_data = self.text_to_audio_data(text)
            if audio_format.upper() in sf.available_formats().keys():
                sf.write(
                    data=audio_data, file=file_path, format=audio_format, subtype=subtype, samplerate=self.model.fs
                )
            else:
                self.compress_audio(
                    data=audio_data,
                    path=file_path,
                    format=audio_format,
                    sample_rate=self.model.fs,
                    sample_width=sample_width,
                    channels_count=channels_count,
                    bitrate=bitrate,
                    normalized=normalized,
                )

        return file_path

    def text_to_audio_data(self, text: str, _models_output_key: str = "wav") -> np.array:  # type: ignore [valid-type]
        """
        Convert an input string into a numpy array representing the audio.

        :param text: The text to convert into audio.
        :param _models_output_key: The key in the prediction dictionary that contains the audio data. Defaults to 'wav'.
        :return: A numpy array representing the audio generated by the model.
        """
        prediction = self.model(text)
        if not prediction:
            raise AudioNodeError(
                "The model returned no predictions. Make sure you selected a valid text-to-speech model."
            )
        output = prediction.get(_models_output_key, None)
        if output is None:
            raise AudioNodeError(
                f"The model returned no output under the {_models_output_key} key. The available output keys are {prediction.keys()}. Make sure you selected the right key."
            )
        return output.cpu().numpy()

    def compress_audio(
        self,
        data: np.array,  # type: ignore [valid-type]
        path: Path,
        format: str,
        sample_rate: int,
        sample_width: int = 2,
        channels_count: int = 1,
        bitrate: str = "320k",
        normalized=True,
    ):
        """
        Export a numpy array into a compressed audio file of the desired format.

        :param data: The audio data to compress.
        :param path: The path to save the compressed audio at.
        :param format: The format to compress the data into ('mp3', 'wav', 'raw', 'ogg' or other ffmpeg/avconv supported files).
        :param sample_rate: The sample rate of the audio. Depends on the model.
        :param sample_width: The sample width of your audio. Defaults to 2.
        :param channels count: The number of channels your audio file has: 1 for mono, 2 for stereo. Depends on the model, but it's often mono so it defaults to 1.
        :param bitrate: The desired bitrate of your compressed audio. Default to '320k'.
        :param normalized: Normalizes the audio before compression (range 2^15) or leaves it untouched.
        """
        data = np.int16((data * 2**15) if normalized else data)
        audio = AudioSegment(data.tobytes(), frame_rate=sample_rate, sample_width=sample_width, channels=channels_count)
        audio.export(path, format=format, bitrate=bitrate)
