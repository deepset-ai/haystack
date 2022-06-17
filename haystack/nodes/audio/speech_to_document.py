from email.generator import Generator
from typing import List, Union

import math
import logging
from pathlib import Path

from tqdm import tqdm
from pydub import AudioSegment

from haystack.nodes import BaseComponent
from haystack.schema import AudioAlignment, Span, SpeechDocument
from haystack.errors import AudioNodeError
from haystack.nodes.audio.utils import get_speech_transcriber, get_transcript_aligner


class SpeechToDocument(BaseComponent):
    """
    Converts audio files into Documents with alignment data using a speech-to-text
    model and a forced aligner.
    """
    
    outgoing_edges = 1

    def __init__(
        self,
        transcriber_model_name_or_path: Union[str, Path] = "facebook/wav2vec2-base-960h", 
        transcriber_implementation: str = "wav2vec",
        aligner_implementation: str = "aeneas",
        fragment_length: int = 30
    ):
        super().__init__()
        self.transcriber = get_speech_transcriber(transcriber_implementation)(model_name_or_path=transcriber_model_name_or_path)
        self.aligner = get_transcript_aligner(aligner_implementation)()
        self.fragment_length = fragment_length

    def _get_num_fragments(self, path: Path):
        audio_format = path.suffix.replace(".", "")
        audio = AudioSegment.from_file(path, format=audio_format)
        return math.ceil(audio.duration_seconds / self.fragment_length) 

    def _get_fragments(self, path: Path):
        """
        Returns the input audio in chunks that can be processed by the transcriber.
        """
        audio_format = path.suffix.replace(".", "")
        audio = AudioSegment.from_file(path, format=audio_format)
        n_fragments = math.ceil(audio.duration_seconds / self.fragment_length) 
        fragment_path = Path(f"/tmp/[frag]__{path.name}")

        for fragment_id in range(n_fragments):
            fragment = audio[fragment_id*self.fragment_length*1000: (fragment_id+1)*self.fragment_length*1000]
            fragment.export(fragment_path, format=audio_format)

            yield fragment_path

    def _align(self, audio_file: Path, transcript_path: Path):
        """
        Generates the alignments and returns a list of AudioAlignment objects.
        """
        raw_alignments = self.aligner.align(audio_file=audio_file, transcript_file=transcript_path)

        accumulator = 0
        alignments = []
        for raw_alignment in raw_alignments:
            if raw_alignment["lines"]:
                word = raw_alignment["lines"][0]
                word_len = len(word) + 1   # 1 for the whitespace
                alignment = AudioAlignment(
                    offset_audio=Span(int(float(raw_alignment["begin"]) * 1000),int(float(raw_alignment["end"]) * 1000)),
                    offset_text=Span(accumulator, accumulator+word_len),
                    aligned_string=word
                )
                alignments.append(alignment)
                accumulator += word_len

        return alignments

    def run(self, file_paths: List[Path]):  # type: ignore
        documents = []
        for audio_file in file_paths:

            if audio_file.suffix != ".wav":
                raise AudioNodeError(
                    f"{audio_file.suffix} files are not supported by SpeechToDocument. "
                    "For now only .wav files are supported. Please convert your files."
                )

            complete_transcript = ""

            logging.info(f"Processing {audio_file}")
            for fragment_file in tqdm(self._get_fragments(audio_file), total=self._get_num_fragments(audio_file)):
                transcript = self.transcriber.transcribe(fragment_file)

                complete_transcript += transcript

            transcript_path = Path(f"/tmp/{audio_file.name}_transcript.txt")
            with open(transcript_path, 'w') as tf:
                tf.write(complete_transcript.replace(" ", "\n"))

            document = SpeechDocument(
                content=complete_transcript, 
                content_audio=audio_file, 
                content_type="audio",
                alignment_data=self._align(audio_file=audio_file, transcript_path=transcript_path),
                meta={"name": audio_file}
            )
            documents.append(document)

        return {"documents": documents}, "output_1"

    def run_batch():
        raise NotImplemented
