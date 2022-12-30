import os

import pytest
import numpy as np

try:
    import soundfile as sf

    soundfile_not_found = False
except:
    soundfile_not_found = True

from haystack.schema import Span, Answer, SpeechAnswer, Document, SpeechDocument
from haystack.nodes.audio import AnswerToSpeech, DocumentToSpeech
from haystack.nodes.audio._text_to_speech import TextToSpeech

from ..conftest import SAMPLES_PATH


@pytest.mark.skipif(soundfile_not_found, reason="soundfile not found")
class TestTextToSpeech:
    def test_text_to_speech_train_mode(self):
        text2speech = TextToSpeech(
            model_name_or_path="espnet/kan-bayashi_ljspeech_vits",
            transformers_params={"seed": 777, "always_fix_seed": True},
        )
        eval_mode_audio_data = text2speech.text_to_audio_data(text="answer")
        text2speech.model.model.train()
        train_mode_audio_data = text2speech.text_to_audio_data(text="answer")

        assert np.allclose(train_mode_audio_data, eval_mode_audio_data)

    def test_text_to_speech_audio_data(self):
        text2speech = TextToSpeech(
            model_name_or_path="espnet/kan-bayashi_ljspeech_vits",
            transformers_params={"seed": 777, "always_fix_seed": True},
        )
        expected_audio_data, _ = sf.read(SAMPLES_PATH / "audio" / "answer.wav")
        audio_data = text2speech.text_to_audio_data(text="answer")

        assert np.allclose(expected_audio_data, audio_data, atol=0.001)

    def test_text_to_speech_audio_file(self, tmp_path):
        text2speech = TextToSpeech(
            model_name_or_path="espnet/kan-bayashi_ljspeech_vits",
            transformers_params={"seed": 777, "always_fix_seed": True},
        )
        expected_audio_data, _ = sf.read(SAMPLES_PATH / "audio" / "answer.wav")
        audio_file = text2speech.text_to_audio_file(text="answer", generated_audio_dir=tmp_path / "test_audio")
        assert os.path.exists(audio_file)
        assert np.allclose(expected_audio_data, sf.read(audio_file)[0], atol=0.001)

    def test_text_to_speech_compress_audio(self, tmp_path):
        text2speech = TextToSpeech(
            model_name_or_path="espnet/kan-bayashi_ljspeech_vits",
            transformers_params={"seed": 777, "always_fix_seed": True},
        )
        expected_audio_file = SAMPLES_PATH / "audio" / "answer.wav"
        audio_file = text2speech.text_to_audio_file(
            text="answer", generated_audio_dir=tmp_path / "test_audio", audio_format="mp3"
        )
        assert os.path.exists(audio_file)
        assert audio_file.suffix == ".mp3"
        # FIXME find a way to make sure the compressed audio is similar enough to the wav version.
        # At a manual inspection, the code seems to be working well.

    def test_text_to_speech_naming_function(self, tmp_path):
        text2speech = TextToSpeech(
            model_name_or_path="espnet/kan-bayashi_ljspeech_vits",
            transformers_params={"seed": 777, "always_fix_seed": True},
        )
        expected_audio_file = SAMPLES_PATH / "audio" / "answer.wav"
        audio_file = text2speech.text_to_audio_file(
            text="answer", generated_audio_dir=tmp_path / "test_audio", audio_naming_function=lambda text: text
        )
        assert os.path.exists(audio_file)
        assert audio_file.name == expected_audio_file.name
        assert np.allclose(sf.read(expected_audio_file)[0], sf.read(audio_file)[0], atol=0.001)

    def test_answer_to_speech(self, tmp_path):
        text_answer = Answer(
            answer="answer",
            type="extractive",
            context="the context for this answer is here",
            offsets_in_document=[Span(31, 37)],
            offsets_in_context=[Span(21, 27)],
            meta={"some_meta": "some_value"},
        )
        expected_audio_answer = SAMPLES_PATH / "audio" / "answer.wav"
        expected_audio_context = SAMPLES_PATH / "audio" / "the context for this answer is here.wav"

        answer2speech = AnswerToSpeech(
            generated_audio_dir=tmp_path / "test_audio",
            audio_params={"audio_naming_function": lambda text: text},
            transformers_params={"seed": 777, "always_fix_seed": True},
        )
        results, _ = answer2speech.run(answers=[text_answer])

        audio_answer: SpeechAnswer = results["answers"][0]
        assert isinstance(audio_answer, SpeechAnswer)
        assert audio_answer.type == "generative"
        assert audio_answer.answer_audio.name == expected_audio_answer.name
        assert audio_answer.context_audio.name == expected_audio_context.name
        assert audio_answer.answer == "answer"
        assert audio_answer.context == "the context for this answer is here"
        assert audio_answer.offsets_in_document == [Span(31, 37)]
        assert audio_answer.offsets_in_context == [Span(21, 27)]
        assert audio_answer.meta["some_meta"] == "some_value"
        assert audio_answer.meta["audio_format"] == "wav"

        assert np.allclose(sf.read(audio_answer.answer_audio)[0], sf.read(expected_audio_answer)[0], atol=0.001)
        assert np.allclose(sf.read(audio_answer.context_audio)[0], sf.read(expected_audio_context)[0], atol=0.001)

    def test_document_to_speech(self, tmp_path):
        text_doc = Document(
            content="this is the content of the document", content_type="text", meta={"name": "test_document.txt"}
        )
        expected_audio_content = SAMPLES_PATH / "audio" / "this is the content of the document.wav"

        doc2speech = DocumentToSpeech(
            generated_audio_dir=tmp_path / "test_audio",
            audio_params={"audio_naming_function": lambda text: text},
            transformers_params={"seed": 777, "always_fix_seed": True},
        )
        results, _ = doc2speech.run(documents=[text_doc])

        audio_doc: SpeechDocument = results["documents"][0]
        assert isinstance(audio_doc, SpeechDocument)
        assert audio_doc.content_type == "audio"
        assert audio_doc.content_audio.name == expected_audio_content.name
        assert audio_doc.content == "this is the content of the document"
        assert audio_doc.meta["name"] == "test_document.txt"
        assert audio_doc.meta["audio_format"] == "wav"

        assert np.allclose(sf.read(audio_doc.content_audio)[0], sf.read(expected_audio_content)[0], atol=0.001)
