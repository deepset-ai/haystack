import os
from pathlib import Path

import numpy as np
import soundfile as sf

from haystack.schema import Span, Answer, GeneratedAudioAnswer, Document, GeneratedAudioDocument
from haystack.nodes.audio import AnswerToSpeech, DocumentToSpeech
from haystack.nodes.audio._text_to_speech import TextToSpeech

from ..conftest import SAMPLES_PATH


def test_text_to_speech_audio_data():
    text2speech = TextToSpeech(
        model_name_or_path="espnet/kan-bayashi_ljspeech_vits",
        transformers_params={"seed": 777, "always_fix_seed": True},
    )
    expected_audio_data, _ = sf.read(SAMPLES_PATH / "audio" / "answer.wav")
    audio_data = text2speech.text_to_audio_data(text="answer")

    assert np.allclose(expected_audio_data, audio_data, atol=0.0001)


def test_text_to_speech_audio_file(tmp_path):
    text2speech = TextToSpeech(
        model_name_or_path="espnet/kan-bayashi_ljspeech_vits",
        transformers_params={"seed": 777, "always_fix_seed": True},
    )
    expected_audio_data, _ = sf.read(SAMPLES_PATH / "audio" / "answer.wav")
    audio_file = text2speech.text_to_audio_file(text="answer", generated_audio_dir=tmp_path / "test_audio")
    assert os.path.exists(audio_file)
    assert np.allclose(expected_audio_data, sf.read(audio_file)[0], atol=0.0001)


def test_text_to_speech_compress_audio(tmp_path):
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


def test_text_to_speech_naming_function(tmp_path):
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
    assert np.allclose(sf.read(expected_audio_file)[0], sf.read(audio_file)[0], atol=0.0001)


def test_answer_to_speech(tmp_path):
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
        audio_params={"audio_naming_function":lambda text: text},
        transformers_params={"seed": 777, "always_fix_seed": True},
    )
    results, _ = answer2speech.run(answers=[text_answer])

    audio_answer: GeneratedAudioAnswer = results["answers"][0]
    assert isinstance(audio_answer, GeneratedAudioAnswer)
    assert audio_answer.type == "generative"
    assert audio_answer.answer.name == expected_audio_answer.name
    assert audio_answer.context.name == expected_audio_context.name
    assert audio_answer.answer_transcript == "answer"
    assert audio_answer.context_transcript == "the context for this answer is here"
    assert audio_answer.offsets_in_document == [Span(31, 37)]
    assert audio_answer.offsets_in_context == [Span(21, 27)]
    assert audio_answer.meta["some_meta"] == "some_value"
    assert audio_answer.meta["audio_format"] == "wav"

    assert np.array_equal(sf.read(audio_answer.answer)[0], sf.read(expected_audio_answer)[0])
    assert np.array_equal(sf.read(audio_answer.context)[0], sf.read(expected_audio_context)[0])


def test_document_to_speech(tmp_path):
    text_doc = Document(
        content="this is the content of the document", content_type="text", meta={"name": "test_document.txt"}
    )
    expected_audio_content = SAMPLES_PATH / "audio" / "this is the content of the document.wav"

    doc2speech = DocumentToSpeech(
        generated_audio_dir=tmp_path / "test_audio",
        audio_params={"audio_naming_function":lambda text: text},
        transformers_params={"seed": 777, "always_fix_seed": True},
    )
    results, _ = doc2speech.run(documents=[text_doc])

    audio_doc: GeneratedAudioDocument = results["documents"][0]
    assert isinstance(audio_doc, GeneratedAudioDocument)
    assert audio_doc.content_type == "audio"
    assert audio_doc.content.name == expected_audio_content.name
    assert audio_doc.content_transcript == "this is the content of the document"
    assert audio_doc.meta["name"] == "test_document.txt"
    assert audio_doc.meta["audio_format"] == "wav"

    assert np.array_equal(sf.read(audio_doc.content)[0], sf.read(expected_audio_content)[0])
