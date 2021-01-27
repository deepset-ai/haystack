from haystack import Document

import pytest

EXPECTED_OUTPUT = "Ich lebe in Berlin"
INPUT = "I live in Berlin"


def test_translator_with_query(translator):
    assert translator.translate(query=INPUT) == EXPECTED_OUTPUT


def test_translator_with_list(translator):
    assert translator.translate(documents=[INPUT])[0] == EXPECTED_OUTPUT


def test_translator_with_document(translator):
    assert translator.translate(documents=[Document(text=INPUT)])[0].text == EXPECTED_OUTPUT


def test_translator_with_dictionary(translator):
    assert translator.translate(documents=[{"text": INPUT}])[0]["text"] == EXPECTED_OUTPUT


def test_translator_with_dictionary_with_dict_key(translator):
    assert translator.translate(documents=[{"key": INPUT}], dict_key="key")[0]["key"] == EXPECTED_OUTPUT


def test_translator_with_empty_input(translator):
    with pytest.raises(AttributeError):
        translator.translate()


def test_translator_with_query_and_documents(translator):
    with pytest.raises(AttributeError):
        translator.translate(query=INPUT, documents=[INPUT])


def test_translator_with_dict_without_text_key(translator):
    with pytest.raises(AttributeError):
        translator.translate(documents=[{"text1": INPUT}])


def test_translator_with_dict_with_non_string_value(translator):
    with pytest.raises(AttributeError):
        translator.translate(documents=[{"text": 123}])
