from haystack.schema import Document

import pytest

EXPECTED_OUTPUT = "Ich lebe in Berlin"
INPUT = "I live in Berlin"


def test_translator_with_query(en_to_de_translator):
    assert en_to_de_translator.translate(query=INPUT) == EXPECTED_OUTPUT


def test_translator_with_list(en_to_de_translator):
    assert en_to_de_translator.translate(documents=[INPUT])[0] == EXPECTED_OUTPUT


def test_translator_with_document(en_to_de_translator):
    assert en_to_de_translator.translate(documents=[Document(content=INPUT)])[0].content == EXPECTED_OUTPUT


def test_translator_with_dictionary(en_to_de_translator):
    assert en_to_de_translator.translate(documents=[{"content": INPUT}])[0]["content"] == EXPECTED_OUTPUT


def test_translator_with_dictionary_with_dict_key(en_to_de_translator):
    assert en_to_de_translator.translate(documents=[{"key": INPUT}], dict_key="key")[0]["key"] == EXPECTED_OUTPUT


def test_translator_with_empty_input(en_to_de_translator):
    with pytest.raises(AttributeError):
        en_to_de_translator.translate()


def test_translator_with_query_and_documents(en_to_de_translator):
    with pytest.raises(AttributeError):
        en_to_de_translator.translate(query=INPUT, documents=[INPUT])


def test_translator_with_dict_without_text_key(en_to_de_translator):
    with pytest.raises(AttributeError):
        en_to_de_translator.translate(documents=[{"text1": INPUT}])


def test_translator_with_dict_with_non_string_value(en_to_de_translator):
    with pytest.raises(AttributeError):
        en_to_de_translator.translate(documents=[{"text": 123}])
