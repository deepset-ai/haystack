from haystack import Document
from haystack.nodes import TransformersTranslator


def test_translator():
    en_to_de_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")

    original = "I live in Berlin"
    translation = "Ich lebe in Berlin"

    assert en_to_de_translator.translate(query=original) == translation
    assert en_to_de_translator.translate(documents=[original])[0] == translation
    assert en_to_de_translator.translate(documents=[Document(content=original)])[0].content == translation
