import pytest

import haystack
from haystack.schema import Document
from haystack.nodes import TransformersTranslator


ORIGINAL = "TEST QUERY"
TRANSLATION = "MOCK TRANSLATION"


class MockTokenizer:
    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def __call__(self, *a, **k):
        return self

    def to(self, *a, **k):
        return {}

    def batch_decode(self, *a, **k):
        return [TRANSLATION]


class MockModel:
    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def generate(self, *a, **k):
        return None

    def to(self, *a, **k):
        return None


@pytest.fixture
def mock_models(monkeypatch):
    monkeypatch.setattr(haystack.nodes.translator.transformers, "AutoModelForSeq2SeqLM", MockModel)
    monkeypatch.setattr(haystack.nodes.translator.transformers, "AutoTokenizer", MockTokenizer)


@pytest.fixture
def en_to_de_translator(mock_models) -> TransformersTranslator:
    return TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")


@pytest.fixture
def de_to_en_translator(mock_models) -> TransformersTranslator:
    return TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-de-en")


@pytest.mark.unit
def test_translator_with_query(en_to_de_translator):
    assert en_to_de_translator.translate(query=ORIGINAL) == TRANSLATION


@pytest.mark.unit
def test_translator_with_list(en_to_de_translator):
    assert en_to_de_translator.translate(documents=[ORIGINAL])[0] == TRANSLATION


@pytest.mark.unit
def test_translator_with_document(en_to_de_translator):
    assert en_to_de_translator.translate(documents=[Document(content=ORIGINAL)])[0].content == TRANSLATION


@pytest.mark.unit
def test_translator_with_document_preserves_original(en_to_de_translator):
    original_document = Document(content=ORIGINAL)
    en_to_de_translator.translate(documents=[original_document])
    assert original_document.content == ORIGINAL


@pytest.mark.unit
def test_translator_with_dictionary(en_to_de_translator):
    assert en_to_de_translator.translate(documents=[{"content": ORIGINAL}])[0]["content"] == TRANSLATION


@pytest.mark.unit
def test_translator_with_dictionary_preserves_original(en_to_de_translator):
    original_document = {"content": ORIGINAL}
    en_to_de_translator.translate(documents=[original_document])
    assert original_document["content"] == ORIGINAL


@pytest.mark.unit
def test_translator_with_dictionary_with_dict_key(en_to_de_translator):
    assert en_to_de_translator.translate(documents=[{"key": ORIGINAL}], dict_key="key")[0]["key"] == TRANSLATION


@pytest.mark.unit
def test_translator_with_empty_original(en_to_de_translator):
    with pytest.raises(AttributeError):
        en_to_de_translator.translate()


@pytest.mark.unit
def test_translator_with_query_and_documents(en_to_de_translator):
    with pytest.raises(AttributeError):
        en_to_de_translator.translate(query=ORIGINAL, documents=[ORIGINAL])


@pytest.mark.unit
def test_translator_with_dict_without_text_key(en_to_de_translator):
    with pytest.raises(AttributeError):
        en_to_de_translator.translate(documents=[{"text1": ORIGINAL}])


@pytest.mark.unit
def test_translator_with_dict_with_non_string_value(en_to_de_translator):
    with pytest.raises(AttributeError):
        en_to_de_translator.translate(documents=[{"text": 123}])
