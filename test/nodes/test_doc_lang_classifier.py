import pytest

from haystack.schema import Document
from haystack.nodes.doc_language_classifier import LangdetectDocumentLanguageClassifier

DOCUMENTS = [
    Document(content="My name is Matteo and I live in Rome"),
    Document(content="Mi chiamo Matteo e vivo a Roma"),
    Document(content="Mi nombre es Matteo y vivo en Roma"),
]

EXPECTED_LANGUAGES = ["en", "it", "es"]


@pytest.fixture
def langdetect_doc_lang_classifier():
    return LangdetectDocumentLanguageClassifier(route_by_language=True, languages_to_route=["en", "es", "it"])


@pytest.mark.integration
def test_langdetect_predict(langdetect_doc_lang_classifier):
    results = langdetect_doc_lang_classifier.predict(documents=DOCUMENTS)
    for doc, expected_language in zip(results, EXPECTED_LANGUAGES):
        assert doc.to_dict()["meta"]["language"] == expected_language


@pytest.mark.integration
def test_langdetect_predict_batch(langdetect_doc_lang_classifier):
    results = langdetect_doc_lang_classifier.predict_batch(documents=[DOCUMENTS, DOCUMENTS[:2]])
    expected_languages = [EXPECTED_LANGUAGES, EXPECTED_LANGUAGES[:2]]
    for lst_docs, lst_expected_languages in zip(results, expected_languages):
        for doc, expected_language in zip(lst_docs, lst_expected_languages):
            assert doc.to_dict()["meta"]["language"] == expected_language


@pytest.mark.integration
def test_langdetect_run_not_route(langdetect_doc_lang_classifier):
    langdetect_doc_lang_classifier.route_by_language = False
    results, edge = langdetect_doc_lang_classifier.run(documents=DOCUMENTS)
    assert edge == "output_1"
    for doc, expected_language in zip(results["documents"], EXPECTED_LANGUAGES):
        assert doc.to_dict()["meta"]["language"] == expected_language


@pytest.mark.integration
def test_langdetect_run_route_fail_on_mixed_languages(langdetect_doc_lang_classifier):
    with pytest.raises(ValueError, match="Documents of multiple languages"):
        langdetect_doc_lang_classifier.run(documents=DOCUMENTS)


@pytest.mark.integration
def test_langdetect_run_batch(langdetect_doc_lang_classifier):
    docs = [[doc] for doc in DOCUMENTS]

    results, split_edge = langdetect_doc_lang_classifier.run_batch(documents=docs)
    assert split_edge == "split"
    for edge, result in results.items():
        document = result["documents"][0][0]
        num_document = DOCUMENTS.index(document)
        expected_language = EXPECTED_LANGUAGES[num_document]
        assert edge == langdetect_doc_lang_classifier._get_edge_from_language(expected_language)
        assert document.to_dict()["meta"]["language"] == expected_language


# TODO: test more the base class
