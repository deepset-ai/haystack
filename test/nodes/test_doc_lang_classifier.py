import pytest
import logging

from haystack.schema import Document
from haystack.nodes.doc_language_classifier import LangdetectDocumentLanguageClassifier

DOCUMENTS = [
    Document(content="My name is Matteo and I live in Rome"),
    Document(content="Mi chiamo Matteo e vivo a Roma"),
    Document(content="Mi nombre es Matteo y vivo en Roma"),
]

EXPECTED_LANGUAGES = ["en", "it", "es"]


@pytest.fixture(params=["langdetect", "transformers"])
def doclangclassifier(request):
    if request.param == "langdetect":
        return LangdetectDocumentLanguageClassifier(route_by_language=True, languages_to_route=["en", "es", "it"])
    # replace
    return LangdetectDocumentLanguageClassifier(route_by_language=True, languages_to_route=["en", "es", "it"])


@pytest.mark.integration
@pytest.mark.parametrize("doclangclassifier", ["langdetect"], indirect=True)
def test_doclangclassifier_predict(doclangclassifier):
    results = doclangclassifier.predict(documents=DOCUMENTS)
    for doc, expected_language in zip(results, EXPECTED_LANGUAGES):
        assert doc.to_dict()["meta"]["language"] == expected_language


@pytest.mark.integration
@pytest.mark.parametrize("doclangclassifier", ["langdetect"], indirect=True)
def test_doclangclassifier_predict_batch(doclangclassifier):
    results = doclangclassifier.predict_batch(documents=[DOCUMENTS, DOCUMENTS[:2]])
    expected_languages = [EXPECTED_LANGUAGES, EXPECTED_LANGUAGES[:2]]
    for lst_docs, lst_expected_languages in zip(results, expected_languages):
        for doc, expected_language in zip(lst_docs, lst_expected_languages):
            assert doc.to_dict()["meta"]["language"] == expected_language


@pytest.mark.integration
@pytest.mark.parametrize("doclangclassifier", ["langdetect"], indirect=True)
def test_doclangclassifier_run_not_route(doclangclassifier):
    doclangclassifier.route_by_language = False
    results, edge = doclangclassifier.run(documents=DOCUMENTS)
    assert edge == "output_1"
    for doc, expected_language in zip(results["documents"], EXPECTED_LANGUAGES):
        assert doc.to_dict()["meta"]["language"] == expected_language


@pytest.mark.integration
@pytest.mark.parametrize("doclangclassifier", ["langdetect"], indirect=True)
def test_doclangclassifier_run_route_fail_on_mixed_languages(doclangclassifier):
    with pytest.raises(ValueError, match="Documents of multiple languages"):
        doclangclassifier.run(documents=DOCUMENTS)


@pytest.mark.integration
@pytest.mark.parametrize("doclangclassifier", ["langdetect"], indirect=True)
def test_doclangclassifier_run_route_cannot_detect_language(doclangclassifier, caplog):
    doc_unidentifiable_lang = Document("01234, 56789, ")
    with caplog.at_level(logging.WARNING):
        results, edge = doclangclassifier.run(documents=[doc_unidentifiable_lang])
        assert "The model cannot detect the language of any of the documents." in caplog.text
    assert edge == "output_1"
    assert results["documents"][0].to_dict()["meta"]["language"] is None


@pytest.mark.integration
@pytest.mark.parametrize("doclangclassifier", ["langdetect"], indirect=True)
def test_doclangclassifier_run_route_fail_on_language_not_in_list(doclangclassifier, caplog):
    doc_other_lang = Document("Meu nome é Matteo e moro em Roma")
    with pytest.raises(ValueError, match="is not in the list of languages to route"):
        doclangclassifier.run(documents=[doc_other_lang])


@pytest.mark.integration
@pytest.mark.parametrize("doclangclassifier", ["langdetect"], indirect=True)
def test_doclangclassifier_run_batch(doclangclassifier):
    docs = [[doc] for doc in DOCUMENTS]
    results, split_edge = doclangclassifier.run_batch(documents=docs)
    assert split_edge == "split"
    for edge, result in results.items():
        document = result["documents"][0][0]
        num_document = DOCUMENTS.index(document)
        expected_language = EXPECTED_LANGUAGES[num_document]
        assert edge == doclangclassifier._get_edge_from_language(expected_language)
        assert document.to_dict()["meta"]["language"] == expected_language


# TODO: test more the base class
