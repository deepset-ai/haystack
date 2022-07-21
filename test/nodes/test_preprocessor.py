import sys
from pathlib import Path
import os

import pytest

from haystack import Document
from haystack.nodes.file_converter.pdf import PDFToTextConverter
from haystack.nodes.preprocessor.preprocessor import PreProcessor

from ..conftest import SAMPLES_PATH


NLTK_TEST_MODELS = SAMPLES_PATH.absolute() / "preprocessor" / "nltk_models"


TEXT = """
This is a sample sentence in paragraph_1. This is a sample sentence in paragraph_1. This is a sample sentence in 
paragraph_1. This is a sample sentence in paragraph_1. This is a sample sentence in paragraph_1.

This is a sample sentence in paragraph_2. This is a sample sentence in paragraph_2. This is a sample sentence in 
paragraph_2. This is a sample sentence in paragraph_2. This is a sample sentence in paragraph_2.

This is a sample sentence in paragraph_3. This is a sample sentence in paragraph_3. This is a sample sentence in 
paragraph_3. This is a sample sentence in paragraph_3. This is to trick the test with using an abbreviation like Dr. 
in the sentence. 
"""

LEGAL_TEXT_PT = """
A Lei nº 9.514/1997, que instituiu a alienação fiduciária de
bens imóveis, é norma especial e posterior ao Código de Defesa do
Consumidor – CDC. Em tais circunstâncias, o inadimplemento do
devedor fiduciante enseja a aplicação da regra prevista nos arts. 26 e 27
da lei especial” (REsp 1.871.911/SP, rel. Min. Nancy Andrighi, DJe
25/8/2020).

A Emenda Constitucional n. 35 alterou substancialmente esse mecanismo,
ao determinar, na nova redação conferida ao art. 53: “§ 3º Recebida a
denúncia contra o Senador ou Deputado, por crime ocorrido após a
diplomação, o Supremo Tribunal Federal dará ciência à Casa respectiva, que,
por iniciativa de partido político nela representado e pelo voto da maioria de
seus membros, poderá, até a decisão final, sustar o andamento da ação”.
Vale ressaltar, contudo, que existem, antes do encaminhamento ao
Presidente da República, os chamados autógrafos. Os autógrafos ocorrem já
com o texto definitivamente aprovado pelo Plenário ou pelas comissões,
quando for o caso. Os autógrafos devem reproduzir com absoluta fidelidade a
redação final aprovada. O projeto aprovado será encaminhado em autógrafos
ao Presidente da República. O tema encontra-se regulamentado pelo art. 200
do RICD e arts. 328 a 331 do RISF.
"""


@pytest.mark.parametrize("split_length_and_results", [(1, 15), (10, 2)])
def test_preprocess_sentence_split(split_length_and_results):
    split_length, expected_documents_count = split_length_and_results

    document = Document(content=TEXT)
    preprocessor = PreProcessor(
        split_length=split_length, split_overlap=0, split_by="sentence", split_respect_sentence_boundary=False
    )
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count


@pytest.mark.parametrize("split_length_and_results", [(1, 15), (10, 2)])
def test_preprocess_sentence_split_custom_models_wrong_file_format(split_length_and_results):
    split_length, expected_documents_count = split_length_and_results

    document = Document(content=TEXT)
    preprocessor = PreProcessor(
        split_length=split_length,
        split_overlap=0,
        split_by="sentence",
        split_respect_sentence_boundary=False,
        tokenizer_model_folder=NLTK_TEST_MODELS / "wrong",
        language="en",
    )
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count


@pytest.mark.parametrize("split_length_and_results", [(1, 15), (10, 2)])
def test_preprocess_sentence_split_custom_models_non_default_language(split_length_and_results):
    split_length, expected_documents_count = split_length_and_results

    document = Document(content=TEXT)
    preprocessor = PreProcessor(
        split_length=split_length,
        split_overlap=0,
        split_by="sentence",
        split_respect_sentence_boundary=False,
        language="ca",
    )
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count


@pytest.mark.parametrize("split_length_and_results", [(1, 8), (8, 1)])
def test_preprocess_sentence_split_custom_models(split_length_and_results):
    split_length, expected_documents_count = split_length_and_results

    document = Document(content=LEGAL_TEXT_PT)
    preprocessor = PreProcessor(
        split_length=split_length,
        split_overlap=0,
        split_by="sentence",
        split_respect_sentence_boundary=False,
        language="pt",
        tokenizer_model_folder=NLTK_TEST_MODELS,
    )
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count


def test_preprocess_word_split():
    document = Document(content=TEXT)
    preprocessor = PreProcessor(
        split_length=10, split_overlap=0, split_by="word", split_respect_sentence_boundary=False
    )
    documents = preprocessor.process(document)
    assert len(documents) == 11

    preprocessor = PreProcessor(split_length=15, split_overlap=0, split_by="word", split_respect_sentence_boundary=True)
    documents = preprocessor.process(document)
    for i, doc in enumerate(documents):
        if i == 0:
            assert len(doc.content.split(" ")) == 14
        assert len(doc.content.split(" ")) <= 15 or doc.content.startswith("This is to trick")
    assert len(documents) == 8

    preprocessor = PreProcessor(
        split_length=40, split_overlap=10, split_by="word", split_respect_sentence_boundary=True
    )
    documents = preprocessor.process(document)
    assert len(documents) == 5

    preprocessor = PreProcessor(split_length=5, split_overlap=0, split_by="word", split_respect_sentence_boundary=True)
    documents = preprocessor.process(document)
    assert len(documents) == 15


@pytest.mark.parametrize("split_length_and_results", [(1, 3), (2, 2)])
def test_preprocess_passage_split(split_length_and_results):
    split_length, expected_documents_count = split_length_and_results

    document = Document(content=TEXT)
    preprocessor = PreProcessor(
        split_length=split_length, split_overlap=0, split_by="passage", split_respect_sentence_boundary=False
    )
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="FIXME Footer not detected correctly on Windows")
def test_clean_header_footer():
    converter = PDFToTextConverter()
    document = converter.convert(
        file_path=Path(SAMPLES_PATH / "pdf" / "sample_pdf_2.pdf")
    )  # file contains header/footer

    preprocessor = PreProcessor(clean_header_footer=True, split_by=None)
    documents = preprocessor.process(document)

    assert len(documents) == 1

    assert "This is a header." not in documents[0].content
    assert "footer" not in documents[0].content


def test_remove_substrings():
    document = Document(content="This is a header. Some additional text. wiki. Some emoji ✨ 🪲 Weird whitespace\b\b\b.")

    # check that the file contains the substrings we are about to remove
    assert "This is a header." in document.content
    assert "wiki" in document.content
    assert "🪲" in document.content
    assert "whitespace" in document.content
    assert "✨" in document.content

    preprocessor = PreProcessor(remove_substrings=["This is a header.", "wiki", "🪲"])
    documents = preprocessor.process(document)

    assert "This is a header." not in documents[0].content
    assert "wiki" not in documents[0].content
    assert "🪲" not in documents[0].content
    assert "whitespace" in documents[0].content
    assert "✨" in documents[0].content


def test_id_hash_keys_from_pipeline_params():
    document_1 = Document(content="This is a document.", meta={"key": "a"})
    document_2 = Document(content="This is a document.", meta={"key": "b"})
    assert document_1.id == document_2.id

    preprocessor = PreProcessor(split_length=2, split_respect_sentence_boundary=False)
    output, _ = preprocessor.run(documents=[document_1, document_2], id_hash_keys=["content", "meta"])
    documents = output["documents"]
    unique_ids = set(d.id for d in documents)

    assert len(documents) == 4
    assert len(unique_ids) == 4
