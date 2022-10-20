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


This is a sample sentence in paragraph_1 1. This is a sample sentence in paragraph_1 2. This is a sample sentence in
paragraph_1 3. This is a sample sentence in paragraph_1 4. This is a sample sentence in paragraph_1 5.\f

This is a sample sentence in paragraph_2 1. This is a sample sentence in paragraph_2 2. This is a sample sentence in
paragraph_2 3. This is a sample sentence in paragraph_2 4. This is a sample sentence in paragraph_2 5.

This is a sample sentence in paragraph_3 1. This is a sample sentence in paragraph_3 2. This is a sample sentence in
paragraph_3 3. This is a sample sentence in paragraph_3 4. This is to trick the test with using an abbreviation\f like Dr.
in the sentence 5 extra words.
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

# TODO: Add tests for PDF with both tika and pdf converter - check for page break, remove line break, merge etc...

# Cleaning Tests
@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="FIXME Footer not detected correctly on Windows")
def test_clean_header_footer():
    converter = PDFToTextConverter()
    document = converter.convert(
        file_path=Path(SAMPLES_PATH / "pdf" / "sample_pdf_2.pdf")
    )  # file contains header/footer

    # TODO: add arguments/variations for different n_chars, n_first_pages_to_ignore, n_last_pages_to_ignore
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
    documents = preprocessor.process([document])

    assert "This is a header." not in documents[0].content
    assert "wiki" not in documents[0].content
    assert "🪲" not in documents[0].content
    assert "whitespace" in documents[0].content
    assert "✨" in documents[0].content


# TODO: add tests
## Clean Empty Lines

## Remove Numeric Tables

## Remove Line Breaks


# Split Tests
@pytest.mark.parametrize("split_length_and_results", [(1, 3), (2, 2)])
def test_preprocess_passage_split(split_length_and_results):
    split_length, expected_documents_count = split_length_and_results

    document = Document(content=TEXT)
    preprocessor = PreProcessor(
        split_length=split_length, split_overlap=0, split_by="passage", split_respect_sentence_boundary=False
    )
    documents = preprocessor.process([document])
    assert len(documents) == expected_documents_count

    # TODO: Add tests for merge_short and merge_lowercase. Make sure the test data has appropriate examples of short text and lines ending with punctuation
    # TODO: add split_overlap and split_length tests for passage split


@pytest.mark.parametrize(
    "split_length_and_results",
    [
        (True, 1, 0, 15),
        (False, 1, 0, 15),
        (True, 2, 0, 9),
        (False, 2, 0, 8),
        (True, 2, 1, 12),
        (False, 2, 1, 14),
        (True, 3, 1, 6),
        (False, 3, 1, 7),
    ],
)
def test_preprocess_sentence_split(split_length_and_results):
    pre_split_paragraphs, split_length, split_overlap, expected_documents_count = split_length_and_results

    document = Document(content=TEXT)
    preprocessor = PreProcessor(
        pre_split_paragraphs=pre_split_paragraphs,
        split_length=split_length,
        split_overlap=split_overlap,
        split_by="sentence",
        split_respect_sentence_boundary=False,
    )
    documents = preprocessor.process([document])
    assert len(documents) == expected_documents_count


@pytest.mark.parametrize(
    "params",
    [
        (True, 6, 0, False, 23),
        (False, 6, 0, False, 22),
        (True, 6, 0, True, 15),
        (False, 6, 0, True, 15),
        (True, 12, 0, False, 13),
        (False, 12, 0, False, 11),
        (True, 12, 0, True, 15),
        (False, 12, 0, True, 15),
        (True, 40, 10, False, 4),
        (False, 40, 10, False, 4),
        (True, 40, 10, True, 4),
        (False, 40, 10, True, 5),
    ],
)
def test_preprocess_word_split(params):
    (
        pre_split_paragraphs,
        split_length,
        split_overlap,
        split_respect_sentence_boundary,
        expected_documents_count,
    ) = params

    document = Document(content=TEXT)
    preprocessor = PreProcessor(
        pre_split_paragraphs=pre_split_paragraphs,
        split_length=split_length,
        split_overlap=split_overlap,
        split_by="word",
        split_respect_sentence_boundary=split_respect_sentence_boundary,
    )
    documents = preprocessor.process([document])
    assert len(documents) == expected_documents_count


@pytest.mark.parametrize(
    "test_input", [(10, 0, True, 5, 14), (10, 0, False, 4, 13), (10, 5, True, 5, 14), (10, 5, False, 8, 25)]
)
def test_page_number_extraction(test_input):
    split_length, overlap, resp_sent_boundary, exp_page2_index, exp_page3_index = test_input
    preprocessor = PreProcessor(
        add_page_number=True,
        split_by="word",
        split_length=split_length,
        split_overlap=overlap,
        split_respect_sentence_boundary=resp_sent_boundary,
    )
    document = Document(content=TEXT)
    documents = preprocessor.process([document])
    for idx, doc in enumerate(documents):

        if idx < exp_page2_index:
            assert doc.meta["page"] == 1
        elif idx < exp_page3_index:
            assert doc.meta["page"] == 2
        else:
            assert doc.meta["page"] == 3




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
    documents = preprocessor.process([document])
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
    documents = preprocessor.process([document])
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
    documents = preprocessor.process([document])
    assert len(documents) == expected_documents_count


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


# test_input is a tuple consisting of the parameters for split_length, split_overlap and split_respect_sentence_boundary
# and the expected index in the output list of Documents where the page number changes from 1 to 2
@pytest.mark.parametrize("test_input", [(10, 0, True, 5), (10, 0, False, 4), (10, 5, True, 6), (10, 5, False, 7)])
def test_page_number_extraction(test_input):
    split_length, overlap, resp_sent_boundary, exp_doc_index = test_input
    preprocessor = PreProcessor(
        add_page_number=True,
        split_by="word",
        split_length=split_length,
        split_overlap=overlap,
        split_respect_sentence_boundary=resp_sent_boundary,
    )
    document = Document(content=TEXT)
    documents = preprocessor.process(document)
    for idx, doc in enumerate(documents):
        if idx < exp_doc_index:
            assert doc.meta["page"] == 1
        else:
            assert doc.meta["page"] == 2


def test_page_number_extraction_on_empty_pages():
    """
    Often "marketing" documents contain pages without text (visuals only). When extracting page numbers, these pages should be counted as well to avoid
    issues when mapping results back to the original document.
    """
    preprocessor = PreProcessor(add_page_number=True, split_by="word", split_length=7, split_overlap=0)
    text_page_one = "This is a text on page one."
    text_page_three = "This is a text on page three."
    # this is what we get from PDFToTextConverter in case of an "empty" page
    document_with_empty_pages = f"{text_page_one}\f\f{text_page_three}"
    document = Document(content=document_with_empty_pages)

    documents = preprocessor.process(document)

    assert documents[0].meta["page"] == 1
    assert documents[1].meta["page"] == 3

    # verify the placeholder for the empty page has been removed
    assert documents[0].content.strip() == text_page_one
    assert documents[1].content.strip() == text_page_three


def test_substitute_page_break():
    # Page breaks at the end of sentences should be replaced by "[NEW_PAGE]", while page breaks in between of
    # sentences should not be replaced.
    result = PreProcessor._substitute_page_breaks(TEXT)
    assert result[223:233] == "[NEW_PAGE]"
    assert result[684] == "\f"


def test_split_paragraphs():
    # pre_split_paragraphs, split_length, split_overlap, split_by, split_respect_boundary, merge_short, expected_documents_count = split_length_and_results
    text = """This is a long paragraph 1. This is a long paragraph 2. 
    
    Short para.
    """
    document = Document(content=text)
    preprocessor = PreProcessor(merge_short=12)
    result = preprocessor.process([document])
    for doc in result:
        print(doc.content)
    assert len(result) == 1

    text = """This is a senten-
    ce that spans two lines.
    
    This is a sentence\f

    that spans two pages.
    """
    document = Document(content=text)
    preprocessor = PreProcessor(
        pre_split_paragraphs=True, split_by="word", split_length=1, split_respect_sentence_boundary=True
    )
    result = preprocessor.process([document])
    for doc in result:
        print(doc.content)
    assert result[0].content.find("-") == -1
    assert result[1].content == ("This is a sentence that spans two pages.")
