from typing import Set, List

import sys
from pathlib import Path
import os

import pytest

from haystack import Document
from haystack.nodes.file_converter.pdf import PDFToTextConverter
from haystack.nodes.preprocessor.preprocessor import PreProcessor, ngram, longest_common_ngram

from ..conftest import SAMPLES_PATH


NLTK_TEST_MODELS = SAMPLES_PATH.absolute() / "preprocessor" / "nltk_models"


TEXT = """
This is a sample sentence in paragraph_1. This is a sample sentence in paragraph_1. This is a sample sentence in
paragraph_1. This is a sample sentence in paragraph_1. This is a sample sentence in paragraph_1.\f

This is a sample sentence in paragraph_2. This is a sample sentence in paragraph_2. This is a sample sentence in
paragraph_2. This is a sample sentence in paragraph_2. This is a sample sentence in paragraph_2.

This is a sample sentence in paragraph_3. This is a sample sentence in paragraph_3. This is a sample sentence in
paragraph_3. This is a sample sentence in paragraph_3. This is to trick the test with using an abbreviation\f like Dr.
in the sentence.
"""

HEADLINES = [
    {"headline": "sample sentence in paragraph_1", "start_idx": 11, "level": 0},
    {"headline": "paragraph_1", "start_idx": 198, "level": 1},
    {"headline": "sample sentence in paragraph_2", "start_idx": 223, "level": 0},
    {"headline": "in paragraph_2", "start_idx": 365, "level": 1},
    {"headline": "sample sentence in paragraph_3", "start_idx": 434, "level": 0},
    {"headline": "trick the test", "start_idx": 603, "level": 1},
]

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


@pytest.mark.parametrize(
    "ngram_len,result",
    [
        (1, {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}),
        (3, {"a b c", "b c d", "c d e", "d e f", "e f g", "f g h", "g h i", "h i j"}),
        (5, {"a b c d e", "b c d e f", "c d e f g", "d e f g h", "e f g h i", "f g h i j"}),
        (10, {"a b c d e f g h i j"}),
        (20, {"a b c d e f g h i j"}),
    ],
)
def test_preprocessor_ngram(ngram_len: int, result: Set[str]):
    text = "a b c d e f g h i j"
    ngrams = ngram(text=text, ngram_len=ngram_len)
    assert ngrams == result


@pytest.mark.parametrize(
    "strings,min_ngram_len,max_ngram_len,shared_ngram",
    [
        (["a b c d e", "a b c d e"], 3, 10, {"a b c d e"}),
        (["a b c d e", "a b c d e"], 3, 4, {"a b c d", "b c d e"}),
        (["a b c d e", "e f g h i"], 1, 3, {"e"}),
        (["a b c d e", "d e f g h"], 2, 3, {"d e"}),
        (["a b c d e", "e f g h i"], 2, 3, set()),
        (["a b c d e", "e f g h a"], 1, 3, {"e", "a"}),
    ],
)
def test_preprocessor_longest_shared_ngram(
    strings: List[str], min_ngram_len: int, max_ngram_len: int, shared_ngram: Set[str]
):
    assert shared_ngram == longest_common_ngram(pages=strings, min_ngram=min_ngram_len, max_ngram=max_ngram_len)


@pytest.mark.parametrize(
    "text,clean_text,headlines,clean_headlines",
    [
        # Nothing to clean, no headlines
        ("a\fb\nc", "a\fb\nc", None, None),
        # Trailing newlines and form feeds, no headlines
        ("a\n\fb\nc\f", "a\fb\nc\f", None, None),
        # Nothing to clean, with headlines
        (
            "a\f#Title\nc\f",
            "a\f#Title\nc\f",
            [{"content": "#Title", "start_idx": 2}],
            [{"content": "#Title", "start_idx": 2}],
        ),
        # Single page, no headlines
        (" a \nb c\nd    \n", "a\nb c\nd", None, None),
        # Multiple pages, no headlines
        (" a \f  b\nc     \f", "a\fb\nc\f", None, None),
        # Single page with headlines
        (
            "   #Title \n#Title2   ",
            "#Title\n#Title2",
            [{"content": "#Title", "start_idx": 0}, {"content": "#Title2", "start_idx": 11}],
            [{"content": "#Title", "start_idx": 0}, {"content": "#Title2", "start_idx": 7}],
        ),
        # Multi page with headlines
        (
            "   #Title \f#Title2   ",
            "#Title\f#Title2",
            [{"content": "#Title", "start_idx": 0}, {"content": "#Title2", "start_idx": 11}],
            [{"content": "#Title", "start_idx": 0}, {"content": "#Title2", "start_idx": 7}],
        ),
        # Empty page with headlines
        (
            "   #Title \f\f\f#Title2   ",
            "#Title\f\f\f#Title2",
            [{"content": "#Title", "start_idx": 0}, {"content": "#Title2", "start_idx": 13}],
            [{"content": "#Title", "start_idx": 0}, {"content": "#Title2", "start_idx": 9}],
        ),
        # With multiple pages, headlines and text
        (
            " a  \n#Title \n\f d  \n #Title2 \n f",
            "a\n#Title\fd\n#Title2\nf",
            [{"content": "#Title", "start_idx": 5}, {"content": "#Title2", "start_idx": 18}],
            [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 11}],
        ),
        # Unsorted headlines will be sorted
        (
            " a  \n#Title \f d  \n #Title2 \n f",
            "a\n#Title\fd\n#Title2\nf",
            [{"content": "#Title2", "start_idx": 18}, {"content": "#Title", "start_idx": 5}],
            [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 11}],
        ),
        # With headlines and multiple empty lines
        (
            "\n\n a \n#Title \n\n\n d  \n\n\n\n",
            "\n\na\n#Title\n\n\nd\n\n\n",
            [{"content": "#Title", "start_idx": 6}],
            [{"content": "#Title", "start_idx": 4}],
        ),
        # With headlines and multiple empty lines/pages
        (
            "\n\n a \n#Title \f\f\f d  \n #Title2 \f\n\n\n\n",
            "\n\na\n#Title\f\f\fd\n#Title2\f\n\n\n",
            [{"content": "#Title2", "start_idx": 21}, {"content": "#Title", "start_idx": 6}],
            [{"content": "#Title", "start_idx": 4}, {"content": "#Title2", "start_idx": 15}],
        ),
    ],
)
def test_preprocessor_remove_whitespace(text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = PreProcessor().remove_whitespace(doc_to_clean)

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


@pytest.mark.parametrize(
    "text,clean_text,headlines,clean_headlines",
    [
        # Nothing to clean, no headlines
        ("a\fb\nc\f", "a\fb\nc\f", None, None),
        # Nothing to clean, with headlines
        (
            "a\f#Title\nc\n\f",
            "a\f#Title\nc\f",
            [{"content": "#Title", "start_idx": 2}],
            [{"content": "#Title", "start_idx": 2}],
        ),
        # Single page, no headlines
        ("\n\na\n\n\nb\n", "a\nb", None, None),
        # multiple pages, no headlines
        ("\n\na\n\n\fb\n\n\n\nc\n\n\f\f\f", "a\fb\nc\f\f\f", None, None),
        # Single page with headlines
        (
            "\n\n#Title\n\n\n\n#Title2",
            "#Title\n#Title2",
            [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 12}],
            [{"content": "#Title", "start_idx": 0}, {"content": "#Title2", "start_idx": 7}],
        ),
        # Multi page with headlines
        (
            "\n\n#Title\n\n\n\n\f#Title2\n\f\n",
            "#Title\f#Title2\f",
            [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 12}],
            [{"content": "#Title", "start_idx": 0}, {"content": "#Title2", "start_idx": 7}],
        ),
        # With multiple pages, headlines and text
        (
            "a\n\n#Title\n\n\n\nb c\n\f#Title2\n\f\n",
            "a\n#Title\nb c\f#Title2\f",
            [{"content": "#Title", "start_idx": 3}, {"content": "#Title2", "start_idx": 17}],
            [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 13}],
        ),
        # Unsorted headlines will be sorted
        (
            "a\n\n#Title\n\n\n\nb c\n\f#Title2\n\f\n",
            "a\n#Title\nb c\f#Title2\f",
            [{"content": "#Title2", "start_idx": 17}, {"content": "#Title", "start_idx": 3}],
            [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 13}],
        ),
    ],
)
def test_preprocessor_remove_empty_lines(text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = PreProcessor().remove_empty_lines(doc_to_clean)

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


@pytest.mark.parametrize(
    "substrings,text,clean_text,headlines,clean_headlines",
    [
        (["AA"], "AAAAabcdAAAefgAhAA", "abcdAefgAh", None, None),
        (["🪲"], "abcd🪲efgh🪲", "abcdefgh", None, None),
        (["🪲"], "✨abc✨d🪲ef✨gh🪲", "✨abc✨def✨gh", None, None),
        (["🪲"], "\b\b\babc\n\nde🪲\f\nfgh", "\b\b\babc\n\nde\f\nfgh", None, None),
        (["🪲", "A"], "aA🪲bA🪲", "ab", None, None),
        (["\n"], "a\nb\n", "ab", None, None),
        # header/footer removal example
        (["a.\nb c.\n", "c.\nd.\n"], "a.\nb c.\ncde fg hi c.\nd.\n", "cde fg hi ", None, None),
        (
            ["bb"],
            "a # Test header bb cbb def",
            "a # Test header  c def",
            [{"content": "# Test header", "start_idx": 2}],
            [{"content": "# Test header", "start_idx": 2}],
        ),
        (
            ["bb"],
            "bb # Test header a cbb def",
            " # Test header a c def",
            [{"content": "# Test header", "start_idx": 3}],
            [{"content": "# Test header", "start_idx": 1}],
        ),
        (
            ["Test"],
            "a # Test header b c def",
            "a #  header b c def",
            [{"content": "# Test header", "start_idx": 2}],
            [{"content": "# Test header", "start_idx": 2}],
        ),
        (
            ["Test"],
            "a # Test header 1 b c # Test header 2 def",
            "a #  header 1 b c #  header 2 def",
            [{"content": "# Test header 2", "start_idx": 22}, {"content": "# Test header 1", "start_idx": 2}],
            [{"content": "# Test header 1", "start_idx": 2}, {"content": "# Test header 2", "start_idx": 18}],
        ),
        # header/footer removal example
        (
            ["the header", "the footer", "Test"],
            "the header\na\n# Test header 1\nb c\n# Test header 2\ndef\nthe footer",
            "\na\n#  header 1\nb c\n#  header 2\ndef\n",
            [{"content": "# Test header 2", "start_idx": 33}, {"content": "# Test header 1", "start_idx": 13}],
            [{"content": "# Test header 1", "start_idx": 3}, {"content": "# Test header 2", "start_idx": 19}],
        ),
    ],
)
def test_preprocessor_remove_substrings(substrings, text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = PreProcessor().remove_substrings(doc_to_clean, substrings=substrings)

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


@pytest.mark.parametrize(
    "text,clean_text,headlines,clean_headlines",
    [("I'm a header. Hello! I'm a footer.\fI'm a header. Hi! I'm a footer\f", "Hello!\fHello 2!\f", None, None)],
)
def test_preprocessor_remove_header_footer(text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = PreProcessor().remove_header_footer(doc_to_clean)

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


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
            assert len(doc.content.split()) == 14
        assert len(doc.content.split()) <= 15 or doc.content.startswith("This is to trick")
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


def test_headline_processing_split_by_word():
    expected_headlines = [
        [{"headline": "sample sentence in paragraph_1", "start_idx": 11, "level": 0}],
        [
            {"headline": "sample sentence in paragraph_1", "start_idx": None, "level": 0},
            {"headline": "paragraph_1", "start_idx": 19, "level": 1},
            {"headline": "sample sentence in paragraph_2", "start_idx": 44, "level": 0},
            {"headline": "in paragraph_2", "start_idx": 186, "level": 1},
        ],
        [
            {"headline": "sample sentence in paragraph_2", "start_idx": None, "level": 0},
            {"headline": "in paragraph_2", "start_idx": None, "level": 1},
            {"headline": "sample sentence in paragraph_3", "start_idx": 53, "level": 0},
        ],
        [
            {"headline": "sample sentence in paragraph_3", "start_idx": None, "level": 0},
            {"headline": "trick the test", "start_idx": 36, "level": 1},
        ],
    ]

    document = Document(content=TEXT, meta={"headlines": HEADLINES})
    preprocessor = PreProcessor(
        split_length=30, split_overlap=0, split_by="word", split_respect_sentence_boundary=False
    )
    documents = preprocessor.process(document)

    for doc, expected in zip(documents, expected_headlines):
        assert doc.meta["headlines"] == expected


def test_headline_processing_split_by_word_overlap():
    expected_headlines = [
        [{"headline": "sample sentence in paragraph_1", "start_idx": 11, "level": 0}],
        [
            {"headline": "sample sentence in paragraph_1", "start_idx": None, "level": 0},
            {"headline": "paragraph_1", "start_idx": 71, "level": 1},
            {"headline": "sample sentence in paragraph_2", "start_idx": 96, "level": 0},
        ],
        [
            {"headline": "sample sentence in paragraph_2", "start_idx": None, "level": 0},
            {"headline": "in paragraph_2", "start_idx": 110, "level": 1},
            {"headline": "sample sentence in paragraph_3", "start_idx": 179, "level": 0},
        ],
        [
            {"headline": "sample sentence in paragraph_2", "start_idx": None, "level": 0},
            {"headline": "in paragraph_2", "start_idx": None, "level": 1},
            {"headline": "sample sentence in paragraph_3", "start_idx": 53, "level": 0},
        ],
        [
            {"headline": "sample sentence in paragraph_3", "start_idx": None, "level": 0},
            {"headline": "trick the test", "start_idx": 95, "level": 1},
        ],
    ]

    document = Document(content=TEXT, meta={"headlines": HEADLINES})
    preprocessor = PreProcessor(
        split_length=30, split_overlap=10, split_by="word", split_respect_sentence_boundary=False
    )
    documents = preprocessor.process(document)

    for doc, expected in zip(documents, expected_headlines):
        assert doc.meta["headlines"] == expected


def test_headline_processing_split_by_word_respect_sentence_boundary():
    expected_headlines = [
        [{"headline": "sample sentence in paragraph_1", "start_idx": 11, "level": 0}],
        [
            {"headline": "sample sentence in paragraph_1", "start_idx": None, "level": 0},
            {"headline": "paragraph_1", "start_idx": 71, "level": 1},
            {"headline": "sample sentence in paragraph_2", "start_idx": 96, "level": 0},
        ],
        [
            {"headline": "sample sentence in paragraph_2", "start_idx": None, "level": 0},
            {"headline": "in paragraph_2", "start_idx": 110, "level": 1},
        ],
        [
            {"headline": "sample sentence in paragraph_2", "start_idx": None, "level": 0},
            {"headline": "in paragraph_2", "start_idx": None, "level": 1},
            {"headline": "sample sentence in paragraph_3", "start_idx": 53, "level": 0},
        ],
        [
            {"headline": "sample sentence in paragraph_3", "start_idx": None, "level": 0},
            {"headline": "trick the test", "start_idx": 95, "level": 1},
        ],
    ]

    document = Document(content=TEXT, meta={"headlines": HEADLINES})
    preprocessor = PreProcessor(split_length=30, split_overlap=5, split_by="word", split_respect_sentence_boundary=True)
    documents = preprocessor.process(document)

    for doc, expected in zip(documents, expected_headlines):
        assert doc.meta["headlines"] == expected


def test_headline_processing_split_by_sentence():
    expected_headlines = [
        [
            {"headline": "sample sentence in paragraph_1", "start_idx": 11, "level": 0},
            {"headline": "paragraph_1", "start_idx": 198, "level": 1},
        ],
        [
            {"headline": "sample sentence in paragraph_1", "start_idx": None, "level": 0},
            {"headline": "paragraph_1", "start_idx": None, "level": 1},
            {"headline": "sample sentence in paragraph_2", "start_idx": 10, "level": 0},
            {"headline": "in paragraph_2", "start_idx": 152, "level": 1},
        ],
        [
            {"headline": "sample sentence in paragraph_2", "start_idx": None, "level": 0},
            {"headline": "in paragraph_2", "start_idx": None, "level": 1},
            {"headline": "sample sentence in paragraph_3", "start_idx": 10, "level": 0},
            {"headline": "trick the test", "start_idx": 179, "level": 1},
        ],
    ]

    document = Document(content=TEXT, meta={"headlines": HEADLINES})
    preprocessor = PreProcessor(
        split_length=5, split_overlap=0, split_by="sentence", split_respect_sentence_boundary=False
    )
    documents = preprocessor.process(document)

    for doc, expected in zip(documents, expected_headlines):
        assert doc.meta["headlines"] == expected


def test_headline_processing_split_by_sentence_overlap():
    expected_headlines = [
        [
            {"headline": "sample sentence in paragraph_1", "start_idx": 11, "level": 0},
            {"headline": "paragraph_1", "start_idx": 198, "level": 1},
        ],
        [
            {"headline": "sample sentence in paragraph_1", "start_idx": None, "level": 0},
            {"headline": "paragraph_1", "start_idx": 29, "level": 1},
            {"headline": "sample sentence in paragraph_2", "start_idx": 54, "level": 0},
            {"headline": "in paragraph_2", "start_idx": 196, "level": 1},
        ],
        [
            {"headline": "sample sentence in paragraph_2", "start_idx": None, "level": 0},
            {"headline": "in paragraph_2", "start_idx": 26, "level": 1},
            {"headline": "sample sentence in paragraph_3", "start_idx": 95, "level": 0},
        ],
        [
            {"headline": "sample sentence in paragraph_3", "start_idx": None, "level": 0},
            {"headline": "trick the test", "start_idx": 95, "level": 1},
        ],
    ]

    document = Document(content=TEXT, meta={"headlines": HEADLINES})
    preprocessor = PreProcessor(
        split_length=5, split_overlap=1, split_by="sentence", split_respect_sentence_boundary=False
    )
    documents = preprocessor.process(document)

    for doc, expected in zip(documents, expected_headlines):
        assert doc.meta["headlines"] == expected


def test_headline_processing_split_by_passage():
    expected_headlines = [
        [
            {"headline": "sample sentence in paragraph_1", "start_idx": 11, "level": 0},
            {"headline": "paragraph_1", "start_idx": 198, "level": 1},
        ],
        [
            {"headline": "sample sentence in paragraph_1", "start_idx": None, "level": 0},
            {"headline": "paragraph_1", "start_idx": None, "level": 1},
            {"headline": "sample sentence in paragraph_2", "start_idx": 10, "level": 0},
            {"headline": "in paragraph_2", "start_idx": 152, "level": 1},
        ],
        [
            {"headline": "sample sentence in paragraph_2", "start_idx": None, "level": 0},
            {"headline": "in paragraph_2", "start_idx": None, "level": 1},
            {"headline": "sample sentence in paragraph_3", "start_idx": 10, "level": 0},
            {"headline": "trick the test", "start_idx": 179, "level": 1},
        ],
    ]

    document = Document(content=TEXT, meta={"headlines": HEADLINES})
    preprocessor = PreProcessor(
        split_length=1, split_overlap=0, split_by="passage", split_respect_sentence_boundary=False
    )
    documents = preprocessor.process(document)

    for doc, expected in zip(documents, expected_headlines):
        assert doc.meta["headlines"] == expected


def test_headline_processing_split_by_passage_overlap():
    expected_headlines = [
        [
            {"headline": "sample sentence in paragraph_1", "start_idx": 11, "level": 0},
            {"headline": "paragraph_1", "start_idx": 198, "level": 1},
            {"headline": "sample sentence in paragraph_2", "start_idx": 223, "level": 0},
            {"headline": "in paragraph_2", "start_idx": 365, "level": 1},
        ],
        [
            {"headline": "sample sentence in paragraph_1", "start_idx": None, "level": 0},
            {"headline": "paragraph_1", "start_idx": None, "level": 1},
            {"headline": "sample sentence in paragraph_2", "start_idx": 10, "level": 0},
            {"headline": "in paragraph_2", "start_idx": 152, "level": 1},
            {"headline": "sample sentence in paragraph_3", "start_idx": 221, "level": 0},
            {"headline": "trick the test", "start_idx": 390, "level": 1},
        ],
    ]

    document = Document(content=TEXT, meta={"headlines": HEADLINES})
    preprocessor = PreProcessor(
        split_length=2, split_overlap=1, split_by="passage", split_respect_sentence_boundary=False
    )
    documents = preprocessor.process(document)

    for doc, expected in zip(documents, expected_headlines):
        assert doc.meta["headlines"] == expected
