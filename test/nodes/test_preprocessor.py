import sys
from pathlib import Path
from typing import Any, Optional, List
from unittest import mock
from unittest.mock import Mock

import nltk.data
import pytest
import tiktoken
from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempPathFactory
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from haystack import Document
from haystack.nodes.file_converter.pdf_xpdf import PDFToTextConverter
from haystack.nodes.preprocessor.preprocessor import PreProcessor


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


@pytest.fixture(scope="module")
def module_tmp_dir(tmp_path_factory: TempPathFactory) -> Path:
    """Module fixture to avoid that the model data is downloaded for each test."""
    return tmp_path_factory.mktemp("nltk_data")


@pytest.fixture(autouse=True)
def patched_nltk_data_path(module_tmp_dir: Path, monkeypatch: MonkeyPatch, tmp_path: Path) -> Path:
    """Patch the NLTK data path to use a temporary directory instead of a local, persistent directory."""
    old_find = nltk.data.find

    def patched_find(resource_name: str, paths: Optional[List[str]] = None) -> str:
        return old_find(resource_name, paths=[str(tmp_path)])

    monkeypatch.setattr(nltk.data, nltk.data.find.__name__, patched_find)

    old_download = nltk.download

    def patched_download(*args: Any, **kwargs: Any) -> bool:
        return old_download(*args, **kwargs, download_dir=str(tmp_path))

    monkeypatch.setattr(nltk, nltk.download.__name__, patched_download)

    return tmp_path


@pytest.fixture
def mock_huggingface_tokenizer():
    class MockTokenizer(PreTrainedTokenizerBase):
        """Simple Mock tokenizer splitting the text into 2-character chunks."""

        @staticmethod
        def tokenize(text, **kwargs):
            return [text[i : i + 2] for i in range(0, len(text), 2)]

        @staticmethod
        def encode_plus(text, **kwargs):
            return Mock(offset_mapping=[(i, min(len(text), i + 2)) for i in range(0, len(text), 2)])

    mock_tokenizer_instance = MockTokenizer()

    with mock.patch.object(AutoTokenizer, "from_pretrained", return_value=mock_tokenizer_instance):
        yield mock_tokenizer_instance


@pytest.fixture
def mock_tiktoken_tokenizer():
    class MockTokenizer:
        """Simple Mock tokenizer "encoding" the text into a 0 for every 5-character chunk."""

        @staticmethod
        def encode(text, **kwargs):
            return [0 for i in range(0, len(text), 5)]

        @staticmethod
        def decode_single_token_bytes(token):
            return b"mock "

    mock_tokenizer_instance = MockTokenizer()

    with mock.patch.object(tiktoken, "get_encoding", return_value=mock_tokenizer_instance):
        yield mock_tokenizer_instance


@pytest.mark.unit
@pytest.mark.parametrize("split_length_and_results", [(1, 15), (10, 2)])
def test_preprocess_sentence_split(split_length_and_results):
    split_length, expected_documents_count = split_length_and_results

    document = Document(content=TEXT)
    preprocessor = PreProcessor(
        split_length=split_length, split_overlap=0, split_by="sentence", split_respect_sentence_boundary=False
    )
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count


@pytest.mark.unit
@pytest.mark.parametrize("split_length_and_results", [(1, 15), (10, 2)])
def test_preprocess_sentence_split_custom_models_wrong_file_format(split_length_and_results, samples_path):
    split_length, expected_documents_count = split_length_and_results

    document = Document(content=TEXT)
    preprocessor = PreProcessor(
        split_length=split_length,
        split_overlap=0,
        split_by="sentence",
        split_respect_sentence_boundary=False,
        tokenizer_model_folder=samples_path / "preprocessor" / "nltk_models" / "wrong",
        language="en",
    )
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count


@pytest.mark.unit
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


@pytest.mark.unit
@pytest.mark.parametrize("split_length_and_results", [(1, 8), (8, 1)])
@pytest.mark.skip(reason="Skipped after upgrade to nltk 3.9, can't load this model pt anymore")
def test_preprocess_sentence_split_custom_models(split_length_and_results, samples_path):
    split_length, expected_documents_count = split_length_and_results

    document = Document(content=LEGAL_TEXT_PT)
    preprocessor = PreProcessor(
        split_length=split_length,
        split_overlap=0,
        split_by="sentence",
        split_respect_sentence_boundary=False,
        language="pt",
        tokenizer_model_folder=samples_path / "preprocessor" / "nltk_models",
    )
    documents = preprocessor.process(document)
    assert len(documents) == expected_documents_count


@pytest.mark.unit
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


@pytest.mark.unit
def test_preprocess_page_split():
    doc = Document(
        content="This is a document on page 1.\fThis is a document on page 2.\fThis is a document on page 3."
    )
    output = PreProcessor(
        split_by="page", split_length=1, split_respect_sentence_boundary=False, split_overlap=0, add_page_number=True
    ).run([doc])[0]["documents"]
    assert len(output) == 3
    assert output[0] == Document(content="This is a document on page 1.", meta={"_split_id": 0, "page": 1})
    assert output[1] == Document(content="This is a document on page 2.", meta={"_split_id": 1, "page": 2})
    assert output[2] == Document(content="This is a document on page 3.", meta={"_split_id": 2, "page": 3})


@pytest.mark.unit
def test_preprocess_page_split_and_split_length():
    doc = Document(
        content="This is a document on page 1.\fThis is a document on page 2.\fThis is a document on page 3."
    )
    output = PreProcessor(
        split_by="page", split_length=2, split_respect_sentence_boundary=False, split_overlap=0, add_page_number=True
    ).run([doc])[0]["documents"]
    assert len(output) == 2
    assert output[0] == Document(
        content="This is a document on page 1.\fThis is a document on page 2.", meta={"_split_id": 0, "page": 1}
    )
    assert output[1] == Document(content="This is a document on page 3.", meta={"_split_id": 1, "page": 3})


@pytest.mark.unit
def test_preprocess_page_split_and_split_overlap():
    doc = Document(
        content="This is a document on page 1.\fThis is a document on page 2.\fThis is a document on page 3."
    )
    output = PreProcessor(
        split_by="page", split_length=2, split_respect_sentence_boundary=False, split_overlap=1, add_page_number=True
    ).run([doc])[0]["documents"]
    assert len(output) == 2
    assert output[0].content == "This is a document on page 1.\fThis is a document on page 2."
    assert output[0].meta["_split_id"] == 0
    assert output[0].meta["page"] == 1
    assert output[1].content == "This is a document on page 2.\fThis is a document on page 3."
    assert output[1].meta["_split_id"] == 1
    assert output[1].meta["page"] == 2


@pytest.mark.unit
def test_preprocess_page_split_with_empty_pages():
    doc = Document(
        content="This is a document on page 1.\f\fThis is a document on page 3.\f\fThis is a document on page 5."
    )
    output = PreProcessor(
        split_by="page", split_length=1, split_respect_sentence_boundary=False, split_overlap=0, add_page_number=True
    ).run([doc])[0]["documents"]
    assert len(output) == 3
    assert output[0] == Document(content="This is a document on page 1.", meta={"_split_id": 0, "page": 1})
    assert output[1] == Document(content="This is a document on page 3.", meta={"_split_id": 1, "page": 3})
    assert output[2] == Document(content="This is a document on page 5.", meta={"_split_id": 2, "page": 5})


@pytest.mark.unit
def test_preprocess_tiktoken_token_split(mock_tiktoken_tokenizer):
    raw_docs = [
        "This is a document. It has two sentences and eleven words.",
        "This is a document with a long sentence (longer than my split length), it has seventeen words.",
    ]
    docs = [Document(content=content) for content in raw_docs]
    split_length = 10
    token_split_docs_not_respecting_sentences = PreProcessor(
        split_by="token",
        split_length=split_length,
        split_respect_sentence_boundary=False,
        split_overlap=0,
        tokenizer="tiktoken",
    ).process(docs)
    assert len(token_split_docs_not_respecting_sentences) == 4
    enc = tiktoken.get_encoding("cl100k_base")
    split_documents_encoded = [
        enc.encode(d.content, allowed_special="all", disallowed_special=())
        for d in token_split_docs_not_respecting_sentences
    ]
    assert all(len(d) <= split_length for d in split_documents_encoded)
    token_split_docs_respecting_sentences = PreProcessor(
        split_by="token",
        split_length=split_length,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        tokenizer="tiktoken",
    ).process(docs)
    assert len(token_split_docs_respecting_sentences) == 3  # should not be more than there are sentences


@pytest.mark.unit
def test_preprocess_huggingface_token_split(mock_huggingface_tokenizer):
    raw_docs = [
        "This is a document. It has two sentences and eleven words.",
        "This is a document with a long sentence (longer than my split length), it has seventeen words.",
    ]
    docs = [Document(content=content) for content in raw_docs]
    split_length = 10
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    token_split_docs_not_respecting_sentences = PreProcessor(
        split_by="token",
        split_length=split_length,
        split_respect_sentence_boundary=False,
        split_overlap=0,
        tokenizer=tokenizer,
    ).process(docs)
    assert len(token_split_docs_not_respecting_sentences) == 8
    split_documents_retokenized = [tokenizer.tokenize(d.content) for d in token_split_docs_not_respecting_sentences]
    assert all(len(d) <= split_length for d in split_documents_retokenized)
    token_split_docs_respecting_sentences = PreProcessor(
        split_by="token",
        split_length=split_length,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        tokenizer=tokenizer,
    ).process(docs)
    assert len(token_split_docs_respecting_sentences) == 3  # should not be more than there are sentences
    token_split_docs_not_respecting_sentences_instantiate_by_name = PreProcessor(
        split_by="token",
        split_length=split_length,
        split_respect_sentence_boundary=False,
        split_overlap=0,
        tokenizer="bert-base-uncased",
    ).process(docs)
    assert token_split_docs_not_respecting_sentences == token_split_docs_not_respecting_sentences_instantiate_by_name


@pytest.mark.unit
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
def test_clean_header_footer(samples_path):
    converter = PDFToTextConverter()
    document = converter.convert(
        file_path=Path(samples_path / "pdf" / "sample_pdf_2.pdf")
    )  # file contains header/footer

    preprocessor = PreProcessor(clean_header_footer=True, split_by=None)
    documents = preprocessor.process(document)

    assert len(documents) == 1

    assert "This is a header." not in documents[0].content
    assert "footer" not in documents[0].content


@pytest.mark.unit
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


@pytest.mark.unit
def test_id_hash_keys_from_pipeline_params():
    document_1 = Document(content="This is a document.", meta={"key": "a"})
    document_2 = Document(content="This is a document.", meta={"key": "b"})
    assert document_1.id == document_2.id

    preprocessor = PreProcessor(split_length=2, split_respect_sentence_boundary=False)
    output, _ = preprocessor.run(documents=[document_1, document_2], id_hash_keys=["content", "meta"])
    documents = output["documents"]
    unique_ids = {d.id for d in documents}

    assert len(documents) == 4
    assert len(unique_ids) == 4


# test_input is a tuple consisting of the parameters for split_length, split_overlap and split_respect_sentence_boundary
# and the expected index in the output list of Documents where the page number changes from 1 to 2
@pytest.mark.unit
@pytest.mark.parametrize("test_input", [(10, 0, True, 5), (10, 0, False, 4), (10, 5, True, 5), (10, 5, False, 7)])
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_file_exists_error_during_download(monkeypatch: MonkeyPatch, module_tmp_dir: Path):
    # Pretend the model resources were not found in the first attempt
    monkeypatch.setattr(nltk.data, "find", Mock(side_effect=[LookupError, str(module_tmp_dir)]))

    # Pretend download throws a `FileExistsError` exception as a different process already downloaded it
    monkeypatch.setattr(nltk, "download", Mock(side_effect=FileExistsError))

    # This shouldn't raise an exception as the `FileExistsError` is ignored
    PreProcessor(split_length=2, split_respect_sentence_boundary=False)


@pytest.mark.unit
def test_preprocessor_very_long_document(caplog):
    preproc = PreProcessor(
        clean_empty_lines=False, clean_header_footer=False, clean_whitespace=False, split_by=None, max_chars_check=10
    )
    documents = [Document(content=str(i) + ("." * i)) for i in range(0, 30, 3)]
    results = preproc.process(documents)
    assert len(results) == 19
    assert any(d.content.startswith(".") for d in results)
    assert any(not d.content.startswith(".") for d in results)
    assert "characters long after preprocessing, where the maximum length should be 10." in caplog.text


@pytest.mark.unit
def test_split_respect_sentence_boundary_exceeding_split_len_not_repeated():
    preproc = PreProcessor(split_length=13, split_overlap=3, split_by="word", split_respect_sentence_boundary=True)
    document = Document(
        content=(
            "This is a test sentence with many many words that exceeds the split length and should not be repeated. "
            "This is another test sentence. (This is a third test sentence.) "
            "This is the last test sentence."
        )
    )
    documents = preproc.process(document)
    assert len(documents) == 3
    assert (
        documents[0].content
        == "This is a test sentence with many many words that exceeds the split length and should not be repeated. "
    )
    assert "This is a test sentence with many many words" not in documents[1].content
    assert "This is a test sentence with many many words" not in documents[2].content


@pytest.mark.unit
def test_split_overlap_information():
    preproc = PreProcessor(split_length=13, split_overlap=3, split_by="word", split_respect_sentence_boundary=True)
    document = Document(
        content=(
            "This is a test sentence with many many words that exceeds the split length and should not be repeated. "
            "This is another test sentence. (This is a third test sentence.) This is the fourth sentence. "
            "This is the last test sentence."
        )
    )
    documents = preproc.process(document)
    assert len(documents) == 4
    # The first Document should not overlap with any other Document as it exceeds the split length, the other Documents
    # should overlap with the previous Document (if applicable) and the next Document (if applicable)
    assert len(documents[0].meta["_split_overlap"]) == 0
    assert len(documents[1].meta["_split_overlap"]) == 1
    assert len(documents[2].meta["_split_overlap"]) == 2
    assert len(documents[3].meta["_split_overlap"]) == 1

    assert documents[1].meta["_split_overlap"][0]["doc_id"] == documents[2].id
    assert documents[2].meta["_split_overlap"][0]["doc_id"] == documents[1].id
    assert documents[2].meta["_split_overlap"][1]["doc_id"] == documents[3].id
    assert documents[3].meta["_split_overlap"][0]["doc_id"] == documents[2].id

    doc1_overlap_doc2 = documents[1].meta["_split_overlap"][0]["range"]
    doc2_overlap_doc1 = documents[2].meta["_split_overlap"][0]["range"]
    assert (
        documents[1].content[doc1_overlap_doc2[0] : doc1_overlap_doc2[1]]
        == documents[2].content[doc2_overlap_doc1[0] : doc2_overlap_doc1[1]]
    )

    doc2_overlap_doc3 = documents[2].meta["_split_overlap"][1]["range"]
    doc3_overlap_doc2 = documents[3].meta["_split_overlap"][0]["range"]
    assert (
        documents[2].content[doc2_overlap_doc3[0] : doc2_overlap_doc3[1]]
        == documents[3].content[doc3_overlap_doc2[0] : doc3_overlap_doc2[1]]
    )
