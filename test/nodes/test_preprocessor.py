import pytest
import pandas as pd

from haystack import Document
from haystack.nodes.preprocessor.preprocessor import PreProcessor

from ..conftest import SAMPLES_PATH

NLTK_TEST_MODELS = SAMPLES_PATH.absolute() / "preprocessor" / "nltk_models"


@pytest.fixture
def preprocessor():
    # Note: this are all simply fallback values.
    # Each test will call directly either run, split or clean providing the required input parameters.
    # If testing PreProcessor.__init__() they should not use this fixture
    return PreProcessor(
        split_by="page",
        split_length=1,
        clean_whitespace=True,
        clean_empty_lines=True,
        clean_header_footer=True,
        add_page_number=True,
    )


#
# Deprecations
#


def test_deprecated_run_with_one_doc(preprocessor: PreProcessor, fail_in_v1_14):
    with pytest.deprecated_call():
        preprocessor.run(documents=Document(content="abcde"))


def test_deprecated_run_with_one_dict_doc(preprocessor: PreProcessor, fail_in_v1_14):
    with pytest.deprecated_call():
        preprocessor.run(documents={"content": "abcde"})


def test_deprecated_run_with_list_of_dict_doc(preprocessor: PreProcessor, fail_in_v1_14):
    with pytest.deprecated_call():
        preprocessor.run(documents=[{"content": "abcde"}])


def test_deprecated_run_respect_sentence_boundary(preprocessor: PreProcessor, fail_in_v1_14):
    with pytest.deprecated_call():
        preprocessor.run(
            documents=[{"content": "abcde"}], split_by="page", split_length=500, split_respect_sentence_boundary=False
        )


def test_deprecated_run_clean_substrings(preprocessor: PreProcessor, fail_in_v1_14):
    with pytest.deprecated_call():
        preprocessor.run(
            documents=[{"content": "abcde"}], split_by="page", split_length=500, clean_substrings=["a", "b"]
        )


#
# Validations
#


def test_init_with_wrong_header_footer_n_chars():
    with pytest.raises(ValueError, match="header_footer_n_chars"):
        PreProcessor(
            split_by="page",
            split_length=1,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_n_chars=-1,
        )
    with pytest.raises(ValueError, match="header_footer_n_chars"):
        PreProcessor(
            split_by="page",
            split_length=1,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_n_chars=0.5,
        )


def test_init_with_wrong_header_footer_pages_to_ignore():
    with pytest.raises(ValueError, match="header_footer_pages_to_ignore"):
        PreProcessor(
            split_by="page",
            split_length=1,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_pages_to_ignore=2,
        )
    with pytest.raises(ValueError, match="header_footer_pages_to_ignore"):
        PreProcessor(
            split_by="page",
            split_length=1,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_pages_to_ignore=[1, 2, 3, 0.4, 5],
        )
    with pytest.raises(ValueError, match="header_footer_pages_to_ignore"):
        PreProcessor(
            split_by="page",
            split_length=1,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_pages_to_ignore=[1, 0.2, 3, -0.4, -5],
        )
    # Negative values are ok, they are counted from the end of the array.
    PreProcessor(
        split_by="page",
        split_length=1,
        clean_whitespace=True,
        clean_empty_lines=True,
        clean_header_footer=True,
        header_footer_pages_to_ignore=[1, 2, 3, -4, 5],
    )


def test_init_with_wrong_split_length():
    with pytest.raises(ValueError, match="split_length"):
        PreProcessor(
            split_by="page", split_length=0, clean_whitespace=True, clean_empty_lines=True, clean_header_footer=True
        )
    with pytest.raises(ValueError, match="split_length"):
        PreProcessor(
            split_by="page", split_length=-1, clean_whitespace=True, clean_empty_lines=True, clean_header_footer=True
        )
    with pytest.raises(ValueError, match="split_length"):
        PreProcessor(
            split_by="page", split_length=0.5, clean_whitespace=True, clean_empty_lines=True, clean_header_footer=True
        )


def test_init_with_wrong_split_overlap():
    with pytest.raises(ValueError, match="split_overlap"):
        PreProcessor(
            split_by="page",
            split_length=1,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            split_overlap=-1,
        )
    with pytest.raises(ValueError, match="split_overlap"):
        PreProcessor(
            split_by="page",
            split_length=1,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            split_overlap=0.5,
        )


def test_init_with_split_length_lower_or_equal_than_split_overlap():
    with pytest.raises(ValueError, match="split_length must be higher than split_overlap"):
        PreProcessor(
            split_by="page",
            split_length=1,
            split_overlap=2,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
        )
    with pytest.raises(ValueError, match="split_length must be higher than split_overlap"):
        PreProcessor(
            split_by="page",
            split_length=2,
            split_overlap=2,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
        )


def test_init_with_wrong_header_footer_n_chars(preprocessor: PreProcessor):
    with pytest.raises(ValueError, match="header_footer_n_chars"):
        preprocessor.run(
            documents=[Document(content="test")],
            split_by="page",
            split_length=1,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_n_chars=-1,
        )
    with pytest.raises(ValueError, match="header_footer_n_chars"):
        preprocessor.run(
            documents=[Document(content="test")],
            split_by="page",
            split_length=1,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_n_chars=0.5,
        )


def test_init_with_wrong_header_footer_pages_to_ignore(preprocessor: PreProcessor):
    with pytest.raises(ValueError, match="header_footer_pages_to_ignore"):
        preprocessor.run(
            documents=[Document(content="test")],
            split_by="page",
            split_length=1,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_pages_to_ignore=2,
        )
    with pytest.raises(ValueError, match="header_footer_pages_to_ignore"):
        preprocessor.run(
            documents=[Document(content="test")],
            split_by="page",
            split_length=1,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_pages_to_ignore=[1, 2, 3, 0.4, 5],
        )
    with pytest.raises(ValueError, match="header_footer_pages_to_ignore"):
        preprocessor.run(
            documents=[Document(content="test")],
            split_by="page",
            split_length=1,
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_pages_to_ignore=[1, 0.2, 3, -0.4, -5],
        )
    # Negative values are ok, they are counted from the end of the array.
    preprocessor.run(
        documents=[Document(content="test")],
        split_by="page",
        split_length=1,
        clean_whitespace=True,
        clean_empty_lines=True,
        clean_header_footer=True,
        header_footer_pages_to_ignore=[1, 2, 3, -4, 5],
    )


def test_run_with_wrong_object(preprocessor: PreProcessor):
    with pytest.raises(ValueError, match="list of Document"):
        preprocessor.run(documents="the document")
    with pytest.raises(ValueError, match="list of Document"):
        preprocessor.run(documents=["the", "documents"])


def test_run_with_wrong_content_type(preprocessor: PreProcessor):
    table_doc = Document(content=pd.DataFrame([1, 2]), content_type="table")
    with pytest.raises(ValueError, match="Preprocessor only handles text documents"):
        preprocessor.run(documents=[table_doc])

    image_doc = Document(content=str(SAMPLES_PATH / "images" / "apple.jpg"), content_type="image")
    with pytest.raises(ValueError, match="Preprocessor only handles text documents"):
        preprocessor.run(documents=[image_doc])
