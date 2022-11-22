from typing import Set, List

import sys
from pathlib import Path

import pytest
import pandas as pd

from haystack import __version__ as haystack_version
from haystack import Document
from haystack.nodes.preprocessor.preprocessor import PreProcessor
from haystack.nodes.preprocessor.splitter import load_tokenizer
from haystack.nodes.preprocessor.cleaner import longest_common_prefix, longest_common_suffix, replace_regex_matches

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
# Validations and deprecations
#

current_version = tuple(int(num) for num in haystack_version.split(".")[:2])


@pytest.fixture
def fail_in_v1_13():
    if current_version >= (1, 13):
        pytest.fail(reason="This feature should be removed in v1.13, as it was deprecated in v1.11")


def test_deprecated_run_with_one_doc(preprocessor, fail_in_v1_13):
    with pytest.deprecated_call():
        preprocessor.run(documents=Document(content="abcde"))


def test_deprecated_run_with_one_dict_doc(preprocessor, fail_in_v1_13):
    with pytest.deprecated_call():
        preprocessor.run(documents={"content": "abcde"})


def test_deprecated_run_with_list_of_dict_doc(preprocessor, fail_in_v1_13):
    with pytest.deprecated_call():
        preprocessor.run(documents=[{"content": "abcde"}])


def test_deprecated_run_respect_sentence_boundary(preprocessor, fail_in_v1_13):
    with pytest.deprecated_call():
        preprocessor.run(
            documents=[{"content": "abcde"}], split_by="page", split_length=500, split_respect_sentence_boundary=False
        )


def test_deprecated_run_clean_substrings(preprocessor, fail_in_v1_13):
    with pytest.deprecated_call():
        preprocessor.run(
            documents=[{"content": "abcde"}], split_by="page", split_length=500, clean_substrings=["a", "b"]
        )


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
            header_footer_pages_to_ignore=[1, 2, 3, -4, 5],
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


def test_run_with_wrong_object(preprocessor):
    with pytest.raises(ValueError, match="list of Document"):
        preprocessor.run(documents="the document")
    with pytest.raises(ValueError, match="list of Document"):
        preprocessor.run(documents=["the", "documents"])


def test_run_with_wrong_content_type(preprocessor):
    table_doc = Document(content=pd.DataFrame([1, 2]), content_type="table")
    with pytest.raises(ValueError, match="Preprocessor only handles text documents"):
        preprocessor.run(documents=[table_doc])

    image_doc = Document(content=str(SAMPLES_PATH / "images" / "apple.jpg"), content_type="image")
    with pytest.raises(ValueError, match="Preprocessor only handles text documents"):
        preprocessor.run(documents=[image_doc])


#
# Cleaning tests
#

longest_common_prefix_args = [
    ("full match", ["abcde", "abcde"], 3, 10, "abcde"),
    ("match too long", ["abcde", "abcde"], 3, 4, "abcd"),
    ("no match", ["abcde", "efghi"], 1, 3, None),
    ("match too short", ["abcde", "abefg"], 3, 5, None),
    ("newline in match", ["a\nbc\nde", "a\nbcfg ha"], 1, 5, "a\nbc"),
    ("too long with \n", ["a\nbc\nde", "a\nbcfg ha"], 1, 3, "a\nb"),
    ("full match one side", ["a\nbc\nde", "a\nbc"], 1, 5, "a\nbc"),
]


@pytest.mark.parametrize(
    "strings,min_len,max_len,prefix",
    [args[1:] for args in longest_common_prefix_args],
    ids=[i[0] for i in longest_common_prefix_args],
)
def test_longest_common_prefix(strings: List[str], min_len: int, max_len: int, prefix: str):
    assert prefix == longest_common_prefix(texts=strings, min_len=min_len, max_len=max_len)


longest_common_suffix_args = [
    ("full match", ["abcde", "abcde"], 3, 10, "abcde"),
    ("match too long", ["abcde", "abcde"], 3, 4, "bcde"),
    ("no match", ["abcde", "efghi"], 1, 3, None),
    ("match too short", ["abcde", "efgde"], 3, 5, None),
    ("newline in match", ["defa\nbc", "ghief a\nbc"], 1, 5, "a\nbc"),
    ("too long with \n", ["defa\nbc", "ghief a\nbc"], 1, 3, "\nbc"),
    ("full match one side", ["defa\nbc", "a\nbc"], 1, 5, "a\nbc"),
]


@pytest.mark.parametrize(
    "strings,min_len,max_len,suffix",
    [args[1:] for args in longest_common_suffix_args],
    ids=[i[0] for i in longest_common_suffix_args],
)
def test_longest_common_suffix(strings: List[str], min_len: int, max_len: int, suffix: Set[str]):
    assert suffix == longest_common_suffix(texts=strings, min_len=min_len, max_len=max_len)


remove_whitespace_args = [
    ("Nothing to clean, no headlines", "a\fb\nc", "a\fb\nc", None, None),
    ("Trailing newlines and form feeds are kept, no headlines", "a\n\fb\nc\f", "a\n\fb\nc\f", None, None),
    (
        "Nothing to clean, with headlines",
        "a\f#Title\nc\f",
        "a\f#Title\nc\f",
        [{"content": "#Title", "start_idx": 2}],
        [{"content": "#Title", "start_idx": 2}],
    ),
    ("Single page, no headlines", " a \nb c\nd    \n   e ", "a\nb c\nd\ne", None, None),
    ("Multiple pages, no headlines", " a \f  b\nc     \f", "a\fb\nc\f", None, None),
    (
        "Single page with headlines",
        "   #Title \n#Title2   ",
        "#Title\n#Title2",
        [{"content": "#Title", "start_idx": 3}, {"content": "#Title2", "start_idx": 11}],
        [{"content": "#Title", "start_idx": 0}, {"content": "#Title2", "start_idx": 7}],
    ),
    (
        "Single page with headlines containing spaces",
        "#Title    \n    #Title2   ",
        "#Title\n#Title2",
        [{"content": "#Title    ", "start_idx": 0}, {"content": "    #Title2   ", "start_idx": 15}],
        [{"content": "#Title", "start_idx": 0}, {"content": "#Title2", "start_idx": 7}],
    ),
    (
        "Single page with headlines containing spaces at the start",
        "a\n   #Title \n#Title2   ",
        "a\n#Title\n#Title2",
        [{"content": "   #Title ", "start_idx": 2}, {"content": "#Title2  ", "start_idx": 13}],
        [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 9}],
    ),
    (
        "Multi page with headlines",
        "   #Title \f#Title2   ",
        "#Title\f#Title2",
        [{"content": "#Title", "start_idx": 3}, {"content": "#Title2", "start_idx": 11}],
        [{"content": "#Title", "start_idx": 0}, {"content": "#Title2", "start_idx": 7}],
    ),
    (
        "Empty page with headlines",
        "   #Title \f\f\f#Title2   ",
        "#Title\f\f\f#Title2",
        [{"content": "#Title", "start_idx": 3}, {"content": "#Title2", "start_idx": 13}],
        [{"content": "#Title", "start_idx": 0}, {"content": "#Title2", "start_idx": 9}],
    ),
    (
        "With multiple pages, headlines and text",
        " a  \n#Title \n\f d  \n #Title2 \n f",
        "a\n#Title\n\fd\n#Title2\nf",
        [{"content": "#Title", "start_idx": 5}, {"content": "#Title2", "start_idx": 18}],
        [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 11}],
    ),
    (
        "Unsorted headlines will be sorted",
        " a \n#Title \f d  \n #Title2 \n f",
        "a\n#Title\fd\n#Title2\nf",
        [{"content": "#Title2", "start_idx": 18}, {"content": "#Title", "start_idx": 4}],
        [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 11}],
    ),
    (
        "With headlines and multiple empty lines",
        "\n\n a \n#Title \n\n\n d  \n\n\n\n",
        "\n\na\n#Title\n\n\nd\n\n\n",
        [{"content": "#Title", "start_idx": 6}],
        [{"content": "#Title", "start_idx": 4}],
    ),
    (
        "With headlines and multiple empty lines/pages",
        "\n\n a \n#Title \f\f\f d  \n #Title2 \f\n\n\n\n",
        "\n\na\n#Title\f\f\fd\n#Title2\f\n\n\n",
        [{"content": "#Title2", "start_idx": 21}, {"content": "#Title", "start_idx": 6}],
        [{"content": "#Title", "start_idx": 4}, {"content": "#Title2", "start_idx": 15}],
    ),
]


@pytest.mark.parametrize(
    "text,clean_text,headlines,clean_headlines",
    [args[1:] for args in remove_whitespace_args],
    ids=[i[0] for i in remove_whitespace_args],
)
def test_remove_whitespace(preprocessor: PreProcessor, text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = preprocessor.cleaner.run(
        documents=[doc_to_clean], clean_whitespace=True, clean_header_footer=False, clean_empty_lines=False
    )[0]["documents"][0]

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


remove_empty_lines_args = [
    ("Nothing to clean, no headlines", "a\fb\nc\f", "a\fb\nc\f", None, None),
    (
        "Nothing to clean, with headlines",
        "a\f#Title\nc\n\f",
        "a\f#Title\nc\n\f",
        [{"content": "#Title", "start_idx": 2}],
        [{"content": "#Title", "start_idx": 2}],
    ),
    ("Single page, no headlines", "\n\na\n\n\nb\n", "\na\nb\n", None, None),
    ("Multiple pages, no headlines", "\n\na\n\n\fb\n\n\n\nc\n\n\f\f\f", "\na\n\fb\nc\n\f\f\f", None, None),
    (
        "Single page with headlines",
        "\n\n#Title\n\n\n\n#Title2",
        "\n#Title\n#Title2",
        [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 12}],
        [{"content": "#Title", "start_idx": 1}, {"content": "#Title2", "start_idx": 8}],
    ),
    (
        "Multi page with headlines",
        "\n\n#Title\n\n\n\n\f#Title2\n\f\n",
        "\n#Title\n\f#Title2\n\f\n",
        [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 13}],
        [{"content": "#Title", "start_idx": 1}, {"content": "#Title2", "start_idx": 9}],
    ),
    (
        "With multiple pages, headlines and text",
        "a\n\n#Title\n\n\n\nb c\n\f#Title2\n\f\n",
        "a\n#Title\nb c\n\f#Title2\n\f\n",
        [{"content": "#Title", "start_idx": 3}, {"content": "#Title2", "start_idx": 17}],
        [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 13}],
    ),
    (
        "Unsorted headlines will be sorted",
        "a\n\n#Title\n\n\n\nb c\n\f#Title2\n\f\n",
        "a\n#Title\nb c\n\f#Title2\n\f\n",
        [{"content": "#Title2", "start_idx": 17}, {"content": "#Title", "start_idx": 3}],
        [{"content": "#Title", "start_idx": 2}, {"content": "#Title2", "start_idx": 13}],
    ),
]


@pytest.mark.parametrize(
    "text,clean_text,headlines,clean_headlines",
    [args[1:] for args in remove_empty_lines_args],
    ids=[i[0] for i in remove_empty_lines_args],
)
def test_remove_empty_lines(preprocessor: PreProcessor, text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = preprocessor.cleaner.run(
        document=doc_to_clean, clean_whitespace=False, clean_header_footer=False, clean_empty_lines=True
    )[0]["documents"][0]

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


remove_substrings_args = [
    ("No strings to remove, no headers", [], "abcdefg", "abcdefg", None, None),
    ("Empty string to remove, no headers", [""], "abcdefg", "abcdefg", None, None),
    ("One single char, no headers", ["A"], "aAbc", "abc", None, None),
    ("No string of a few chars, no headers", ["AA"], "AAAAabcdAAAefgAhAA", "abcdAefgAh", None, None),
    ("Remove unicode/emoji, no headers", ["🪲"], "abcd🪲efgh🪲", "abcdefgh", None, None),
    ("Remove only the given unicode/emoji, no headers", ["🪲"], "✨abc✨d🪲ef✨gh🪲", "✨abc✨def✨gh", None, None),
    ("Do not affect whitespace, no headers", ["🪲"], "\b\b\babc\n\nde🪲\f\nfgh", "\b\b\babc\n\nde\f\nfgh", None, None),
    ("Multiple single char strings, no headers", ["🪲", "A"], "aA🪲bA🪲", "ab", None, None),
    ("Multiple multi char strings, no headers", ["A🪲", "AA"], "AAaA🪲bA🪲", "ab", None, None),
    ("Remove single char whitespace, no headers", ["\n"], "a\nb\n", "ab", None, None),
    ("Remove multi char whitespace, no headers", ["\n "], "a\nb\n ", "a\nb", None, None),
    (
        "Simple header/footer removal, no headers",
        ["a.\nb c.\n", "c.\nd.\n"],
        "a.\nb c.\ncde fg hi c.\nd.\n",
        "cde fg hi ",
        None,
        None,
    ),
    (
        "multichar string with headers, sigle removal before header",
        ["aa"],
        "a # Test header aa caa def",
        "a # Test header  c def",
        [{"content": "# Test header", "start_idx": 2}],
        [{"content": "# Test header", "start_idx": 2}],
    ),
    (
        "multichar string with headers, many removals in a row before header",
        ["aa "],
        "aa aa aa aa aa  # Test header a caa  def aa aa ",
        " # Test header a c def ",
        [{"content": "# Test header", "start_idx": 16}],
        [{"content": "# Test header", "start_idx": 1}],
    ),
    (
        "multichar string with headers, many headers after removal",
        ["aa "],
        "aa # Test header a # Test header # Test header cdef aa # Test header ",
        "# Test header a # Test header # Test header cdef # Test header ",
        [
            {"content": "# Test header", "start_idx": 3},
            {"content": "# Test header", "start_idx": 19},
            {"content": "# Test header", "start_idx": 33},
            {"content": "# Test header", "start_idx": 55},
        ],
        [
            {"content": "# Test header", "start_idx": 0},
            {"content": "# Test header", "start_idx": 16},
            {"content": "# Test header", "start_idx": 30},
            {"content": "# Test header", "start_idx": 49},
        ],
    ),
    (
        "The removal affects the header lenght, no other headers follow",
        ["Test"],
        "a # Test header b c def",
        "a #  header b c def",
        [{"content": "# Test header", "start_idx": 2}],
        [{"content": "# Test header", "start_idx": 2}],
    ),
    (
        "The removal affects the header lenght, other headers follow",
        ["Test"],
        "a # Test header 1 b c # Test header 2 def",
        "a #  header 1 b c #  header 2 def",
        [{"content": "# Test header 2", "start_idx": 22}, {"content": "# Test header 1", "start_idx": 2}],
        [{"content": "# Test header 1", "start_idx": 2}, {"content": "# Test header 2", "start_idx": 18}],
    ),
    (
        "More complete header/footer removal example with removal affecting multiple headers",
        ["the header", "the footer", "Test"],
        "the header\na\n# Test header 1\nb c\n# Test header 2\ndef\nthe footer",
        "\na\n#  header 1\nb c\n#  header 2\ndef\n",
        [{"content": "# Test header 2", "start_idx": 33}, {"content": "# Test header 1", "start_idx": 13}],
        [{"content": "# Test header 1", "start_idx": 3}, {"content": "# Test header 2", "start_idx": 19}],
    ),
]


@pytest.mark.parametrize(
    "substrings,text,clean_text,headlines,clean_headlines",
    [args[1:] for args in remove_substrings_args],
    ids=[i[0] for i in remove_substrings_args],
)
def test_remove_substrings(substrings, text, clean_text, headlines, clean_headlines, fail_in_v1_13):
    # Replaced by the test below, test_remove_regex_matches
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = replace_regex_matches(doc_to_clean, pattern=f"({'|'.join(substrings)})", string="")

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


remove_regex_matches_args = [
    ("Empty group causes no-op, no headers", "()", "abcdefg", "abcdefg", None, None),
    ("Single char string, no headers", "A", "aAbc", "abc", None, None),
    ("Simple group, no headers", "[0-9]*", "a135b56c00", "abc", None, None),
    ("Multichar string group, no headers", "(AA)", "AAAAabcdAAAefgAhAA", "abcdAefgAh", None, None),
    ("Unicode/emoji match, no headers", "🪲", "abcd🪲efgh🪲", "abcdefgh", None, None),
    (
        "Unicode/emoji match does not affect other unicode/emoji, no headers",
        "🪲",
        "✨abc✨d🪲ef✨gh🪲",
        "✨abc✨def✨gh",
        None,
        None,
    ),
    ("Do not affect whitespace, no headers", "🪲", "\b\b\babc\n\nde🪲\f\nfgh", "\b\b\babc\n\nde\f\nfgh", None, None),
    ("Multistring match, no headers", "(🪲|A)", "aA🪲bA🪲", "ab", None, None),
    ("Single char whitespace match, no headers", "(\n)", "a\nb\n", "ab", None, None),
    ("Multi char whitespace match, no headers", "(\n )", "a\n \n\nb\n \n ", "a\n\nb", None, None),
    (
        "Simple header/footer removal, no headers",
        "(a.\nb c.\n|c.\nd.\n)",
        "a.\nb c.\ncde fg hi. c.\nd.\n",
        "cde fg hi. ",
        None,
        None,
    ),
    (
        "Parametric header/footer removal, no headers",
        "(-- Page [0-9]* of [0-9]* --|~~ Chapter [0-9]*: .* ~~)",
        "~~ Chapter 1: a test ~~ some text -- Page 1 of 2000 --\n~~ Chapter 2: another test ~~ some more text -- Page 2 of 2000 --",
        " some text \n some more text ",
        None,
        None,
    ),
    (
        "multichar string with headers, single removal before header",
        "(aa)",
        "a # Test header aa caa def",
        "a # Test header  c def",
        [{"content": "# Test header", "start_idx": 2}],
        [{"content": "# Test header", "start_idx": 2}],
    ),
    (
        "multichar string with headers, many removals in a row before header",
        "(aa )",
        "aa aa aa aa aa  # Test header a caa  def aa aa ",
        " # Test header a c def ",
        [{"content": "# Test header", "start_idx": 16}],
        [{"content": "# Test header", "start_idx": 1}],
    ),
    (
        "multichar string with headers, many headers after removal",
        "(aa )",
        "aa # Test header a # Test header # Test header cdef aa # Test header ",
        "# Test header a # Test header # Test header cdef # Test header ",
        [
            {"content": "# Test header", "start_idx": 3},
            {"content": "# Test header", "start_idx": 19},
            {"content": "# Test header", "start_idx": 33},
            {"content": "# Test header", "start_idx": 55},
        ],
        [
            {"content": "# Test header", "start_idx": 0},
            {"content": "# Test header", "start_idx": 16},
            {"content": "# Test header", "start_idx": 30},
            {"content": "# Test header", "start_idx": 49},
        ],
    ),
    (
        "The removal affects the header lenght, no other headers follow",
        "(Test)",
        "a # Test header b c def",
        "a #  header b c def",
        [{"content": "# Test header", "start_idx": 2}],
        [{"content": "#  header", "start_idx": 2}],
    ),
    (
        "The removal affects the header lenght, other headers follow",
        "(Test)",
        "a # Test header 1 b c # Test header 2 def",
        "a #  header 1 b c #  header 2 def",
        [{"content": "# Test header 2", "start_idx": 22}, {"content": "# Test header 1", "start_idx": 2}],
        [{"content": "#  header 1", "start_idx": 2}, {"content": "#  header 2", "start_idx": 18}],
    ),
    (
        "The removal affects a header multiple times",
        "(Test)",
        "a # Test Test Test header 1 b c # Test header Test 2 def",
        "a #    header 1 b c #  header  2 def",
        [
            {"content": "# Test header Test 2", "start_idx": 32},
            {"content": "# Test Test Test header 1", "start_idx": 2},
        ],
        [{"content": "#    header 1", "start_idx": 2}, {"content": "#  header  2", "start_idx": 20}],
    ),
    (
        "Header/footer removal example with removal affecting multiple headers",
        "(the header|the footer|Test)",
        "the header\na\n# Test header 1\nb c\n# Test header 2\ndef\nthe footer",
        "\na\n#  header 1\nb c\n#  header 2\ndef\n",
        [{"content": "# Test header 2", "start_idx": 33}, {"content": "# Test header 1", "start_idx": 13}],
        [{"content": "#  header 1", "start_idx": 3}, {"content": "#  header 2", "start_idx": 19}],
    ),
    (
        "Variable header/footer removal example with removal affecting multiple headers",
        "(the [0-9]*th header|the [0-9]*th footer|Test [0-9]* )",
        "the 4th header\na\n# Test 234 header 1\nb c\n# Test 33 header 2\ndef\nthe 80th footer",
        "\na\n# header 1\nb c\n# header 2\ndef\n",
        [{"content": "# Test 33 header 2", "start_idx": 41}, {"content": "# Test 234 header 1", "start_idx": 17}],
        [{"content": "# header 1", "start_idx": 3}, {"content": "# header 2", "start_idx": 18}],
    ),
]


@pytest.mark.parametrize(
    "regex,text,clean_text,headlines,clean_headlines",
    [args[1:] for args in remove_regex_matches_args],
    ids=[i[0] for i in remove_regex_matches_args],
)
def test_remove_regex_matches(regex, text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = replace_regex_matches(doc_to_clean, pattern=regex, string="")

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


replace_regex_matches_args = [
    ("Empty group causes no-op, no headers", "()", "abcdefg", "abcdefg", None, None),
    ("Single char string, no headers", "A", "aAbc", "a@@bc", None, None),
    ("Simple group, no headers", "[0-9]+", "a135b56c00", "a@@b@@c@@", None, None),
    ("Multichar string group, no headers", "(AA)", "AAAAabcdAAAefgAhAA", "@@@@abcd@@AefgAh@@", None, None),
    ("Unicode/emoji match, no headers", "🪲", "abcd🪲efgh🪲", "abcd@@efgh@@", None, None),
    (
        "Unicode/emoji match does not affect other unicode/emoji, no headers",
        "🪲",
        "✨abc✨d🪲ef✨gh🪲",
        "✨abc✨d@@ef✨gh@@",
        None,
        None,
    ),
    ("Do not affect whitespace, no headers", "🪲", "\b\b\babc\n\nde🪲\f\nfgh", "\b\b\babc\n\nde@@\f\nfgh", None, None),
    ("Multistring match, no headers", "(🪲|A)", "aA🪲bA🪲", "a@@@@b@@@@", None, None),
    ("Single char whitespace match, no headers", "(\n)", "a\nb\n", "a@@b@@", None, None),
    ("Multi char whitespace match, no headers", "(\n )", "a\n \n\nb\n \n ", "a@@\n\nb@@@@", None, None),
    (
        "Simple header/footer removal, no headers",
        "(a.\nb c.\n|c.\nd.\n)",
        "a.\nb c.\ncde fg hi. c.\nd.\n",
        "@@cde fg hi. @@",
        None,
        None,
    ),
    (
        "Parametric header/footer removal, no headers",
        "(-- Page [0-9]* of [0-9]* --|~~ Chapter [0-9]*: .* ~~)",
        "~~ Chapter 1: a test ~~ some text -- Page 1 of 2000 --\n~~ Chapter 2: another test ~~ some more text -- Page 2 of 2000 --",
        "@@ some text @@\n@@ some more text @@",
        None,
        None,
    ),
    (
        "multichar string with headers, single removal before header",
        "(aa)",
        "a # Test header aa caa def",
        "a # Test header @@ c@@ def",
        [{"content": "# Test header", "start_idx": 2}],
        [{"content": "# Test header", "start_idx": 2}],
    ),
    (
        "multichar string with headers, many removals in a row before header",
        "(aa )",
        "aa aa aa aa aa  # Test header a caa  def aa aa ",
        "@@@@@@@@@@ # Test header a c@@ def @@@@",
        [{"content": "# Test header", "start_idx": 16}],
        [{"content": "# Test header", "start_idx": 11}],
    ),
    (
        "multichar string with headers, many headers after removal",
        "(aa )",
        "aa # Test header a # Test header # Test header cdef aa # Test header ",
        "@@# Test header a # Test header # Test header cdef @@# Test header ",
        [
            {"content": "# Test header", "start_idx": 3},
            {"content": "# Test header", "start_idx": 19},
            {"content": "# Test header", "start_idx": 33},
            {"content": "# Test header", "start_idx": 55},
        ],
        [
            {"content": "# Test header", "start_idx": 2},
            {"content": "# Test header", "start_idx": 18},
            {"content": "# Test header", "start_idx": 32},
            {"content": "# Test header", "start_idx": 53},
        ],
    ),
    (
        "The removal affects the header lenght, no other headers follow",
        "(Test)",
        "a # Test header b c def",
        "a # @@ header b c def",
        [{"content": "# Test header", "start_idx": 2}],
        [{"content": "# @@ header", "start_idx": 2}],
    ),
    (
        "The removal affects the header lenght, other headers follow",
        "(Test)",
        "a # Test header 1 b c # Test header 2 def",
        "a # @@ header 1 b c # @@ header 2 def",
        [{"content": "# Test header 2", "start_idx": 22}, {"content": "# Test header 1", "start_idx": 2}],
        [{"content": "# @@ header 1", "start_idx": 2}, {"content": "# @@ header 2", "start_idx": 20}],
    ),
    (
        "The removal affects a header multiple times",
        "(Test)",
        "a # Test Test Test header 1 b c # Test header Test 2 def",
        "a # @@ @@ @@ header 1 b c # @@ header @@ 2 def",
        [
            {"content": "# Test header Test 2", "start_idx": 32},
            {"content": "# Test Test Test header 1", "start_idx": 2},
        ],
        [{"content": "# @@ @@ @@ header 1", "start_idx": 2}, {"content": "# @@ header @@ 2", "start_idx": 26}],
    ),
    (
        "Header/footer removal example with removal affecting multiple headers",
        "(the header|the footer|Test)",
        "the header\na\n# Test header 1\nb c\n# Test header 2\ndef\nthe footer",
        "@@\na\n# @@ header 1\nb c\n# @@ header 2\ndef\n@@",
        [{"content": "# Test header 2", "start_idx": 33}, {"content": "# Test header 1", "start_idx": 13}],
        [{"content": "# @@ header 1", "start_idx": 5}, {"content": "# @@ header 2", "start_idx": 23}],
    ),
    (
        "Variable header/footer removal example with removal affecting multiple headers",
        "(the [0-9]*th header|the [0-9]*th footer|Test [0-9]* )",
        "the 4th header\na\n# Test 234 header 1\nb c\n# Test 33 header 2\ndef\nthe 80th footer",
        "@@\na\n# @@header 1\nb c\n# @@header 2\ndef\n@@",
        [{"content": "# Test 33 header 2", "start_idx": 41}, {"content": "# Test 234 header 1", "start_idx": 17}],
        [{"content": "# @@header 1", "start_idx": 5}, {"content": "# @@header 2", "start_idx": 22}],
    ),
]


@pytest.mark.parametrize(
    "regex,text,clean_text,headlines,clean_headlines",
    [args[1:] for args in replace_regex_matches_args],
    ids=[i[0] for i in replace_regex_matches_args],
)
def test_replace_regex_matches(regex, text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = replace_regex_matches(doc_to_clean, pattern=regex, string="@@")

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


remove_header_footer_args = [
    (
        "Base case, no headers",
        """
--- I'm a header. ---
A long piece of text that comes from the first page of the document,
~~~~~ I'm a footer. ~~~~~\f
--- I'm a header. ---
and this is another long piece of text from the second page of the document.
~~~~~ I'm a footer. ~~~~~\f""",
        """A long piece of text that comes from the first page of the document,\fand this is another long piece of text from the second page of the document.\f""",
        None,
        None,
    ),
    (
        "Header too long to be fully matched, no headers",
        """
--- I'm such a long header that I will not be matched entirely due to the n_char parameter. ---
A long piece of text that comes from the first page of the document,
~~~~~ I'm a footer. ~~~~~\f
--- I'm such a long header that I will not be matched entirely due to the n_char parameter. ---
and this is another long piece of text from the second page of the document.
~~~~~ I'm a footer. ~~~~~\f""",
        """ched entirely due to the n_char parameter. ---
A long piece of text that comes from the first page of the document,\fched entirely due to the n_char parameter. ---
and this is another long piece of text from the second page of the document.\f""",
        None,
        None,
    ),
    (
        "Base case with headers",
        """
--- I'm a header. ---
First headline
A long piece of text that comes from the first page of the document,
~~~~~ I'm a footer. ~~~~~\f
--- I'm a header. ---
Second headline
and this is another long piece of text from the second page of the document.
~~~~~ I'm a footer. ~~~~~\f""",
        """First headline
A long piece of text that comes from the first page of the document,\fSecond headline
and this is another long piece of text from the second page of the document.\f""",
        [{"content": "First headline", "start_idx": 23}, {"content": "Second headline", "start_idx": 156}],
        [{"content": "First headline", "start_idx": 0}, {"content": "Second headline", "start_idx": 84}],
    ),
    (
        "Header too long to be fully matched, with headers",
        """
--- I'm such a long header that I will not be matched entirely due to the n_char parameter. ---
First headline
A long piece of text that comes from the first page of the document,
~~~~~ I'm a footer. ~~~~~\f
--- I'm such a long header that I will not be matched entirely due to the n_char parameter. ---
Second headline
and this is another long piece of text from the second page of the document.
~~~~~ I'm a footer. ~~~~~\f""",
        """ched entirely due to the n_char parameter. ---
First headline
A long piece of text that comes from the first page of the document,\fched entirely due to the n_char parameter. ---
Second headline
and this is another long piece of text from the second page of the document.\f""",
        [{"content": "First headline", "start_idx": 97}, {"content": "Second headline", "start_idx": 304}],
        [{"content": "First headline", "start_idx": 47}, {"content": "Second headline", "start_idx": 178}],
    ),
    (
        "Headlines that are also headers are removed",
        """
--- Headline ---
A long piece of text that comes from the first page of the document,
~~~~~ I'm a footer. ~~~~~\f
--- Headline ---
### Sub headline
and this is another long piece of text from the second page of the document.
~~~~~ I'm a footer. ~~~~~\f""",
        """A long piece of text that comes from the first page of the document,\f### Sub headline
and this is another long piece of text from the second page of the document.\f""",
        [
            {"content": "--- Headline ---", "start_idx": 0},
            {"content": "--- Headline ---", "start_idx": 114},
            {"content": "### Sub headline", "start_idx": 131},
        ],
        [{"content": "### Sub headline", "start_idx": 69}],
    ),
    (
        "Headlines that are also footers are removed",
        """
--- I'm a header ---
A long piece of text that comes from the first page of the document,
~~~~~ Footer Headline ~~~~~\f
--- I'm a header ---
### Sub headline
and this is another long piece of text from the second page of the document.
~~~~~ Footer Headline ~~~~~\f""",
        """A long piece of text that comes from the first page of the document,\f### Sub headline
and this is another long piece of text from the second page of the document.\f""",
        [
            {"content": "--- Footer Headline ---", "start_idx": 91},
            {"content": "--- Footer Headline ---", "start_idx": 235},
            {"content": "### Sub headline", "start_idx": 141},
        ],
        [{"content": "### Sub headline", "start_idx": 69}],
    ),
]


@pytest.mark.parametrize(
    "text,clean_text,headlines,clean_headlines",
    [args[1:] for args in remove_header_footer_args],
    ids=[i[0] for i in remove_header_footer_args],
)
def test_remove_header_footer(preprocessor: PreProcessor, text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = preprocessor.run(
        documents=[doc_to_clean],
        clean_empty_lines=False,
        clean_header_footer=True,
        clean_whitespace=False,
        split_by="regex",
        split_regex="(I'm nowhere to be found!)",
    )

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


#
# Splitting tests
#


def test_split_by_random(preprocessor):
    with pytest.raises(ValueError, match="split_length"):
        preprocessor.splitter.run(
            documents=[Document(content="test")], split_by="random", split_length=1, split_overlap=10
        )


def test_split_overlap_above_split_length(preprocessor):
    with pytest.raises(ValueError, match="split_length"):
        preprocessor.splitter.run(
            documents=[Document(content="test")], split_by="page", split_length=1, split_overlap=10
        )
    with pytest.raises(ValueError, match="split_length"):
        preprocessor.splitter.run(
            documents=[Document(content="test")], split_by="page", split_length=10, split_overlap=10
        )


#
# Regex-based splits
#


def test_split_by_regex_no_regex_given(preprocessor):
    with pytest.raises(ValueError, match="split_regex"):
        preprocessor.splitter.run(
            documents=[Document(content="test doc")],
            split_by="regex",
            split_length=2,
            split_overlap=1,
            split_max_chars=20,
            add_page_number=True,
        )


def test_split_by_something_with_regex_given(preprocessor, caplog):
    preprocessor.splitter.run(
        documents=[Document(content="test doc")],
        split_by="page",
        split_regex=r"[a-z]*",
        split_length=2,
        split_overlap=1,
        split_max_chars=20,
        add_page_number=True,
    )
    assert "regex pattern will be ignored" in caplog.text


split_by_regex_no_headlines_args = [
    ("Empty string", "", [""], [1], 1, 0),
    ("Whitespace only string", "  '\f\t\n\n  \n", ["  '\f\t\n\n  \n"], [1], 1, 0),
    ("No matches", "This doesn't need to be split", ["This doesn't need to be split"], [1], 1, 0),
    ("No matches with overlap", "This doesn't need to be split", ["This doesn't need to be split"], [1], 10, 5),
    (
        "Simplest case",
        "Paragraph 1~0000000~Paragraph 2___12234556___a\fParagraph 3~header~Paragraph 4___footer___",
        ["Paragraph 1~0000000~", "Paragraph 2___12234556___", "a\fParagraph 3~header~", "Paragraph 4___footer___"],
        [1, 1, 1, 2],
        1,
        0,
    ),
    (
        "Page breaks and other whitespace don't matter around the match",
        "Paragraph 1\n___111___\nParagraph\f2~2222~Paragraph 3~3333333~Paragraph 4",
        ["Paragraph 1\n___111___", "\nParagraph\f2~2222~", "Paragraph 3~3333333~", "Paragraph 4"],
        [1, 1, 2, 2],
        1,
        0,
    ),
    (
        "Page breaks and other whitespace don't matter far from the match",
        "Paragraph 1\nStill Paragraph 1~header~\fParagraph 2___footer___Paragraph 3",
        ["Paragraph 1\nStill Paragraph 1~header~", "\fParagraph 2___footer___", "Paragraph 3"],
        [1, 1, 2],
        1,
        0,
    ),
    (
        "Multiple form feed don't do anything",
        "Paragraph 1___footer___Paragraph 2~header~\f\f\fParagraph 3~header~Paragraph 4",
        ["Paragraph 1___footer___", "Paragraph 2~header~", "\f\f\fParagraph 3~header~", "Paragraph 4"],
        [1, 1, 1, 4],
        1,
        0,
    ),
    (
        # By empty document we mean both document made only by whitespace and document made only by the regex match.
        "Empty documents",
        "Paragraph 1~header~Paragraph 2___footer___\fParagraph 3___footer___~header~Paragraph 5~header~ \f ~09876~    \n\f ",
        [
            "Paragraph 1~header~",
            "Paragraph 2___footer___",
            "\fParagraph 3___footer___",
            "~header~",  # Empty document: contains only the regex match
            "Paragraph 5~header~",
            " \f ~09876~",  # Empty document: contains only whitespace and the regex match
            "    \n\f ",  # Empty document: contains only whitespace
        ],
        [1, 1, 1, 2, 2, 2, 3],
        1,
        0,
    ),
    (
        "Group by 2",
        "Paragraph 1___footer___Paragraph 2\f~header~Paragraph 3~3333~Paragraph 4",
        ["Paragraph 1___footer___Paragraph 2\f~header~", "Paragraph 3~3333~Paragraph 4"],
        [1, 2],
        2,
        0,
    ),
    (
        "Group by 3",
        "Paragraph 1~header~Paragraph 2___footer___\fParagraph 3___footer___Paragraph 4",
        ["Paragraph 1~header~Paragraph 2___footer___\fParagraph 3___footer___", "Paragraph 4"],
        [1, 2],
        3,
        0,
    ),
    (
        "Group with empty documents",
        "___11111___~11111~___22222___\fParagraph 3~33333~Paragraph 4",
        [
            "___11111___~11111~",  # This can be considered as a fully empty document
            "___22222___\fParagraph 3~33333~",
            "Paragraph 4",
        ],
        [1, 1, 2],
        2,
        0,
    ),
    (
        "Overlap of 1",
        "Paragraph 1___footer___Paragraph 2\f~header~Paragraph 3~header~Paragraph 4___footer___",
        [
            "Paragraph 1___footer___Paragraph 2\f~header~",
            "Paragraph 2\f~header~Paragraph 3~header~",
            "Paragraph 3~header~Paragraph 4___footer___",
        ],
        [1, 1, 2],
        2,
        1,
    ),
    (
        "Overlap of 1 with empty documents",
        "Paragraph 1\f___footer___~header~~header~Paragraph 2\f~header~~header~~header~",
        [
            "Paragraph 1\f___footer___~header~",
            "~header~~header~",  # This can be considered as a fully empty document
            "~header~Paragraph 2\f~header~",
            "Paragraph 2\f~header~~header~",
            "~header~~header~",
        ],
        [1, 2, 2, 2, 3],
        2,
        1,
    ),
    (
        "Overlap of 2 with empty documents",
        "Paragraph 1\f___footer___~header~___footer___~header~Paragraph 2___footer___\f~header~~header~",
        [
            "Paragraph 1\f___footer___~header~___footer___",
            "~header~___footer___~header~",  # This can be considered as a fully empty document
            "___footer___~header~Paragraph 2___footer___",
            "~header~Paragraph 2___footer___\f~header~",
            "Paragraph 2___footer___\f~header~~header~",
        ],
        [1, 2, 2, 2, 2],
        3,
        2,
    ),
    (
        "Overlap of 1 with many empty documents",
        "~header~Paragraph 1\f___footer___~1111~___footer______000000___~header~~22222~",
        [
            "~header~Paragraph 1\f___footer___~1111~",
            "~1111~___footer______000000___",  # This can be considered as a fully empty document
            "___000000___~header~~22222~",  # This can be considered as a fully empty document
        ],
        [1, 2, 2],
        3,
        1,
    ),
]


@pytest.mark.parametrize(
    "document,expected_documents,expected_pages,length,overlap",
    [args[1:] for args in split_by_regex_no_headlines_args],
    ids=[i[0] for i in split_by_regex_no_headlines_args],
)
def test_split_by_regex_no_headlines(preprocessor, document, expected_documents, expected_pages, length, overlap):
    split_documents = preprocessor.splitter.run(
        documents=[Document(content=document)],
        split_by="regex",
        split_regex="(~(header|[0-9]*)~|___(footer|[0-9]*)___)",
        split_length=length,
        split_overlap=overlap,
        split_max_chars=500,
        add_page_number=True,
    )[0]["documents"]

    assert expected_documents == [document.content for document in split_documents]
    assert expected_pages == [document.meta["page"] for document in split_documents]


split_by_regex_with_headlines_args = [
    (
        "No-op",
        "TITLE1: No matches\fTITLE2: No splitting",
        ["TITLE1: No matches\fTITLE2: No splitting"],
        [{"content": "TITLE1", "start_idx": 0}, {"content": "TITLE3", "start_idx": 20}],
        [[{"content": "TITLE1", "start_idx": 0}, {"content": "TITLE3", "start_idx": 20}]],
        [1],
        1,
        0,
    ),
    (
        "No-op with overlap",
        "TITLE1: No matches\fTITLE2: No splitting",
        ["TITLE1: No matches\fTITLE2: No splitting"],
        [{"content": "TITLE1", "start_idx": 0}, {"content": "TITLE3", "start_idx": 20}],
        [[{"content": "TITLE1", "start_idx": 0}, {"content": "TITLE3", "start_idx": 20}]],
        [1],
        10,
        5,
    ),
    (
        "Simplest case",
        "TITLE1~header~Page1___footer___a\fPage2 TITLE3___footer___more text",
        ["TITLE1~header~", "Page1___footer___", "a\fPage2 TITLE3___footer___", "more text"],
        [{"content": "TITLE1", "start_idx": 0}, {"content": "TITLE3", "start_idx": 39}],
        [[{"content": "TITLE1", "start_idx": 0}], [], [{"content": "TITLE3", "start_idx": 8}], []],
        [1, 1, 1, 2],
        1,
        0,
    ),
    (
        "Group by 2",
        "TITLE1~header~\fPage1___footer___Page2 TITLE3___footer___more text",
        ["TITLE1~header~\fPage1___footer___", "Page2 TITLE3___footer___more text"],
        [{"content": "TITLE1", "start_idx": 0}, {"content": "TITLE3", "start_idx": 38}],
        [[{"content": "TITLE1", "start_idx": 0}], [{"content": "TITLE3", "start_idx": 6}]],
        [1, 2],
        2,
        0,
    ),
    (
        "Group by 3",
        "TITLE1~header~Page1___footer___\fPage2 TITLE3___footer___more text",
        ["TITLE1~header~Page1___footer___\fPage2 TITLE3___footer___", "more text"],
        [{"content": "TITLE1", "start_idx": 0}, {"content": "TITLE3", "start_idx": 38}],
        [[{"content": "TITLE1", "start_idx": 0}, {"content": "TITLE3", "start_idx": 38}], []],
        [1, 2],
        3,
        0,
    ),
    (
        "Group by more units than present",
        "TITLE1~header~Page1___footer___\fPage2 TITLE3___footer___more text",
        ["TITLE1~header~Page1___footer___\fPage2 TITLE3___footer___more text"],
        [{"content": "TITLE1", "start_idx": 0}, {"content": "TITLE3", "start_idx": 38}],
        [[{"content": "TITLE1", "start_idx": 0}, {"content": "TITLE3", "start_idx": 38}]],
        [1],
        10,
        0,
    ),
    (
        "Title not in fist docs",
        "Not a title~header~Page1\f___footer___Page2 TITLE3~header~more text___footer___",
        ["Not a title~header~", "Page1\f___footer___", "Page2 TITLE3~header~", "more text___footer___"],
        [{"content": "TITLE3", "start_idx": 43}],
        [[], [], [{"content": "TITLE3", "start_idx": 6}], []],
        [1, 1, 2, 2],
        1,
        0,
    ),
    (
        "Overlapping sections with headlines",
        "doc___111111___another doc\n~222222~TITLE1___3___Page1\f~444444~Page2 TITLE3___555___more text~66~",
        [
            "doc___111111___another doc\n~222222~",
            "another doc\n~222222~TITLE1___3___",
            "TITLE1___3___Page1\f~444444~",
            "Page1\f~444444~Page2 TITLE3___555___",
            "Page2 TITLE3___555___more text~66~",
        ],
        [{"content": "TITLE1", "start_idx": 35}, {"content": "TITLE3", "start_idx": 68}],
        [
            [],
            [{"content": "TITLE1", "start_idx": 20}],
            [{"content": "TITLE1", "start_idx": 0}],
            [{"content": "TITLE3", "start_idx": 20}],
            [{"content": "TITLE3", "start_idx": 6}],
        ],
        [1, 1, 1, 1, 2],
        2,
        1,
    ),
    (
        "Overlapping sections with many headlines",
        "doc___111111___TITLE1___222222___a TITLE2\nPage1\f~333333~Page2 TITLE3___444444___more text~55~",
        [
            "doc___111111___TITLE1___222222___",
            "TITLE1___222222___a TITLE2\nPage1\f~333333~",
            "a TITLE2\nPage1\f~333333~Page2 TITLE3___444444___",
            "Page2 TITLE3___444444___more text~55~",
        ],
        [
            {"content": "TITLE1", "start_idx": 15},
            {"content": "TITLE2", "start_idx": 35},
            {"content": "TITLE3", "start_idx": 62},
        ],
        [
            [{"content": "TITLE1", "start_idx": 15}],
            [{"content": "TITLE1", "start_idx": 0}, {"content": "TITLE2", "start_idx": 20}],
            [{"content": "TITLE2", "start_idx": 2}, {"content": "TITLE3", "start_idx": 29}],
            [{"content": "TITLE3", "start_idx": 6}],
        ],
        [1, 1, 1, 2],
        2,
        1,
    ),
]


@pytest.mark.parametrize(
    "document,expected_documents,headlines,expected_headlines,expected_pages,length,overlap",
    [args[1:] for args in split_by_regex_with_headlines_args],
    ids=[i[0] for i in split_by_regex_with_headlines_args],
)
def test_split_by_regex_with_headlines(
    preprocessor, document, expected_documents, headlines, expected_headlines, expected_pages, length, overlap
):
    split_documents = preprocessor.splitter.run(
        documents=[Document(content=document, meta={"headlines": headlines})],
        split_by="regex",
        split_regex="(~(header|[0-9]*)~|___(footer|[0-9]*)___)",
        split_length=length,
        split_overlap=overlap,
        split_max_chars=500,
        add_page_number=True,
    )[0]["documents"]

    assert expected_documents == [(document.content) for document in split_documents]
    assert expected_pages == [(document.meta["page"]) for document in split_documents]
    assert expected_headlines == [(document.meta["headlines"]) for document in split_documents]


def test_split_by_regex_above_max_chars_single_unit_no_headlines(preprocessor):
    split_documents = preprocessor.splitter.run(
        documents=[Document(content="Very long content that goes\fmany times above the value of max_chars~header~")],
        split_by="regex",
        split_regex="(~header~|___footer___)",
        split_length=1,
        split_overlap=0,
        split_max_chars=20,
        add_page_number=True,
    )[0]["documents"]
    assert ["Very long content th", "at goes\fmany times a", "bove the value of ma", "x_chars~header~"] == [
        doc.content for doc in split_documents
    ]
    assert [1, 1, 2, 2] == [doc.meta["page"] for doc in split_documents]


def test_split_by_regex_above_max_chars_no_overlap_no_headlines(preprocessor):
    split_documents = preprocessor.splitter.run(
        documents=[
            Document(
                content="Para\n very long content that goes\fabove the value of max_chars___footer___Short!\f~header~Para 3 this is also quite long___footer___"
            )
        ],
        split_by="regex",
        split_regex="(~header~|___footer___)",
        split_length=1,
        split_overlap=0,
        split_max_chars=20,
        add_page_number=True,
    )[0]["documents"]
    assert [
        "Para\n very long cont",
        "ent that goes\fabove ",
        "the value of max_cha",
        "rs___footer___",
        "Short!\f~header~",
        "Para 3 this is also ",
        "quite long___footer_",
        "__",
    ] == [doc.content for doc in split_documents]
    assert [1, 1, 2, 2, 2, 3, 3, 3] == [doc.meta["page"] for doc in split_documents]


def test_split_by_regex_above_max_chars_with_overlap_no_headlines(preprocessor):
    split_documents = preprocessor.splitter.run(
        documents=[
            Document(
                content="Para1 very long content that goes\fabove the value of max_chars___footer___Para2~header~Para3~header~Para4 this is also quite long"
            )
        ],
        split_by="regex",
        split_regex="(~header~|___footer___)",
        split_length=2,
        split_overlap=1,
        split_max_chars=20,
        add_page_number=True,
    )[0]["documents"]
    assert [
        "Para1 very long cont",
        "ent that goes\fabove ",
        "the value of max_cha",
        "rs___footer___Para2~",
        "header~",
        "Para2~header~Para3~h",
        "eader~",
        "Para3~header~Para4 t",
        "his is also quite lo",
        "ng",
    ] == [doc.content for doc in split_documents]
    assert [1, 1, 2, 2, 2, 2, 2, 2, 2, 2] == [doc.meta["page"] for doc in split_documents]


def test_split_by_regex_above_max_chars_single_unit_with_headlines(preprocessor):
    split_documents = preprocessor.splitter.run(
        documents=[
            Document(
                content="Para1 very long content that goes\fabove the value of max_chars___footer___",
                meta={"headlines": [{"content": "Para1", "start_idx": 0}, {"content": "value", "start_idx": 44}]},
            )
        ],
        split_by="regex",
        split_regex="(~header~|___footer___)",
        split_length=1,
        split_overlap=0,
        split_max_chars=20,
        add_page_number=True,
    )[0]["documents"]
    assert ["Para1 very long cont", "ent that goes\fabove ", "the value of max_cha", "rs___footer___"] == [
        doc.content for doc in split_documents
    ]
    assert [1, 1, 2, 2] == [doc.meta["page"] for doc in split_documents]
    assert [[{"content": "Para1", "start_idx": 0}], [], [{"content": "value", "start_idx": 4}], []] == [
        doc.meta["headlines"] for doc in split_documents
    ]


def test_split_by_regex_above_max_chars_no_overlap_with_headlines(preprocessor):
    split_documents = preprocessor.splitter.run(
        documents=[
            Document(
                content="Para1 very long content that goes\fabove the value of max_chars~header~Short!___footer___\fPara3 is also quite long___footer___",
                meta={
                    "headlines": [
                        {"content": "Para1", "start_idx": 0},
                        {"content": "value", "start_idx": 44},
                        {"content": "Short!", "start_idx": 70},
                        {"content": "Para3", "start_idx": 89},
                        {"content": "also", "start_idx": 98},
                        {"content": "long", "start_idx": 114},
                    ]
                },
            )
        ],
        split_by="regex",
        split_regex="(~header~|___footer___)",
        split_length=1,
        split_overlap=0,
        split_max_chars=20,
        add_page_number=True,
    )[0]["documents"]
    assert [
        "Para1 very long cont",
        "ent that goes\fabove ",
        "the value of max_cha",
        "rs~header~",
        "Short!___footer___",
        "\fPara3 is also quite",
        " long___footer___",
    ] == [doc.content for doc in split_documents]
    assert [1, 1, 2, 2, 2, 2, 3] == [doc.meta["page"] for doc in split_documents]
    assert [
        [{"content": "Para1", "start_idx": 0}],
        [],
        [{"content": "value", "start_idx": 4}],
        [],
        [{"content": "Short!", "start_idx": 0}],
        [{"content": "Para3", "start_idx": 1}, {"content": "also", "start_idx": 10}],
        [{"content": "long", "start_idx": 6}],
    ] == [doc.meta["headlines"] for doc in split_documents]


def test_split_by_regex_above_max_chars_with_overlap_with_headlines(preprocessor):
    split_documents = preprocessor.splitter.run(
        documents=[
            Document(
                content="Para1 very long content that goes\fabove the value of max_chars___footer___Para2~header~\fPara3~header~Para4 this is also quite long",
                meta={
                    "headlines": [
                        {"content": "Para1", "start_idx": 0},
                        {"content": "value", "start_idx": 44},
                        {"content": "Para2", "start_idx": 74},
                        {"content": "Para3", "start_idx": 88},
                        {"content": "Para4", "start_idx": 101},
                        {"content": "this", "start_idx": 107},
                    ]
                },
            )
        ],
        split_by="regex",
        split_regex="(~header~|___footer___)",
        split_length=2,
        split_overlap=1,
        split_max_chars=20,
        add_page_number=True,
    )[0]["documents"]
    assert [
        "Para1 very long cont",
        "ent that goes\fabove ",
        "the value of max_cha",
        "rs___footer___Para2~",
        "header~",
        "Para2~header~\fPara3~",
        "header~",
        "\fPara3~header~Para4 ",
        "this is also quite l",
        "ong",
    ] == [doc.content for doc in split_documents]
    assert [1, 1, 2, 2, 2, 2, 3, 2, 3, 3] == [doc.meta["page"] for doc in split_documents]
    assert [
        [{"content": "Para1", "start_idx": 0}],
        [],
        [{"content": "value", "start_idx": 4}],
        [{"content": "Para2", "start_idx": 14}],
        [],
        [{"content": "Para2", "start_idx": 0}, {"content": "Para3", "start_idx": 14}],
        [],
        [{"content": "Para3", "start_idx": 1}, {"content": "Para4", "start_idx": 14}],
        [{"content": "this", "start_idx": 0}],
        [],
    ] == [doc.meta["headlines"] for doc in split_documents]


def test_split_by_regex_above_max_chars_with_overlap_page_backtracking(preprocessor):
    split_documents = preprocessor.splitter.run(
        documents=[
            Document(
                content="Para1 very long content that goes\fabove the value of max_chars___footer___Para2~header~a\f\f\f\fra3~header~Para4 that's also quite long",
                meta={
                    "headlines": [
                        {"content": "Para1", "start_idx": 0},
                        {"content": "value", "start_idx": 44},
                        {"content": "Para2", "start_idx": 74},
                        {"content": "ra3", "start_idx": 92},
                        {"content": "Para4", "start_idx": 103},
                        {"content": "that's", "start_idx": 109},
                    ]
                },
            )
        ],
        split_by="regex",
        split_regex="(~header~|___footer___)",
        split_length=2,
        split_overlap=1,
        split_max_chars=20,
        add_page_number=True,
    )[0]["documents"]
    assert [
        "Para1 very long cont",
        "ent that goes\fabove ",
        "the value of max_cha",
        "rs___footer___Para2~",
        "header~",
        "Para2~header~a\f\f\f\fra",
        "3~header~",
        "a\f\f\f\fra3~header~Para",
        "4 that's also quite ",
        "long",
    ] == [doc.content for doc in split_documents]
    # Notice how the overlap + hard break make the page number go backwards
    assert [1, 1, 2, 2, 2, 2, 6, 2, 6, 6] == [doc.meta["page"] for doc in split_documents]
    assert [
        [{"content": "Para1", "start_idx": 0}],
        [],
        [{"content": "value", "start_idx": 4}],
        [{"content": "Para2", "start_idx": 14}],
        [],
        [{"content": "Para2", "start_idx": 0}, {"content": "ra3", "start_idx": 18}],
        [],
        [{"content": "ra3", "start_idx": 5}, {"content": "Para4", "start_idx": 16}],
        [{"content": "that's", "start_idx": 2}],
        [],
    ] == [doc.meta["headlines"] for doc in split_documents]


# Few additional tests on the page split to double-check on the page count algorithm
split_by_page_args = [
    ("No-op", "Don't split me", ["Don't split me"], [1], 1, 0),
    ("No-op with overlap", "Don't split me", ["Don't split me"], [1], 10, 5),
    ("Simplest case", "Page1\fPage2\fPage3\fPage4", ["Page1\f", "Page2\f", "Page3\f", "Page4"], [1, 2, 3, 4], 1, 0),
    (
        "Empty pages",
        "Page1\fPage2\fPage3\f\fPage5",
        ["Page1\f", "Page2\f", "Page3\f", "\f", "Page5"],
        [1, 2, 3, 4, 5],
        1,
        0,
    ),
    (
        "Several empty documents in a row are removed",
        "Page1\fPage2\fPage3\f\f\f\fPage7\f\f\f\f",
        ["Page1\f", "Page2\f", "Page3\f", "\f", "\f", "\f", "Page7\f", "\f", "\f", "\f"],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        1,
        0,
    ),
    (
        "Overlap with many empty pages",
        "Page1\fPage2\fPage3\fPage4\f\f\f\fPage8\f\f\f\f\f",
        [
            "Page1\fPage2\f",
            "Page2\fPage3\f",
            "Page3\fPage4\f",
            "Page4\f\f",
            "\f\f",
            "\f\f",
            "\fPage8\f",
            "Page8\f\f",
            "\f\f",
            "\f\f",
            "\f\f",
        ],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        2,
        1,
    ),
]


@pytest.mark.parametrize(
    "document,expected_documents,expected_pages,length,overlap",
    [args[1:] for args in split_by_page_args],
    ids=[i[0] for i in split_by_page_args],
)
def test_split_by_page(preprocessor, document, expected_documents, expected_pages, length, overlap):
    split_documents = preprocessor.splitter.run(
        documents=[Document(content=document)],
        split_by="page",
        split_length=length,
        split_overlap=overlap,
        split_max_chars=50,
        add_page_number=True,
    )[0]["documents"]
    assert expected_documents == [document.content for document in split_documents]
    assert expected_pages == [document.meta["page"] for document in split_documents]


# Few additional test on the paragraph split to make sure it works only when intended
split_by_paragraph_args = [
    (
        "Page breaks don't matter",
        "Paragraph 1\n\nParagraph\f2\n\nParagraph 3\n\nParagraph 4",
        ["Paragraph 1\n\n", "Paragraph\f2\n\n", "Paragraph 3\n\n", "Paragraph 4"],
        [1, 1, 2, 2],
        1,
        0,
    ),
    (
        "Single return chars don't do anything",
        "Paragraph 1\nStill Paragraph 1\n\n\fParagraph 2\n\nParagraph 3",
        ["Paragraph 1\nStill Paragraph 1\n\n", "\fParagraph 2\n\n", "Paragraph 3"],
        [1, 1, 2],
        1,
        0,
    ),
]


@pytest.mark.parametrize(
    "document,expected_documents,expected_pages,length,overlap",
    [args[1:] for args in split_by_paragraph_args],
    ids=[i[0] for i in split_by_paragraph_args],
)
def test_split_by_paragraph(preprocessor, document, expected_documents, expected_pages, length, overlap):
    split_documents = preprocessor.splitter.run(
        documents=[Document(content=document)],
        split_by="paragraph",
        split_length=length,
        split_overlap=overlap,
        split_max_chars=50,
        add_page_number=True,
    )[0]["documents"]
    assert expected_documents == [document.content for document in split_documents]
    assert expected_pages == [document.meta["page"] for document in split_documents]


#
# Token-based splits
#

split_by_sentence_args = [
    ("Empty string", "", [""]),
    ("Whitespace is kept", "    ", ["    "]),
    ("Single word", "test", ["test"]),
    (
        "Single sentence, no punctuation",
        "a single sentence with no punctuation",
        ["a single sentence with no punctuation"],
    ),
    (
        "Single sentence with punctuation",
        "A single sentence, with some punctuation!",
        ["A single sentence, with some punctuation!"],
    ),
    ("Simple sentences", "One sentence. Another sentence.", ["One sentence. ", "Another sentence."]),
    (
        "Simple sentences with lots of whitespace",
        "   \n   One sentence. \f\t    \n Another sentence\n\n.\n\n    \t \f",
        ["   \n   One sentence. \f\t    \n ", "Another sentence\n\n.\n\n    \t \f"],
    ),
    (
        "Simple sentences with punctuation",
        "One sentence. And this... is another sentence.",
        ["One sentence. ", "And this... is another sentence."],
    ),
    (
        "Simple sentences with wrong punctuation",
        "One sentence.And this... is another sentence.",
        ["One sentence.And this... is another sentence."],
    ),
    ("Acronyms", "This is the U.K. While this is the U.S.A.", ["This is the U.K. ", "While this is the U.S.A."]),
    (
        "Correctly used brackets",
        "Here I (will) use some brackets. Here I won't.",
        ["Here I (will) use some brackets. ", "Here I won't."],
    ),
    (
        "Wrongly used brackets",
        "Here I (will use some brackets. Here I) won't.",
        ["Here I (will use some brackets. ", "Here I) won't."],
    ),
    (
        "Not English, Latin script",
        "Questa frase contiene una S.I.G.L.A. Questa è una prova.",
        ["Questa frase contiene una S.I.G.L.A. ", "Questa è una prova."],
    ),
    (
        "Not English, non-Latin alphabetical script",
        "Это Т.Е.С.Т. А это еще один простой тест. ",
        ["Это Т.Е.С.Т. ", "А это еще один простой тест. "],
    ),
]


@pytest.mark.parametrize(
    "document,expected_documents",
    [args[1:] for args in split_by_sentence_args],
    ids=[i[0] for i in split_by_sentence_args],
)
def test_split_by_sentence(preprocessor, document, expected_documents):
    split_documents = preprocessor.splitter.run(
        documents=[Document(content=document)],
        split_by="sentence",
        split_length=1,
        split_overlap=0,
        split_max_chars=500,
        add_page_number=True,
    )[0]["documents"]
    assert expected_documents == [document.content for document in split_documents]


def test_split_by_sentence_chinese(preprocessor):
    split_documents = preprocessor.splitter.run(
        documents=[Document(content="今天不是晴天，因为下雨了。 昨天天气比较好。")],
        split_by="word",
        split_length=1,
        split_overlap=0,
        split_max_chars=500,
        add_page_number=True,
        tokenizer=load_tokenizer(language="chinese"),
    )[0]["documents"]
    assert [document.content for document in split_documents] == ["今天不是晴天，因为下雨了。 ", "昨天天气比较好。"]


split_by_word_args = [
    ("Empty string", "", [""]),
    ("Whitespace is kept", "    ", ["    "]),
    ("Single word", "test", ["test"]),
    ("Single word with whitespace", "  test    ", ["  ", "test    "]),
    ("Sentence with whitespace", " This is a test    ", [" ", "This ", "is ", "a ", "test    "]),
    ("Sentence with punctuation", " This, is a test.    ", [" ", "This", ", ", "is ", "a ", "test", ".    "]),
    (
        "Sentence with strange punctuation",
        " This!! is a test...!..()    ",
        [" ", "This", "!", "! ", "is ", "a ", "test", "...", "!", "..", "(", ")    "],
    ),
    ("Sentence with units of measure prefixed", "This is $3.08", ["This ", "is ", "$", "3.08"]),
    ("Sentence with units of measure postfixed", "This is 3.08$", ["This ", "is ", "3.08", "$"]),
    ("Contractions", "This isn't a test", ["This ", "is", "n't ", "a ", "test"]),
    (
        "Acronyms",
        "This is the U.K. and this is the U.S.A.",
        ["This ", "is ", "the ", "U.K. ", "and ", "this ", "is ", "the ", "U.S.A", "."],
    ),
    ("Brackets", "This is (in brackets)", ["This ", "is ", "(", "in ", "brackets", ")"]),
    (
        "Weirder brackets",
        "[(This)] is <a variable> {} ( spaced )",
        ["[", "(", "This", ")", "] ", "is ", "<", "a ", "variable", "> ", "{", "} ", "( ", "spaced ", ")"],
    ),
    ("Other language with Latin script", " Questo è un test.  ", [" ", "Questo ", "è ", "un ", "test", ".  "]),
    (
        "Other language with alphabetic, non-latin script",
        "  Это простой тест.  ",
        ["  ", "Это ", "простой ", "тест", ".  "],
    ),
]


@pytest.mark.parametrize(
    "document,expected_documents", [args[1:] for args in split_by_word_args], ids=[i[0] for i in split_by_word_args]
)
def test_split_by_word(preprocessor, document, expected_documents):
    split_documents = preprocessor.splitter.run(
        documents=[Document(content=document)],
        split_by="word",
        split_length=1,
        split_overlap=0,
        split_max_chars=500,
        add_page_number=True,
    )[0]["documents"]
    assert expected_documents == [document.content for document in split_documents]


@pytest.mark.xfail(reason="NLTK can't tokenize Chinese text by word.")
def test_split_by_word_chinese(preprocessor):
    split_documents = preprocessor.splitter.run(
        documents=[Document(content="今天不是晴天，因为下雨了。")],
        split_by="word",
        split_length=1,
        split_overlap=0,
        split_max_chars=500,
        add_page_number=True,
        tokenizer=load_tokenizer(language="chinese"),
    )[0]["documents"]
    assert [document.content for document in split_documents] == ["今天", "不", "是", "晴", "天，", "因为", "下雨", "了。"]
