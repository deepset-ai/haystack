from typing import Set, List

import pytest

from haystack import Document
from haystack.nodes.preprocessor.preprocessor import PreProcessor
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
