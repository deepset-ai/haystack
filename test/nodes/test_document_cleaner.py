from typing import Set, List

import pytest

from haystack import Document
from haystack.nodes.preprocessor.cleaner import DocumentCleaner, longest_common_prefix, longest_common_suffix


@pytest.fixture
def cleaner():
    # Note: this are all simply fallback values.
    # Each test will call directly either run providing the required input parameters.
    # If testing DocumentCleaner.__init__() they should not use this fixture
    return DocumentCleaner(clean_whitespace=True, clean_empty_lines=True, clean_header_footer=True)


#
# Validations
#


def test_init_with_wrong_header_footer_n_chars():
    with pytest.raises(ValueError, match="header_footer_n_chars"):
        DocumentCleaner(
            clean_whitespace=True, clean_empty_lines=True, clean_header_footer=True, header_footer_n_chars=-1
        )
    with pytest.raises(ValueError, match="header_footer_n_chars"):
        DocumentCleaner(
            clean_whitespace=True, clean_empty_lines=True, clean_header_footer=True, header_footer_n_chars=0.5
        )


def test_init_with_wrong_header_footer_pages_to_ignore():
    with pytest.raises(ValueError, match="header_footer_pages_to_ignore"):
        DocumentCleaner(
            clean_whitespace=True, clean_empty_lines=True, clean_header_footer=True, header_footer_pages_to_ignore=2
        )
    with pytest.raises(ValueError, match="header_footer_pages_to_ignore"):
        DocumentCleaner(
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_pages_to_ignore=[1, 2, 3, 0.4, 5],
        )
    with pytest.raises(ValueError, match="header_footer_pages_to_ignore"):
        DocumentCleaner(
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_pages_to_ignore=[1, 0.2, 3, -0.4, -5],
        )
    # Negative values are ok, they are counted from the end of the array.
    DocumentCleaner(
        clean_whitespace=True,
        clean_empty_lines=True,
        clean_header_footer=True,
        header_footer_pages_to_ignore=[1, 2, 3, -4, 5],
    )


def test_run_with_wrong_header_footer_n_chars(cleaner: DocumentCleaner):
    with pytest.raises(ValueError, match="header_footer_n_chars"):
        cleaner.run(
            documents=[Document(content="test")],
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_n_chars=-1,
        )
    with pytest.raises(ValueError, match="header_footer_n_chars"):
        cleaner.run(
            documents=[Document(content="test")],
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_n_chars=0.5,
        )


def test_run_with_wrong_header_footer_pages_to_ignore(cleaner: DocumentCleaner):
    with pytest.raises(ValueError, match="header_footer_pages_to_ignore"):
        cleaner.run(
            documents=[Document(content="test")],
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_pages_to_ignore=2,
        )
    with pytest.raises(ValueError, match="header_footer_pages_to_ignore"):
        cleaner.run(
            documents=[Document(content="test")],
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_pages_to_ignore=[1, 2, 3, 0.4, 5],
        )
    with pytest.raises(ValueError, match="header_footer_pages_to_ignore"):
        cleaner.run(
            documents=[Document(content="test")],
            clean_whitespace=True,
            clean_empty_lines=True,
            clean_header_footer=True,
            header_footer_pages_to_ignore=[1, 0.2, 3, -0.4, -5],
        )
    # Negative values are ok, they are counted from the end of the array.
    cleaner.run(
        documents=[Document(content="test")],
        clean_whitespace=True,
        clean_empty_lines=True,
        clean_header_footer=True,
        header_footer_pages_to_ignore=[1, 2, 3, -4, 5],
    )


#
# Prefix/suffix matching tests
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


#
# Regex replacements
#

remove_regex_matches_args = [
    ("Empty group causes no-op, no headers", "()", "abcdefg", "abcdefg", None, None),
    ("Single char string, no headers", "(A)", "aAbc", "abc", None, None),
    ("Simple group, no headers", "([0-9]*)", "a135b56c00", "abc", None, None),
    ("Multichar string group, no headers", "(AA)", "AAAAabcdAAAefgAhAA", "abcdAefgAh", None, None),
    ("Unicode/emoji match, no headers", "(🪲)", "abcd🪲efgh🪲", "abcdefgh", None, None),
    (
        "Unicode/emoji match does not affect other unicode/emoji, no headers",
        "(🪲)",
        "✨abc✨d🪲ef✨gh🪲",
        "✨abc✨def✨gh",
        None,
        None,
    ),
    ("Do not affect whitespace, no headers", "(🪲)", "\b\b\babc\n\nde🪲\f\nfgh", "\b\b\babc\n\nde\f\nfgh", None, None),
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
        "Multichar string with headers, single removal after header",
        "(aa)",
        "a # Test header aa caa def",
        "a # Test header  c def",
        [{"headline": "# Test header", "start_idx": 2}],
        [{"headline": "# Test header", "start_idx": 2}],
    ),
    (
        "Multichar string with headers, many removals in a row before header",
        "(aa )",
        "aa aa aa aa aa  # Test header a caa  def aa aa ",
        " # Test header a c def ",
        [{"headline": "# Test header", "start_idx": 16}],
        [{"headline": "# Test header", "start_idx": 1}],
    ),
    (
        "Multichar string with headers, many headers after removal",
        "(aa )",
        "aa # Test1header a # Test2header # Test3header cdef aa # Test4header ",
        "# Test1header a # Test2header # Test3header cdef # Test4header ",
        [
            {"headline": "# Test1header", "start_idx": 3},
            {"headline": "# Test2header", "start_idx": 19},
            {"headline": "# Test3header", "start_idx": 33},
            {"headline": "# Test4header", "start_idx": 55},
        ],
        [
            {"headline": "# Test1header", "start_idx": 0},
            {"headline": "# Test2header", "start_idx": 16},
            {"headline": "# Test3header", "start_idx": 30},
            {"headline": "# Test4header", "start_idx": 49},
        ],
    ),
    (
        "The removal affects the header lenght, no other headers follow",
        "(Test)",
        "a # Test header b c def",
        "a #  header b c def",
        [{"headline": "# Test header", "start_idx": 2}],
        [{"headline": "#  header", "start_idx": 2}],
    ),
    (
        "The removal affects the header lenght, other headers follow",
        "(Test)",
        "a # Test header 1 b c # Test header 2 def",
        "a #  header 1 b c #  header 2 def",
        [{"headline": "# Test header 2", "start_idx": 22}, {"headline": "# Test header 1", "start_idx": 2}],
        [{"headline": "#  header 1", "start_idx": 2}, {"headline": "#  header 2", "start_idx": 18}],
    ),
    (
        "The removal affects a header multiple times",
        "(Test)",
        "a # Test Test Test header 1 b c # Test header Test 2 def",
        "a #    header 1 b c #  header  2 def",
        [
            {"headline": "# Test header Test 2", "start_idx": 32},
            {"headline": "# Test Test Test header 1", "start_idx": 2},
        ],
        [{"headline": "#    header 1", "start_idx": 2}, {"headline": "#  header  2", "start_idx": 20}],
    ),
    (
        "Header/footer removal example with removal affecting multiple headers",
        "(the header|the footer|Test)",
        "the header\na\n# Test header 1\nb c\n# Test header 2\ndef\nthe footer",
        "\na\n#  header 1\nb c\n#  header 2\ndef\n",
        [{"headline": "# Test header 2", "start_idx": 33}, {"headline": "# Test header 1", "start_idx": 13}],
        [{"headline": "#  header 1", "start_idx": 3}, {"headline": "#  header 2", "start_idx": 19}],
    ),
    (
        "Variable header/footer removal example with removal affecting multiple headers",
        "(the [0-9]*th header|the [0-9]*th footer|Test [0-9]* )",
        "the 4th header\na\n# Test 234 header 1\nb c\n# Test 33 header 2\ndef\nthe 80th footer",
        "\na\n# header 1\nb c\n# header 2\ndef\n",
        [{"headline": "# Test 33 header 2", "start_idx": 41}, {"headline": "# Test 234 header 1", "start_idx": 17}],
        [{"headline": "# header 1", "start_idx": 3}, {"headline": "# header 2", "start_idx": 18}],
    ),
]


@pytest.mark.parametrize(
    "regex,text,clean_text,headlines,clean_headlines",
    [args[1:] for args in remove_regex_matches_args],
    ids=[i[0] for i in remove_regex_matches_args],
)
def test_remove_regex_matches(cleaner: DocumentCleaner, regex, text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = cleaner.replace_regex_matches(doc_to_clean, pattern=regex, replacement="")

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


replace_regex_matches_args = [
    ("Empty group causes no-op, no headers", "()", "abcdefg", "abcdefg", None, None),
    ("Single char string, no headers", "(A)", "aAbc", "a@@bc", None, None),
    ("Simple group, no headers", "([0-9]+)", "a135b56c00", "a@@b@@c@@", None, None),
    ("Multichar string group, no headers", "(AA)", "AAAAabcdAAAefgAhAA", "@@@@abcd@@AefgAh@@", None, None),
    ("Unicode/emoji match, no headers", "(🪲)", "abcd🪲efgh🪲", "abcd@@efgh@@", None, None),
    (
        "Unicode/emoji match does not affect other unicode/emoji, no headers",
        "(🪲)",
        "✨abc✨d🪲ef✨gh🪲",
        "✨abc✨d@@ef✨gh@@",
        None,
        None,
    ),
    ("Do not affect whitespace, no headers", "(🪲)", "\b\b\babc\n\nde🪲\f\nfgh", "\b\b\babc\n\nde@@\f\nfgh", None, None),
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
        [{"headline": "# Test header", "start_idx": 2}],
        [{"headline": "# Test header", "start_idx": 2}],
    ),
    (
        "multichar string with headers, many removals in a row before header",
        "(aa )",
        "aa aa aa aa aa  # Test header a caa  def aa aa ",
        "@@@@@@@@@@ # Test header a c@@ def @@@@",
        [{"headline": "# Test header", "start_idx": 16}],
        [{"headline": "# Test header", "start_idx": 11}],
    ),
    (
        "multichar string with headers, many headers after removal",
        "(aa )",
        "aa # Test header a # Test header # Test header cdef aa # Test header ",
        "@@# Test header a # Test header # Test header cdef @@# Test header ",
        [
            {"headline": "# Test header", "start_idx": 3},
            {"headline": "# Test header", "start_idx": 19},
            {"headline": "# Test header", "start_idx": 33},
            {"headline": "# Test header", "start_idx": 55},
        ],
        [
            {"headline": "# Test header", "start_idx": 2},
            {"headline": "# Test header", "start_idx": 18},
            {"headline": "# Test header", "start_idx": 32},
            {"headline": "# Test header", "start_idx": 53},
        ],
    ),
    (
        "The removal affects the header lenght, no other headers follow",
        "(Test)",
        "a # Test header b c def",
        "a # @@ header b c def",
        [{"headline": "# Test header", "start_idx": 2}],
        [{"headline": "# @@ header", "start_idx": 2}],
    ),
    (
        "The removal affects the header lenght, other headers follow",
        "(Test)",
        "a # Test header 1 b c # Test header 2 def",
        "a # @@ header 1 b c # @@ header 2 def",
        [{"headline": "# Test header 2", "start_idx": 22}, {"headline": "# Test header 1", "start_idx": 2}],
        [{"headline": "# @@ header 1", "start_idx": 2}, {"headline": "# @@ header 2", "start_idx": 20}],
    ),
    (
        "The removal affects a header multiple times",
        "(Test)",
        "a # Test Test Test header 1 b c # Test header Test 2 def",
        "a # @@ @@ @@ header 1 b c # @@ header @@ 2 def",
        [
            {"headline": "# Test header Test 2", "start_idx": 32},
            {"headline": "# Test Test Test header 1", "start_idx": 2},
        ],
        [{"headline": "# @@ @@ @@ header 1", "start_idx": 2}, {"headline": "# @@ header @@ 2", "start_idx": 26}],
    ),
    (
        "Header/footer removal example with removal affecting multiple headers",
        "(the header|the footer|Test)",
        "the header\na\n# Test header 1\nb c\n# Test header 2\ndef\nthe footer",
        "@@\na\n# @@ header 1\nb c\n# @@ header 2\ndef\n@@",
        [{"headline": "# Test header 2", "start_idx": 33}, {"headline": "# Test header 1", "start_idx": 13}],
        [{"headline": "# @@ header 1", "start_idx": 5}, {"headline": "# @@ header 2", "start_idx": 23}],
    ),
    (
        "Variable header/footer removal example with removal affecting multiple headers",
        "(the [0-9]*th header|the [0-9]*th footer|Test [0-9]* )",
        "the 4th header\na\n# Test 234 header 1\nb c\n# Test 33 header 2\ndef\nthe 80th footer",
        "@@\na\n# @@header 1\nb c\n# @@header 2\ndef\n@@",
        [{"headline": "# Test 33 header 2", "start_idx": 41}, {"headline": "# Test 234 header 1", "start_idx": 17}],
        [{"headline": "# @@header 1", "start_idx": 5}, {"headline": "# @@header 2", "start_idx": 22}],
    ),
]


@pytest.mark.parametrize(
    "regex,text,clean_text,headlines,clean_headlines",
    [args[1:] for args in replace_regex_matches_args],
    ids=[i[0] for i in replace_regex_matches_args],
)
def test_replace_regex_matches(cleaner: DocumentCleaner, regex, text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = cleaner.replace_regex_matches(doc_to_clean, pattern=regex, replacement="@@")

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


#
# Cleaning tests
#

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
        [{"headline": "First headline", "start_idx": 23}, {"headline": "Second headline", "start_idx": 156}],
        [{"headline": "First headline", "start_idx": 0}, {"headline": "Second headline", "start_idx": 84}],
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
        [{"headline": "First headline", "start_idx": 97}, {"headline": "Second headline", "start_idx": 304}],
        [{"headline": "First headline", "start_idx": 47}, {"headline": "Second headline", "start_idx": 178}],
    ),
    (
        "Headlines that are also headers, or within headers, are removed",
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
            {"headline": "--- Headline ---", "start_idx": 0},
            {"headline": "Headline", "start_idx": 118},
            {"headline": "### Sub headline", "start_idx": 131},
        ],
        [{"headline": "### Sub headline", "start_idx": 69}],
    ),
    (
        "Headlines that are also footers, or within footers, are removed",
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
            {"headline": "~~~~~ Footer Headline ~~~~~", "start_idx": 91},
            {"headline": "Footer Headline", "start_idx": 241},
            {"headline": "### Sub headline", "start_idx": 141},
        ],
        [{"headline": "### Sub headline", "start_idx": 69}],
    ),
]


@pytest.mark.parametrize(
    "text,clean_text,headlines,clean_headlines",
    [args[1:] for args in remove_header_footer_args],
    ids=[i[0] for i in remove_header_footer_args],
)
def test_remove_header_footer(cleaner: DocumentCleaner, text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = cleaner.run(
        documents=[doc_to_clean], clean_empty_lines=False, clean_header_footer=True, clean_whitespace=False
    )[0]["documents"][0]

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


remove_empty_lines_args = [
    ("Nothing to clean, no headlines", "a\fb\nc\f", "a\fb\nc\f", None, None),
    (
        "Nothing to clean, with headlines",
        "a\f#Title\nc\n\f",
        "a\f#Title\nc\n\f",
        [{"headline": "#Title", "start_idx": 2}],
        [{"headline": "#Title", "start_idx": 2}],
    ),
    ("Single page, no headlines", "\n\na\n\n\nb\n", "\na\nb\n", None, None),
    ("Multiple pages, no headlines", "\n\na\n\n\fb\n\n\n\nc\n\n\f\f\f", "\na\n\fb\nc\n\f\f\f", None, None),
    (
        "Single page with headlines",
        "\n\n#Title\n\n\n\n#Title2",
        "\n#Title\n#Title2",
        [{"headline": "#Title", "start_idx": 2}, {"headline": "#Title2", "start_idx": 12}],
        [{"headline": "#Title", "start_idx": 1}, {"headline": "#Title2", "start_idx": 8}],
    ),
    (
        "Multi page with headlines",
        "\n\n#Title\n\n\n\n\f#Title2\n\f\n",
        "\n#Title\n\f#Title2\n\f\n",
        [{"headline": "#Title", "start_idx": 2}, {"headline": "#Title2", "start_idx": 13}],
        [{"headline": "#Title", "start_idx": 1}, {"headline": "#Title2", "start_idx": 9}],
    ),
    (
        "With multiple pages, headlines and text",
        "a\n\n#Title\n\n\n\nb c\n\f#Title2\n\f\n",
        "a\n#Title\nb c\n\f#Title2\n\f\n",
        [{"headline": "#Title", "start_idx": 3}, {"headline": "#Title2", "start_idx": 17}],
        [{"headline": "#Title", "start_idx": 2}, {"headline": "#Title2", "start_idx": 13}],
    ),
    (
        "Unsorted headlines will be sorted",
        "a\n\n#Title\n\n\n\nb c\n\f#Title2\n\f\n",
        "a\n#Title\nb c\n\f#Title2\n\f\n",
        [{"headline": "#Title2", "start_idx": 17}, {"headline": "#Title", "start_idx": 3}],
        [{"headline": "#Title", "start_idx": 2}, {"headline": "#Title2", "start_idx": 13}],
    ),
]


@pytest.mark.parametrize(
    "text,clean_text,headlines,clean_headlines",
    [args[1:] for args in remove_empty_lines_args],
    ids=[i[0] for i in remove_empty_lines_args],
)
def test_remove_empty_lines(cleaner: DocumentCleaner, text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = cleaner.run(
        documents=[doc_to_clean], clean_whitespace=False, clean_header_footer=False, clean_empty_lines=True
    )[0]["documents"][0]

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines


remove_whitespace_args = [
    ("Nothing to clean, no headlines", "a\fb\nc", "a\fb\nc", None, None),
    ("Trailing newlines and form feeds are kept, no headlines", "a\n\fb\nc\f", "a\n\fb\nc\f", None, None),
    (
        "Nothing to clean, with headlines",
        "a\f#Title\nc\f",
        "a\f#Title\nc\f",
        [{"headline": "#Title", "start_idx": 2}],
        [{"headline": "#Title", "start_idx": 2}],
    ),
    ("Single page, no headlines", " a \nb c\nd    \n   e ", "a\nb c\nd\ne", None, None),
    ("Multiple pages, no headlines", " a \f  b\nc     \f", "a\fb\nc\f", None, None),
    (
        "Single page with headlines",
        "   #Title \n#Title2   ",
        "#Title\n#Title2",
        [{"headline": "#Title", "start_idx": 3}, {"headline": "#Title2", "start_idx": 11}],
        [{"headline": "#Title", "start_idx": 0}, {"headline": "#Title2", "start_idx": 7}],
    ),
    (
        "Single page with headlines containing spaces",
        "#Title    \n    #Title2   ",
        "#Title\n#Title2",
        [{"headline": "#Title    ", "start_idx": 0}, {"headline": "    #Title2   ", "start_idx": 15}],
        [{"headline": "#Title", "start_idx": 0}, {"headline": "#Title2", "start_idx": 7}],
    ),
    (
        "Single page with headlines containing spaces at the start",
        "a\n   #Title \n#Title2   ",
        "a\n#Title\n#Title2",
        [{"headline": "   #Title ", "start_idx": 2}, {"headline": "#Title2  ", "start_idx": 13}],
        [{"headline": "#Title", "start_idx": 2}, {"headline": "#Title2", "start_idx": 9}],
    ),
    (
        "Multi page with headlines",
        "   #Title \f#Title2   ",
        "#Title\f#Title2",
        [{"headline": "#Title", "start_idx": 3}, {"headline": "#Title2", "start_idx": 11}],
        [{"headline": "#Title", "start_idx": 0}, {"headline": "#Title2", "start_idx": 7}],
    ),
    (
        "Empty page with headlines",
        "   #Title \f\f\f#Title2   ",
        "#Title\f\f\f#Title2",
        [{"headline": "#Title", "start_idx": 3}, {"headline": "#Title2", "start_idx": 13}],
        [{"headline": "#Title", "start_idx": 0}, {"headline": "#Title2", "start_idx": 9}],
    ),
    (
        "With multiple pages, headlines and text",
        " a  \n#Title \n\f d  \n #Title2 \n f",
        "a\n#Title\n\fd\n#Title2\nf",
        [{"headline": "#Title", "start_idx": 5}, {"headline": "#Title2", "start_idx": 18}],
        [{"headline": "#Title", "start_idx": 2}, {"headline": "#Title2", "start_idx": 13}],
    ),
    (
        "Unsorted headlines will be sorted",
        " a \n#Title \f d  \n #Title2 \n f",
        "a\n#Title\fd\n#Title2\nf",
        [{"headline": "#Title2", "start_idx": 18}, {"headline": "#Title", "start_idx": 4}],
        [{"headline": "#Title", "start_idx": 2}, {"headline": "#Title2", "start_idx": 11}],
    ),
    (
        "With headlines and multiple empty lines",
        "\n\n a \n#Title \n\n\n d  \n\n\n\n",
        "\n\na\n#Title\n\n\nd\n\n\n\n",
        [{"headline": "#Title", "start_idx": 6}],
        [{"headline": "#Title", "start_idx": 4}],
    ),
    (
        "With headlines and multiple empty lines/pages",
        "\n\n a \n#Title \f\f\f d  \n #Title2 \f\n\n\n\n",
        "\n\na\n#Title\f\f\fd\n#Title2\f\n\n\n\n",
        [{"headline": "#Title2", "start_idx": 22}, {"headline": "#Title", "start_idx": 6}],
        [{"headline": "#Title", "start_idx": 4}, {"headline": "#Title2", "start_idx": 15}],
    ),
]


@pytest.mark.parametrize(
    "text,clean_text,headlines,clean_headlines",
    [args[1:] for args in remove_whitespace_args],
    ids=[i[0] for i in remove_whitespace_args],
)
def test_remove_whitespace(cleaner: DocumentCleaner, text, clean_text, headlines, clean_headlines):
    doc_to_clean = Document(content=text, meta={"headlines": headlines})
    clean_doc = cleaner.run(
        documents=[doc_to_clean], clean_whitespace=True, clean_header_footer=False, clean_empty_lines=False
    )[0]["documents"][0]

    assert clean_doc.content == clean_text
    assert clean_doc.meta.get("headlines", None) == clean_headlines
