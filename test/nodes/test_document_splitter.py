import pytest
from haystack import Document
from haystack.nodes.preprocessor.splitter import DocumentSplitter

from ..conftest import SAMPLES_PATH


@pytest.fixture
def splitter():
    # Note: this are all simply fallback values.
    # Each test will call directly run providing the required input parameters.
    # If testing DocumentSplitter.__init__(), they should not use this fixture
    return DocumentSplitter(split_by="page", split_length=1)


#
# Basic validation
#


def test_init_split_by_random():
    with pytest.raises(ValueError, match="split_by must be one of"):
        DocumentSplitter(split_by="random", split_length=1)


def test_init_wrong_split_length():
    with pytest.raises(ValueError, match="split_length must be an integer > 0"):
        DocumentSplitter(split_by="page", split_length=0)
    with pytest.raises(ValueError, match="split_length must be an integer > 0"):
        DocumentSplitter(split_by="page", split_length=-1)
    with pytest.raises(ValueError, match="split_length must be an integer > 0"):
        DocumentSplitter(split_by="page", split_length=None)


def test_init_split_overlap_above_split_length():
    with pytest.raises(ValueError, match="split_length"):
        DocumentSplitter(split_by="page", split_length=1, split_overlap=10)
    with pytest.raises(ValueError, match="split_length"):
        DocumentSplitter(split_by="page", split_length=3, split_overlap=3)


def test_run_split_by_random(splitter: DocumentSplitter):
    with pytest.raises(ValueError, match="split_by"):
        splitter.run(documents=[Document(content="test")], split_by="random", split_length=1)


def test_run_use_init_params(splitter: DocumentSplitter):
    splitter.run(documents=[Document(content="test")])


def test_run_wrong_split_length(splitter: DocumentSplitter):
    with pytest.raises(ValueError, match="split_length"):
        splitter.run(documents=[Document(content="test")], split_by="page", split_length=0)
    with pytest.raises(ValueError, match="split_length"):
        splitter.run(documents=[Document(content="test")], split_by="page", split_length=-1)
    splitter.run(
        documents=[Document(content="test")], split_by="page", split_length=None  # uses the init value in this case
    )


def test_run_split_overlap_above_split_length(splitter: DocumentSplitter):
    with pytest.raises(ValueError, match="split_length"):
        splitter.run(documents=[Document(content="test")], split_by="page", split_length=1, split_overlap=10)
    with pytest.raises(ValueError, match="split_length"):
        splitter.run(documents=[Document(content="test")], split_by="page", split_length=10, split_overlap=10)


#
# Regex-based splits
#


def test_split_by_regex_no_regex_given(splitter: DocumentSplitter):
    with pytest.raises(ValueError, match="split_regex"):
        splitter.run(documents=[Document(content="test doc")], split_by="regex")


def test_split_by_something_with_regex_given(splitter: DocumentSplitter, caplog):
    splitter.run(documents=[Document(content="test doc")], split_by="page", split_regex=r"[a-z]*")
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
def test_split_by_regex_no_headlines(
    splitter: DocumentSplitter, document, expected_documents, expected_pages, length, overlap
):
    split_documents = splitter.run(
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
    splitter: DocumentSplitter,
    document,
    expected_documents,
    headlines,
    expected_headlines,
    expected_pages,
    length,
    overlap,
):
    split_documents = splitter.run(
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


def test_split_by_regex_above_max_chars_single_unit_no_headlines(splitter: DocumentSplitter):
    split_documents = splitter.run(
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


def test_split_by_regex_above_max_chars_no_overlap_no_headlines(splitter: DocumentSplitter):
    split_documents = splitter.run(
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


def test_split_by_regex_above_max_chars_with_overlap_no_headlines(splitter: DocumentSplitter):
    split_documents = splitter.run(
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


def test_split_by_regex_above_max_chars_single_unit_with_headlines(splitter: DocumentSplitter):
    split_documents = splitter.run(
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


def test_split_by_regex_above_max_chars_no_overlap_with_headlines(splitter: DocumentSplitter):
    split_documents = splitter.run(
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


def test_split_by_regex_above_max_chars_with_overlap_with_headlines(splitter: DocumentSplitter):
    split_documents = splitter.run(
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


def test_split_by_regex_above_max_chars_with_overlap_page_backtracking(splitter: DocumentSplitter):
    split_documents = splitter.run(
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
def test_split_by_page(splitter: DocumentSplitter, document, expected_documents, expected_pages, length, overlap):
    split_documents = splitter.run(
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
def test_split_by_paragraph(splitter: DocumentSplitter, document, expected_documents, expected_pages, length, overlap):
    split_documents = splitter.run(
        documents=[Document(content=document)],
        split_by="paragraph",
        split_length=length,
        split_overlap=overlap,
        split_max_chars=50,
        add_page_number=True,
    )[0]["documents"]
    assert expected_documents == [document.content for document in split_documents]
    assert expected_pages == [document.meta["page"] for document in split_documents]


# Few additional test on the paragraph split to make sure it works only when intended
split_by_word_args = [
    ("Single word", "An_extremely_long_word", ["An_extremely_long_word"], 1, 0),
    ("Few words whitespace separated", "A few words to split", ["A ", "few ", "words ", "to ", "split"], 1, 0),
    (
        "More complex whitespace",
        "   A    few \n\n words\f \f \f  to split",
        ["   ", "A    ", "few \n\n ", "words\f \f \f  ", "to ", "split"],
        1,
        0,
    ),
    (
        "Punctuation doesn't matter",
        "This sentence: (isn't) split! on tokens single-word ...",
        ["This ", "sentence: ", "(isn't) ", "split! ", "on ", "tokens ", "single-word ", "..."],
        1,
        0,
    ),
    (
        "Split length and overlap",
        "This sentence: (isn't) split! on tokens single-word ...",
        [
            "This sentence: ",
            "sentence: (isn't) ",
            "(isn't) split! ",
            "split! on ",
            "on tokens ",
            "tokens single-word ",
            "single-word ...",
        ],
        2,
        1,
    ),
]


@pytest.mark.parametrize(
    "document,expected_documents,length,overlap",
    [args[1:] for args in split_by_word_args],
    ids=[i[0] for i in split_by_word_args],
)
def test_split_by_word(splitter: DocumentSplitter, document, expected_documents, length, overlap):
    split_documents = splitter.run(
        documents=[Document(content=document)],
        split_by="word",
        split_length=length,
        split_overlap=overlap,
        split_max_chars=50,
        add_page_number=True,
    )[0]["documents"]
    assert expected_documents == [document.content for document in split_documents]


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
        "One sentence. And this... is another sentence? Yes. ",
        ["One sentence. ", "And this... is another sentence? ", "Yes. "],
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
        "Not English, Latin script with the English tokenizer",
        "Questa frase contiene una S.I.G.L.A. Questa è una prova.",
        ["Questa frase contiene una S.I.G.L.A. ", "Questa è una prova."],
    ),
    (
        "Not English, non-Latin alphabetical script with the English tokenizer",
        "Это Т.Е.С.Т. А это еще один простой тест. ",
        ["Это Т.Е.С.Т. ", "А это еще один простой тест. "],
    ),
    # Chinese can't be split properly by the English tokenizer
    (
        "Chinese text script with the English tokenizer",
        "今天不是晴天，因为下雨了。 昨天天气比较好。",
        ["今天不是晴天，因为下雨了。 昨天天气比较好。"],
        # Should be:
        # ["今天不是晴天，因为下雨了。 ", "昨天天气比较好。"]
    ),
    (
        "Gibberish that looks like a sentence",
        "✨abc✨,  d🪲ef✨gh🪲 ✨abc. ✨abc✨d🪲ef✨gh🪲!! ✨abc✨, (d🪲ef) ✨gh🪲?",
        ["✨abc✨,  d🪲ef✨gh🪲 ✨abc. ", "✨abc✨d🪲ef✨gh🪲!! ", "✨abc✨, (d🪲ef) ✨gh🪲?"],
    ),
]


@pytest.mark.parametrize(
    "document,expected_documents",
    [args[1:] for args in split_by_sentence_args],
    ids=[i[0] for i in split_by_sentence_args],
)
def test_split_by_sentence(splitter: DocumentSplitter, document, expected_documents):
    split_documents = splitter.run(
        documents=[Document(content=document)],
        split_by="sentence",
        split_length=1,
        split_overlap=0,
        split_max_chars=500,
        add_page_number=True,
    )[0]["documents"]
    assert expected_documents == [document.content for document in split_documents]


def test_split_by_sentence_with_headlines(splitter: DocumentSplitter):
    split_documents = splitter.run(
        documents=[
            Document(
                content="Title: first sentence.\nTitle2: second sentence.\fanother sentence. And Title 3, another!",
                meta={
                    "headlines": [
                        {"content": "Title", "start_idx": 0},
                        {"content": "Title2", "start_idx": 23},
                        {"content": "Title 3", "start_idx": 70},
                    ]
                },
            )
        ],
        split_by="sentence",
        split_length=1,
        split_overlap=0,
        split_max_chars=500,
        add_page_number=True,
    )[0]["documents"]
    assert [document.content for document in split_documents] == [
        "Title: first sentence.\n",
        "Title2: second sentence.\f",
        "another sentence. ",
        "And Title 3, another!",
    ]
    assert [document.meta["headlines"] for document in split_documents] == [
        [{"content": "Title", "start_idx": 0}],
        [{"content": "Title2", "start_idx": 0}],
        [],
        [{"content": "Title 3", "start_idx": 4}],
    ]


def test_split_by_sentence_with_overlap_and_headlines(splitter: DocumentSplitter):
    split_documents = splitter.run(
        documents=[
            Document(
                content="Title: first sentence.\nTitle2: second sentence.\fanother sentence. And Title 3, another!",
                meta={
                    "headlines": [
                        {"content": "Title", "start_idx": 0},
                        {"content": "Title2", "start_idx": 23},
                        {"content": "Title 3", "start_idx": 70},
                    ]
                },
            )
        ],
        split_by="sentence",
        split_length=2,
        split_overlap=1,
        split_max_chars=500,
        add_page_number=True,
    )[0]["documents"]
    assert [document.content for document in split_documents] == [
        "Title: first sentence.\nTitle2: second sentence.\f",
        "Title2: second sentence.\fanother sentence. ",
        "another sentence. And Title 3, another!",
    ]
    assert [document.meta["headlines"] for document in split_documents] == [
        [{"content": "Title", "start_idx": 0}, {"content": "Title2", "start_idx": 23}],
        [{"content": "Title2", "start_idx": 0}],
        [{"content": "Title 3", "start_idx": 22}],
    ]


split_by_token_args = [
    ("Empty string", "", [""]),
    ("Whitespace is kept", "    ", ["    "]),
    ("Single word", "test", ["test"]),
    ("Single word with whitespace", "  test    ", ["  ", "test    "]),
    ("Sentence with whitespace", " This is a test    ", [" ", "This ", "is ", "a ", "test    "]),
    ("Sentence with punctuation", " This, is a test.    ", [" ", "This", ", ", "is ", "a ", "test", ".    "]),
    (
        "Sentence with strange punctuation",
        " This!! ? is a test...!..()    ",
        [" ", "This", "!", "! ", "? ", "is ", "a ", "test", "...", "!", "..", "(", ")    "],
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
    (
        "Chinese text",
        "今天不是晴天，因为下雨了。",
        ["今天不是晴天，因为下雨了。"],
        # Should be:
        # ["今天", "不", "是", "晴", "天", "，", "因为", "下雨", "了", "。"]
    ),
    (
        "Gibberish that looks like words",
        "✨abc✨,  🪲ef✨gh🪲 ✨abc. ✨abcy✨, (d🪲ef) ✨gh🪲?",
        ["✨abc✨", ",  ", "🪲ef✨gh🪲 ", "✨abc. ", "✨abcy✨", ", ", "(", "d🪲ef", ") ", "✨gh🪲", "?"],
    ),
]


@pytest.mark.parametrize(
    "document,expected_documents", [args[1:] for args in split_by_token_args], ids=[i[0] for i in split_by_token_args]
)
def test_split_by_token(splitter: DocumentSplitter, document, expected_documents):
    split_documents = splitter.run(
        documents=[Document(content=document)],
        split_by="token",
        split_length=1,
        split_overlap=0,
        split_max_chars=500,
        add_page_number=True,
    )[0]["documents"]
    assert expected_documents == [document.content for document in split_documents]


def test_split_by_token_with_headlines(splitter: DocumentSplitter):
    split_documents = splitter.run(
        documents=[
            Document(
                content="Title: some words. Another Title! some more text.",
                meta={
                    "headlines": [{"content": "Title", "start_idx": 0}, {"content": "other Title!", "start_idx": 21}]
                },
            )
        ],
        split_by="token",
        split_length=1,
        split_overlap=0,
        split_max_chars=500,
        add_page_number=True,
    )[0]["documents"]

    for doc in split_documents:
        print(doc.content, doc.meta)

    assert [document.content for document in split_documents] == [
        "Title",
        ": ",
        "some ",
        "words. ",
        "Another ",
        "Title",
        "! ",
        "some ",
        "more ",
        "text",
        ".",
    ]
    assert [document.meta["headlines"] for document in split_documents] == [
        [{"content": "Title", "start_idx": 0}],
        [],
        [],
        [],
        [{"content": "other Title!", "start_idx": 2}],
        [],
        [],
        [],
        [],
        [],
        [],
    ]


def test_split_by_token_with_overlap_and_headlines(splitter: DocumentSplitter):
    split_documents = splitter.run(
        documents=[
            Document(
                content="Title: some words. Another Title! some more text.",
                meta={
                    "headlines": [
                        {"content": "Title", "start_idx": 0},
                        {"content": "other Title!", "start_idx": 21},
                        {"content": "Title!", "start_idx": 27},
                    ]
                },
            )
        ],
        split_by="token",
        split_length=2,
        split_overlap=1,
        split_max_chars=500,
        add_page_number=True,
    )[0]["documents"]
    assert [document.content for document in split_documents] == [
        "Title: ",
        ": some ",
        "some words. ",
        "words. Another ",
        "Another Title",
        "Title! ",
        "! some ",
        "some more ",
        "more text",
        "text.",
    ]
    assert [document.meta["headlines"] for document in split_documents] == [
        [{"content": "Title", "start_idx": 0}],
        [],
        [],
        [{"content": "other Title!", "start_idx": 9}],
        [{"content": "other Title!", "start_idx": 2}, {"content": "Title!", "start_idx": 8}],
        [{"content": "Title!", "start_idx": 0}],
        [],
        [],
        [],
        [],
    ]


#
# Char-based splits
#

split_by_character_args = [
    ("Empty string", "", [""], None, [None], [1]),
    ("Whitespace is kept", "    ", ["    "], None, [None], [1]),
    ("Below split_length", "test", ["test"], None, [None], [1]),
    ("Above split_length", "test  test", ["test  ", "test"], None, [None, None], [1, 1]),
    (
        "Multiple times above split_length",
        "1test 2test 3test 4test 5test 6test",
        ["1test ", "2test ", "3test ", "4test ", "5test ", "6test"],
        None,
        [None, None, None, None, None, None],
        [1, 1, 1, 1, 1, 1],
    ),
    (
        "Punctuation, pagefeeds and newlines are ignored",
        "1test\f2test\n3tes\n\ntest4 5test.6test",
        ["1test\f", "2test\n", "3tes\n\n", "test4 ", "5test.", "6test"],
        None,
        [None, None, None, None, None, None],
        [1, 2, 2, 2, 2, 2],
    ),
    (
        "Headlines are properly assigned",
        "1test\f2test\n3tes\n\ntest4 5test.6test",
        ["1test\f", "2test\n", "3tes\n\n", "test4 ", "5test.", "6test"],
        [{"content": "2", "start_idx": 6}, {"content": "tes\n", "start_idx": 13}, {"content": ".", "start_idx": 29}],
        [
            [],
            [{"content": "2", "start_idx": 0}],
            [{"content": "tes\n", "start_idx": 1}],
            [],
            [{"content": ".", "start_idx": 5}],
            [],
        ],
        [1, 2, 2, 2, 2, 2],
    ),
]


@pytest.mark.parametrize(
    "document,expected_documents,headlines,expected_headlines,expected_pages",
    [args[1:] for args in split_by_character_args],
    ids=[i[0] for i in split_by_character_args],
)
def test_split_by_character(
    splitter: DocumentSplitter, document, expected_documents, headlines, expected_headlines, expected_pages
):
    split_documents = splitter.run(
        documents=[Document(content=document, meta={"headlines": headlines})],
        split_by="character",
        split_length=6,
        split_overlap=0,
        split_max_chars=500,
        add_page_number=True,
    )[0]["documents"]
    assert expected_documents == [document.content for document in split_documents]
    assert expected_headlines == [document.meta["headlines"] for document in split_documents]
    assert expected_pages == [document.meta["page"] for document in split_documents]
