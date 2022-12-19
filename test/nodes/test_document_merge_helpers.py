import pytest

from haystack.nodes.preprocessor.merge_helpers import (
    common_values,
    merge_headlines,
    make_merge_groups,
    validate_unit_boundaries,
)
from haystack.nodes.preprocessor.split_helpers import split_by_separator


common_values_testargs = [
    ("empty list", [], {}),
    ("one empty dict", [{}], {}),
    ("one dict", [{"first": "1", "second": "2"}], {"first": "1", "second": "2"}),
    ("one empty dict, one populated dicts", [{}, {"first": "1", "second": "2"}], {}),
    (
        "two identical flat dicts",
        [{"first": "1", "second": "2"}, {"first": "1", "second": "2"}],
        {"first": "1", "second": "2"},
    ),
    (
        "two identical flat dicts with keys shuffled",
        [{"first": "1", "second": "2"}, {"second": "2", "first": "1"}],
        {"first": "1", "second": "2"},
    ),
    (
        "two identical nested dicts",
        [{"first": {"second": "2"}}, {"first": {"second": "2"}}],
        {"first": {"second": "2"}},
    ),
    (
        "two identical very nested dicts",
        [{"first": {"second": {"third": "2"}}}, {"first": {"second": {"third": "2"}}}],
        {"first": {"second": {"third": "2"}}},
    ),
    ("two identical dicts with lists", [{"first": [1, 2, 3]}, {"first": [1, 2, 3]}], {"first": [1, 2, 3]}),
    ("two identical dicts, keys are not strings", [{1: 1}, {1: 1}], {1: 1}),
    ("same keys, different values", [{"first": "1"}, {"first": "2"}], {}),
    ("different keys, same values", [{"first": "1"}, {"second": "1"}], {}),
    ("many identical dicts, one differs", [{"first": "1"}] * 10 + [{"first": "2"}], {}),
    ("one dict has one additional key", [{"first": "1"}] * 10 + [{"first": "1", "second": "2"}], {"first": "1"}),
    (
        "all dicts have the same two key-values except one that have a different value",
        [{"first": "1", "second": "2"}] * 10 + [{"first": "1", "second": "3"}],
        {"first": "1"},
    ),
    (
        "all dicts have the same two key-values except one that have a different list value",
        [{"first": "1", "second": [2, 2, 2]}] * 10 + [{"first": "1", "second": [2, 2, 3]}],
        {"first": "1"},
    ),
    (
        "all dicts have the same two key-values except one that have a different key",
        [{"first": "1", "second": "2"}] * 10 + [{"first": "1", "third": "2"}],
        {"first": "1"},
    ),
    (
        "all dicts have the same key-values except one that have an extra key",
        [{"first": "1"}] * 10 + [{"first": "1", "second": "2"}],
        {"first": "1"},
    ),
    (
        "all dicts have the same two key-values except one that have one less key",
        [{"first": "1", "second": "2"}] * 10 + [{"second": "2"}],
        {"second": "2"},
    ),
    (
        "Same top-level keys, one nested value differs",
        [{"first": {"second": "2", "third": "3"}}, {"first": {"second": "2", "third": "4"}}],
        {"first": {"second": "2"}},
    ),
    (
        "Same top-level keys, one deep nested value differs",
        [
            {"first": {"second": {"third": "3", "fourth": "4"}, "sixth": "6"}, "seventh": "7"},
            {"first": {"second": {"third": "3", "fourth": "5"}, "sixth": "6"}, "seventh": "7"},
        ],
        {"first": {"second": {"third": "3"}, "sixth": "6"}, "seventh": "7"},
    ),
    (
        "Same top-level keys, one deep nested list differs",
        [
            {"first": {"second": {"third": "3", "fourth": [4]}, "sixth": "6"}, "seventh": "7"},
            {"first": {"second": {"third": "3", "fourth": [4, 5]}, "sixth": "6"}, "seventh": "7"},
        ],
        {"first": {"second": {"third": "3"}, "sixth": "6"}, "seventh": "7"},
    ),
    (
        "Same top-level keys, some deep nested key differs",
        [
            {"first": {"second": {"third": "3", "fourth": "4"}, "sixth": "6"}, "seventh": "7"},
            {"first": {"second": {"third": "3", "fifth": "5"}, "sixth": "6"}, "seventh": "7"},
        ],
        {"first": {"second": {"third": "3"}, "sixth": "6"}, "seventh": "7"},
    ),
]


@pytest.mark.parametrize(
    "dictionaries, expected_output",
    [args[1:] for args in common_values_testargs],
    ids=[i[0] for i in common_values_testargs],
)
def test_common_values_no_exclude(dictionaries, expected_output):
    merged_dictionary = common_values(list_of_dicts=dictionaries)
    assert merged_dictionary == expected_output


common_values_with_exclude_testargs = [
    ("one empty dict", ["first"], [{}], {}),
    ("the only dict with only the key to exclude", ["first"], [{"first": "1"}], {}),
    ("all dicts has only the key to exclude", ["first"], [{"first": "1"}, {"first": "1"}], {}),
    (
        "all dicts has only keys to exclude",
        ["first", "second"],
        [{"first": "1", "second": "2"}, {"first": "1", "second": "2"}],
        {},
    ),
    (
        "all dicts has only keys to exclude and keys that don't match",
        ["first", "second"],
        [{"first": "1", "second": "2", "third": "3"}, {"first": "1", "second": "2", "third": 3}],
        {},
    ),
    (
        "all dicts has only keys to exclude and keys that don't match",
        ["first", "second"],
        [{"first": "1", "second": "2", "third": "3"}, {"first": "1", "second": "2", "third": 3}],
        {},
    ),
    (
        "key to exclude would be excluded anyway",
        ["first", "second"],
        [{"first": "1"}, {"first": "2", "second": "2"}],
        {},
    ),
    (
        "key to exclude would be excluded anyway and others don't match",
        ["first", "second"],
        [{"first": "1", "third": "3", "fourth": "4"}, {"first": "2", "second": "2", "third": "4"}],
        {},
    ),
    (
        "key to exclude would be excluded anyway and others match",
        ["first", "second"],
        [{"first": "1", "third": "3"}, {"first": "2", "second": "2", "third": "3"}],
        {"third": "3"},
    ),
    (
        "key to exclude are only top-level",
        ["second"],
        [{"first": {"second": "2"}, "second": "2"}, {"first": {"second": "2"}, "second": "2"}],
        {"first": {"second": "2"}},
    ),
    (
        "key to exclude are dropped even if they contain matching dicts",
        ["first"],
        [
            {"first": {"first": "1", "second": "2"}, "second": "2"},
            {"first": {"first": "1", "second": "2"}, "second": "2"},
        ],
        {"second": "2"},
    ),
]


@pytest.mark.parametrize(
    "exclude, dictionaries, expected_output",
    [args[1:] for args in common_values_with_exclude_testargs],
    ids=[i[0] for i in common_values_with_exclude_testargs],
)
def test_common_values_with_exclude(exclude, dictionaries, expected_output):
    merged_dictionary = common_values(list_of_dicts=dictionaries, exclude=exclude)
    assert merged_dictionary == expected_output


merge_headlines_testargs = [
    ("empty sources list", [], []),
    ("one source with no headlines", [("Title: the content", [])], []),
    (
        "one source with one headlines",
        [("Title: the content", [{"content": "Title", "start_idx": 0}])],
        [{"content": "Title", "start_idx": 0}],
    ),
    (
        "one source with some headlines",
        [
            (
                "Title: the content. Title 2: more content",
                [{"content": "Title", "start_idx": 0}, {"content": "Title 2", "start_idx": 20}],
            )
        ],
        [{"content": "Title", "start_idx": 0}, {"content": "Title 2", "start_idx": 20}],
    ),
    (
        "two source with no headlines",
        [("Title: the content. Title 2: more content", []), ("Title3: the content. Title 4: more content", [])],
        [],
    ),
    (
        "two sources, only first has headlines",
        [
            (
                "Title: the content. Title 2: more content",
                [{"content": "Title", "start_idx": 0}, {"content": "Title 2", "start_idx": 20}],
            ),
            ("Title3: the content. Title 4: more content", []),
        ],
        [{"content": "Title", "start_idx": 0}, {"content": "Title 2", "start_idx": 20}],
    ),
    (
        "two sources, only second has headlines",
        [
            ("Title: the content. Title 2: more content", []),
            (
                "Title3: the content. Title 4: more content",
                [{"content": "Title3", "start_idx": 0}, {"content": "Title4", "start_idx": 21}],
            ),
        ],
        [{"content": "Title3", "start_idx": 44}, {"content": "Title4", "start_idx": 65}],
    ),
    (
        "two sources, both have headlines",
        [
            (
                "Title: the content. Title 2: more content",
                [{"content": "Title", "start_idx": 0}, {"content": "Title 2", "start_idx": 20}],
            ),
            (
                "Title3: the content. Title 4: more content",
                [{"content": "Title3", "start_idx": 0}, {"content": "Title4", "start_idx": 21}],
            ),
        ],
        [
            {"content": "Title", "start_idx": 0},
            {"content": "Title 2", "start_idx": 20},
            {"content": "Title3", "start_idx": 44},
            {"content": "Title4", "start_idx": 65},
        ],
    ),
    (
        "headlines are NOT validated",
        [
            ("Title: the content", [{"content": "not in the text", "start_idx": 3}]),
            ("Title2: more content", [{"content": "not in the text either", "start_idx": 5}]),
        ],
        [{"content": "not in the text", "start_idx": 3}, {"content": "not in the text either", "start_idx": 26}],
    ),
]


@pytest.mark.parametrize(
    "sources, expected_output",
    [args[1:] for args in merge_headlines_testargs],
    ids=[i[0] for i in merge_headlines_testargs],
)
def test_merge_headlines(sources, expected_output):
    merged_dictionary = merge_headlines(sources=sources, separator=" - ")
    assert merged_dictionary == expected_output


def test_validate_unit_boundaries_without_tokens_or_maxchars():
    merged_dictionary = validate_unit_boundaries(contents=["a" * 10000, "bb " * 50], max_chars=0)
    assert merged_dictionary == [("a" * 10000, 0), ("bb " * 50, 0)]


validate_unit_boundaries_without_tokens_testargs = [
    ("no contents", [], []),
    ("one content", ["text"], [("text", 0)]),
    ("one longer content", ["this is a test"], [("this is a test", 0)]),
    ("some contents", ["text content", "additional content"], [("text content", 0), ("additional content", 0)]),
    ("single content longer than max_chars", ["a" * 30], [("a" * 20, 0), ("a" * 10, 0)]),
    (
        "a few contents longer than max_chars",
        ["a" * 30, "b" * 25],
        [("a" * 20, 0), ("a" * 10, 0), ("b" * 20, 0), ("b" * 5, 0)],
    ),
    (
        "a few contents several times longer than max_chars",
        ["a" * 50, "b" * 65],
        [("a" * 20, 0), ("a" * 20, 0), ("a" * 10, 0), ("b" * 20, 0), ("b" * 20, 0), ("b" * 20, 0), ("b" * 5, 0)],
    ),
]


@pytest.mark.parametrize(
    "contents, expected_output",
    [args[1:] for args in validate_unit_boundaries_without_tokens_testargs],
    ids=[i[0] for i in validate_unit_boundaries_without_tokens_testargs],
)
def test_validate_unit_boundaries_without_tokens(contents, expected_output):
    merged_dictionary = validate_unit_boundaries(contents=contents, max_chars=20)
    assert merged_dictionary == expected_output


validate_unit_boundaries_with_tokens_testargs = [
    ("no contents nor tokens", [], []),
    ("one content, one token", ["text"], [("text", 1)]),
    ("one content, a few tokens", ["this is a test"], [("this is a test", 4)]),
    ("some contents", ["text content", "additional content"], [("text content", 2), ("additional content", 2)]),
    (
        "some contents, different lengths",
        ["text content", "additional_content"],
        [("text content", 2), ("additional_content", 1)],
    ),
    ("single content longer than max_tokens", ["aa bb cc dd ee ff gg"], [("aa bb cc dd ee ", 5), ("ff gg", 2)]),
    (
        "one content longer than max_tokens, others shorter",
        ["aa bb", "cc dd e f g hh iii", "j kk lll"],
        [("aa bb", 2), ("cc dd e f g ", 5), ("hh iii", 2), ("j kk lll", 3)],
    ),
    (
        "few consecutive contents longer than max_tokens",
        ["aa bb", "cc dd e f g hh iii", "j kk lll m nn ooo p q"],
        [("aa bb", 2), ("cc dd e f g ", 5), ("hh iii", 2), ("j kk lll m nn ", 5), ("ooo p q", 3)],
    ),
    (
        "content several times longer than max_tokens",
        ["aa bb cc dd e f g hh iii j kk lll m nn ooo p q"],
        [("aa bb cc dd e ", 5), ("f g hh iii j ", 5), ("kk lll m nn ooo ", 5), ("p q", 2)],
    ),
    (
        "content several times longer than max_tokens plus shorter ones",
        ["11 2", "aa bb cc dd e f g hh iii j kk lll m nn ooo p q", "r ss t"],
        [("11 2", 2), ("aa bb cc dd e ", 5), ("f g hh iii j ", 5), ("kk lll m nn ooo ", 5), ("p q", 2), ("r ss t", 3)],
    ),
    (
        "many contents several times longer than max_tokens, plus shorter ones",
        [
            "11 2",
            "aa bb cc dd e f g hh iii j kk lll m nn ooo p q",
            "aa2 bb2 cc2 dd2 e2 f2 g2 hh2 iii2 j2 kk2 lll2 m2 nn2 ooo2 p2 q2",
            "r ss t",
        ],
        [
            ("11 2", 2),
            ("aa bb cc dd e ", 5),
            ("f g hh iii j ", 5),
            ("kk lll m nn ooo ", 5),
            ("p q", 2),
            ("aa2 bb2 cc2 dd2 e2 ", 5),
            ("f2 g2 hh2 iii2 j2 ", 5),
            ("kk2 lll2 m2 nn2 ooo2 ", 5),
            ("p2 q2", 2),
            ("r ss t", 3),
        ],
    ),
    ("single content with only one token longer than max_chars", ["a" * 30], [("a" * 20, 1), ("a" * 10, 1)]),
    (
        "single content with one token longer than max_chars and a few others",
        ["11 " + "a" * 30 + " 2222 33 4"],
        [("11 " + "a" * 17, 2), ("a" * 13 + " 2222 3", 3), ("3 4", 2)],
    ),
    (
        "single content with two tokens both longer than max_chars",
        ["a" * 30 + " " + "b" * 25],
        [("a" * 20, 1), ("a" * 10 + " " + "b" * 9, 2), ("b" * 16, 1)],
    ),
    (
        "a few contents with only one token longer than max_chars",
        ["a" * 30, "b" * 25],
        [("a" * 20, 1), ("a" * 10, 1), ("b" * 20, 1), ("b" * 5, 1)],
    ),
    (
        "a few contents with only one token longer than max_chars and shorter tokens",
        ["11 " + "a" * 30 + " 22 33333 4", "b" * 25 + " 12345 67890 123"],
        [
            ("11 " + "a" * 17, 2),
            ("a" * 13 + " 22 333", 3),
            ("33 4", 2),
            ("b" * 20, 1),
            ("b" * 5 + " 12345 67890 12", 4),
            ("3", 1),
        ],
    ),
    (
        "a few contents with only one token several times longer than max_chars and shorter tokens",
        ["11 " + "a" * 50 + " 22 33333 4", "b" * 65 + " 12345 67890 123 "],
        [
            ("11 " + "a" * 17, 2),
            ("a" * 20, 1),
            ("a" * 13 + " 22 333", 3),
            ("33 4", 2),
            ("b" * 20, 1),
            ("b" * 20, 1),
            ("b" * 20, 1),
            ("b" * 5 + " 12345 67890 12", 4),
            ("3 ", 1),
        ],
    ),
    (
        "one content passing both max_tokens and max_chars, max_tokens encountered first",
        [" b" * 23],
        [(" b b b b b ", 5)] + [("b " * 5, 5)] * 3 + [("b b b", 3)],
    ),
    (
        "two contents passing both max_tokens and max_chars, max_tokens encountered first",
        [" b" * 23, "a " * 50],
        [(" b b b b b ", 5)] + [("b " * 5, 5)] * 3 + [("b b b", 3)] + [("a " * 5, 5)] * 10,
    ),
    # (
    #     "one content passing both max_tokens and max_chars, max_chars encountered first",
    #     ["  bbbbbbbb" * 10],  # 10 chars
    #     [("  bbbbbbbb" * 3, 3)] * 3 + [("  bbbbbbbb", 1)]
    # ),
    (
        "two contents passing both max_tokens and max_chars, max_chars encountered first",
        ["aaaaaaaaa " * 10],
        [("aaaaaaaaa " * 2, 2)] * 5,
    ),
]


@pytest.mark.parametrize(
    "contents, expected_output",
    [args[1:] for args in validate_unit_boundaries_with_tokens_testargs],
    ids=[i[0] for i in validate_unit_boundaries_with_tokens_testargs],
)
def test_validate_unit_boundaries_with_tokens(contents, expected_output):
    tokens = []
    for content in contents:
        tokens += split_by_separator(text=content, separator=" ")

    merged_dictionary = validate_unit_boundaries(contents=contents, max_chars=20, max_tokens=5, tokens=tokens)
    assert merged_dictionary == expected_output


def test_validate_unit_boundaries_missing_tokens():
    with pytest.raises(ValueError, match="error occurred during tokenization"):
        validate_unit_boundaries(contents=["text content"], max_chars=50, max_tokens=10, tokens=["text"])


def test_validate_unit_boundaries_excess_tokens():
    with pytest.raises(ValueError, match="error occurred during tokenization"):
        validate_unit_boundaries(contents=[], max_chars=50, max_tokens=10, tokens=["excess ", "tokens"])
    with pytest.raises(ValueError, match="This is a bug with tokenization"):
        validate_unit_boundaries(
            contents=["text content"], max_chars=50, max_tokens=10, tokens=["text ", "content", "excess ", "tokens"]
        )
