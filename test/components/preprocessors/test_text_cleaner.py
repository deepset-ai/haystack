# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline
from haystack.components.evaluators import AnswerExactMatchEvaluator
from haystack.components.preprocessors import TextCleaner
from haystack.core.serialization import component_from_dict, component_to_dict


def test_init_default():
    cleaner = TextCleaner()
    assert cleaner._remove_regexps is None
    assert not cleaner._convert_to_lowercase
    assert not cleaner._remove_punctuation
    assert not cleaner._remove_numbers
    assert not cleaner._remove_extra_whitespaces
    assert cleaner._regex is None
    assert cleaner._translator is None


def test_run():
    cleaner = TextCleaner()
    texts = ["Some text", "Some other text", "Yet another text"]
    result = cleaner.run(texts=texts)
    assert len(result) == 1
    assert result["texts"] == texts


def test_run_with_empty_inputs():
    cleaner = TextCleaner()
    result = cleaner.run(texts=[])
    assert len(result) == 1
    assert result["texts"] == []


def test_run_rejects_invalid_inputs():
    cleaner = TextCleaner()

    with pytest.raises(TypeError):
        cleaner.run(texts=["Hello, World!", None, 42, ""])  # type: ignore[list-item]

    with pytest.raises(TypeError):
        cleaner.run(texts=15)  # type: ignore[arg-type]


def test_run_with_regex():
    cleaner = TextCleaner(remove_regexps=[r"\d+"])
    result = cleaner.run(texts=["Open123 Source", "HaystackAI"])
    assert len(result) == 1
    assert result["texts"] == ["Open Source", "HaystackAI"]


def test_run_with_multiple_regexps():
    cleaner = TextCleaner(remove_regexps=[r"\d+", r"[^\w\s]"])
    result = cleaner.run(texts=["Open123! Source", "Haystack.AI"])
    assert len(result) == 1
    assert result["texts"] == ["Open Source", "HaystackAI"]


def test_run_with_convert_to_lowercase():
    cleaner = TextCleaner(convert_to_lowercase=True)
    result = cleaner.run(texts=["Open123! Source", "Haystack.AI"])
    assert len(result) == 1
    assert result["texts"] == ["open123! source", "haystack.ai"]


def test_run_with_remove_punctuation():
    cleaner = TextCleaner(remove_punctuation=True)
    result = cleaner.run(texts=["Open123! Source", "Haystack.AI"])
    assert len(result) == 1
    assert result["texts"] == ["Open123 Source", "HaystackAI"]


def test_run_with_remove_numbers():
    cleaner = TextCleaner(remove_numbers=True)
    result = cleaner.run(texts=["Open123! Source", "Haystack.AI"])
    assert len(result) == 1
    assert result["texts"] == ["Open! Source", "Haystack.AI"]


def test_run_with_multiple_parameters():
    cleaner = TextCleaner(
        remove_regexps=[r"\d+", r"[^\w\s]"], convert_to_lowercase=True, remove_punctuation=True, remove_numbers=True
    )
    result = cleaner.run(texts=["Open%123. !$Source", "Haystack.AI##"])
    assert len(result) == 1
    assert result["texts"] == ["open source", "haystackai"]


def test_serialization_keeps_init_parameters():
    cleaner = TextCleaner(
        remove_regexps=["World"], convert_to_lowercase=True, remove_punctuation=True, remove_numbers=True
    )
    pipeline = Pipeline()
    pipeline.add_component("cleaner", cleaner)
    assert pipeline.to_dict()["components"]["cleaner"]["init_parameters"] == {
        "remove_regexps": ["World"],
        "convert_to_lowercase": True,
        "remove_punctuation": True,
        "remove_numbers": True,
        "remove_extra_whitespaces": False,
    }

    restored = Pipeline.loads(pipeline.dumps()).get_component("cleaner")

    texts = ["Hello, World 42!"]
    assert restored.run(texts=texts) == cleaner.run(texts=texts) == {"texts": ["hello  "]}


def test_init_with_remove_extra_whitespaces():
    cleaner = TextCleaner(remove_extra_whitespaces=True)
    assert cleaner._remove_extra_whitespaces
    assert cleaner._regex is None
    assert cleaner._translator is None


def test_run_keeps_whitespace_by_default():
    cleaner = TextCleaner(remove_punctuation=True)
    result = cleaner.run(texts=["Rome - Italy", "  Paris  ", "Line one\n\nLine two"])
    assert result["texts"] == ["Rome  Italy", "  Paris  ", "Line one\n\nLine two"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Rome  Italy", "Rome Italy"),
        ("  Paris  ", "Paris"),
        ("\tleading tab", "leading tab"),
        ("trailing newline\n", "trailing newline"),
        ("Line one\n\nLine two", "Line one Line two"),
        ("tabs\tand\r\nmixed \x0b\x0c whitespace", "tabs and mixed whitespace"),
        ("non\u00a0breaking\u2003space", "non breaking space"),
        ("already clean", "already clean"),
        ("single", "single"),
        ("", ""),
        ("   ", ""),
        ("\n\t\r ", ""),
    ],
)
def test_run_with_remove_extra_whitespaces(text, expected):
    cleaner = TextCleaner(remove_extra_whitespaces=True)
    assert cleaner.run(texts=[text]) == {"texts": [expected]}


def test_run_with_remove_extra_whitespaces_multiple_texts_keeps_order_and_length():
    cleaner = TextCleaner(remove_extra_whitespaces=True)
    texts = ["  a  b  ", "", "c", " d\ne "]
    result = cleaner.run(texts=texts)
    assert result["texts"] == ["a b", "", "c", "d e"]
    assert texts == ["  a  b  ", "", "c", " d\ne "]


def test_run_with_remove_extra_whitespaces_after_remove_punctuation():
    cleaner = TextCleaner(remove_punctuation=True, remove_extra_whitespaces=True)
    result = cleaner.run(texts=["Rome - Italy", "Hello , World !", "...  Paris  ..."])
    assert result["texts"] == ["Rome Italy", "Hello World", "Paris"]


def test_run_with_remove_extra_whitespaces_after_remove_numbers():
    cleaner = TextCleaner(remove_numbers=True, remove_extra_whitespaces=True)
    result = cleaner.run(texts=["Room 101 is open", "42 apples"])
    assert result["texts"] == ["Room is open", "apples"]


def test_run_with_remove_extra_whitespaces_after_remove_regexps():
    cleaner = TextCleaner(remove_regexps=[r"\[\d+\]"], remove_extra_whitespaces=True)
    result = cleaner.run(texts=["Paris [1] is the capital [2] of France"])
    assert result["texts"] == ["Paris is the capital of France"]


def test_run_with_all_parameters():
    cleaner = TextCleaner(
        remove_regexps=[r"\bthe\b"],
        convert_to_lowercase=True,
        remove_punctuation=True,
        remove_numbers=True,
        remove_extra_whitespaces=True,
    )
    result = cleaner.run(texts=["  The answer is: 42 -- THE Eiffel Tower!  "])
    assert result["texts"] == ["answer is eiffel tower"]


def test_run_with_remove_extra_whitespaces_is_idempotent():
    cleaner = TextCleaner(remove_punctuation=True, remove_extra_whitespaces=True)
    once = cleaner.run(texts=["  Rome -  Italy\n"])["texts"]
    twice = cleaner.run(texts=once)["texts"]
    assert once == twice == ["Rome Italy"]


def test_run_with_remove_extra_whitespaces_rejects_invalid_inputs():
    cleaner = TextCleaner(remove_extra_whitespaces=True)

    with pytest.raises(TypeError):
        cleaner.run(texts=["  ok  ", None])  # type: ignore[list-item]

    with pytest.raises(TypeError):
        cleaner.run(texts="  not a list  ")  # type: ignore[arg-type]


def test_remove_extra_whitespaces_fixes_exact_match_evaluation():
    evaluator = AnswerExactMatchEvaluator()
    ground_truth = ["Paris", "Rome Italy"]
    predicted = ["  Paris  ", "Rome - Italy"]

    without_option = TextCleaner(remove_punctuation=True)
    result = evaluator.run(
        ground_truth_answers=without_option.run(texts=ground_truth)["texts"],
        predicted_answers=without_option.run(texts=predicted)["texts"],
    )
    assert result["score"] == 0.0

    with_option = TextCleaner(remove_punctuation=True, remove_extra_whitespaces=True)
    result = evaluator.run(
        ground_truth_answers=with_option.run(texts=ground_truth)["texts"],
        predicted_answers=with_option.run(texts=predicted)["texts"],
    )
    assert result["individual_scores"] == [1, 1]
    assert result["score"] == 1.0


def test_to_dict_with_remove_extra_whitespaces():
    cleaner = TextCleaner(remove_extra_whitespaces=True)
    assert component_to_dict(cleaner, "cleaner") == {
        "type": "haystack.components.preprocessors.text_cleaner.TextCleaner",
        "init_parameters": {
            "remove_regexps": None,
            "convert_to_lowercase": False,
            "remove_punctuation": False,
            "remove_numbers": False,
            "remove_extra_whitespaces": True,
        },
    }


def test_from_dict_with_remove_extra_whitespaces():
    data = {
        "type": "haystack.components.preprocessors.text_cleaner.TextCleaner",
        "init_parameters": {"remove_punctuation": True, "remove_extra_whitespaces": True},
    }
    cleaner = component_from_dict(TextCleaner, data, "cleaner")
    assert cleaner._remove_extra_whitespaces
    assert cleaner.run(texts=["Rome - Italy"]) == {"texts": ["Rome Italy"]}


def test_from_dict_without_remove_extra_whitespaces_keeps_old_behavior():
    # Pipelines serialized before this parameter existed don't have the key.
    data = {
        "type": "haystack.components.preprocessors.text_cleaner.TextCleaner",
        "init_parameters": {
            "remove_regexps": None,
            "convert_to_lowercase": False,
            "remove_punctuation": True,
            "remove_numbers": False,
        },
    }
    cleaner = component_from_dict(TextCleaner, data, "cleaner")
    assert not cleaner._remove_extra_whitespaces
    assert cleaner.run(texts=["Rome - Italy"]) == {"texts": ["Rome  Italy"]}


def test_pipeline_serialization_roundtrip_with_remove_extra_whitespaces():
    pipeline = Pipeline()
    pipeline.add_component("cleaner", TextCleaner(remove_punctuation=True, remove_extra_whitespaces=True))

    init_parameters = pipeline.to_dict()["components"]["cleaner"]["init_parameters"]
    assert init_parameters["remove_extra_whitespaces"] is True

    restored = Pipeline.loads(pipeline.dumps())
    result = restored.run({"cleaner": {"texts": ["  Rome - Italy  "]}})
    assert result == {"cleaner": {"texts": ["Rome Italy"]}}
