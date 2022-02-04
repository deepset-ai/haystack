import pytest
import pandas as pd
from pathlib import Path

from haystack.utils.preprocessing import convert_files_to_dicts, tika_convert_files_to_dicts
from haystack.utils.cleaning import clean_wiki_text
from haystack.utils.augment_squad import augment_squad
from haystack.utils.squad_data import SquadData

from conftest import SAMPLES_PATH


def test_convert_files_to_dicts():
    documents = convert_files_to_dicts(
        dir_path=(SAMPLES_PATH).absolute(), clean_func=clean_wiki_text, split_paragraphs=True
    )
    assert documents and len(documents) > 0


@pytest.mark.tika
def test_tika_convert_files_to_dicts():
    documents = tika_convert_files_to_dicts(dir_path=SAMPLES_PATH, clean_func=clean_wiki_text, split_paragraphs=True)
    assert documents and len(documents) > 0


def test_squad_augmentation():
    input_ = SAMPLES_PATH / "squad" / "tiny.json"
    output = SAMPLES_PATH / "squad" / "tiny_augmented.json"
    glove_path = SAMPLES_PATH / "glove" / "tiny.txt"  # dummy glove file, will not even be use when augmenting tiny.json
    multiplication_factor = 5
    augment_squad(
        model="distilbert-base-uncased",
        tokenizer="distilbert-base-uncased",
        squad_path=input_,
        output_path=output,
        glove_path=glove_path,
        multiplication_factor=multiplication_factor,
    )
    original_squad = SquadData.from_file(input_)
    augmented_squad = SquadData.from_file(output)
    assert original_squad.count(unit="paragraph") == augmented_squad.count(unit="paragraph") * multiplication_factor


def test_squad_to_df():
    df = pd.DataFrame(
        [["title", "context", "question", "id", "answer", 1, False]],
        columns=["title", "context", "question", "id", "answer_text", "answer_start", "is_impossible"],
    )

    expected_result = [
        {
            "title": "title",
            "paragraphs": [
                {
                    "context": "context",
                    "qas": [
                        {
                            "question": "question",
                            "id": "id",
                            "answers": [{"text": "answer", "answer_start": 1}],
                            "is_impossible": False,
                        }
                    ],
                }
            ],
        }
    ]

    result = SquadData.df_to_data(df)

    assert result == expected_result
