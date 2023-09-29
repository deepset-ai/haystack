import pandas as pd
import pytest

from haystack.utils.squad_data import SquadData
from haystack.utils.augment_squad import augment_squad
from haystack.schema import Document, Label, Answer


def test_squad_augmentation(samples_path):
    input_ = samples_path / "squad" / "tiny.json"
    output = samples_path / "squad" / "tiny_augmented.json"
    glove_path = samples_path / "glove" / "tiny.txt"  # dummy glove file, will not even be use when augmenting tiny.json
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


@pytest.mark.unit
def test_squad_data_converts_df_to_data():
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


@pytest.mark.unit
def test_squad_data_converts_data_to_df():
    data = [
        {
            "title": "title",
            "paragraphs": [
                {
                    "context": "context",
                    "document_id": "document_id",
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
    expected_result = pd.DataFrame(
        [["title", "context", "question", "id", "answer", 1, False, "document_id"]],
        columns=["title", "context", "question", "id", "answer_text", "answer_start", "is_impossible", "document_id"],
    )
    result = SquadData.to_df(data)
    assert result.equals(expected_result)


def test_to_label_object():
    squad_data_list = [
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
                        },
                        {
                            "question": "another question",
                            "id": "another_id",
                            "answers": [{"text": "this is the response", "answer_start": 1}],
                            "is_impossible": False,
                        },
                    ],
                },
                {
                    "context": "the second paragraph context",
                    "qas": [
                        {
                            "question": "the third question",
                            "id": "id_3",
                            "answers": [{"text": "this is another response", "answer_start": 1}],
                            "is_impossible": False,
                        },
                        {
                            "question": "the forth question",
                            "id": "id_4",
                            "answers": [{"text": "this is the response", "answer_start": 1}],
                            "is_impossible": False,
                        },
                    ],
                },
            ],
        }
    ]
    squad_data = SquadData(squad_data=squad_data_list)
    answer_type = "generative"
    labels = squad_data.to_label_objs(answer_type=answer_type)
    for label, expected_question in zip(labels, squad_data.df.iterrows()):
        expected_question = expected_question[1]
        assert isinstance(label, Label)
        assert isinstance(label.document, Document)
        assert isinstance(label.answer, Answer)
        assert label.query == expected_question["question"]
        assert label.document.content == expected_question.context
        assert label.document.id == expected_question.document_id
        assert label.id == expected_question.id
        assert label.answer.answer == expected_question.answer_text
