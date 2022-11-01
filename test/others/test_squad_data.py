import pandas as pd
from haystack.utils.squad_data import SquadData
from haystack.utils.augment_squad import augment_squad
from ..conftest import SAMPLES_PATH
from haystack.schema import Document, Label, Answer


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
                        }
                    ],
                }
            ],
        }
    ]
    squad_data = SquadData(squad_data=squad_data_list)
    answer_type = "generative"
    labels = squad_data.to_label_objs(answer_type=answer_type)
    expected_paragraphs = [data.get("paragraphs") for data in squad_data_list]
    for label, expected_data in zip(labels, expected_paragraphs):
        assert isinstance(label, Label)
        assert isinstance(label.document, Document)
        assert isinstance(label.answer, Answer)
        label.query = expected_data.get("qas")[0].get("question")
        assert label.document.content == expected_data["context"]
        assert label.question == expected_data["paragraphs"][0]["qas"][0]["question"]

    assert False
    # question is string and equal question
    # answer is an answer object with recodr and asnwertype.
    # check the document is of type document.
    # check is label is instance of label
